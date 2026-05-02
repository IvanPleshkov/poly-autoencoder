"""
BEIR retrieval evaluation: 4-way comparison at the same per-vector budget.

Compares retrieval quality (NDCG@10, Recall@10) of:
  - raw     : full-dim embedding from the model (the gold ceiling)
  - matryo  : top-d slice of the embedding (only valid for matryoshka models)
  - pca     : PCA top-d projection, retrieve in d-space
  - poly    : polynomial autoencoder (PCA top-d -> quadratic lift -> ridge),
              retrieve on reconstructed V_hat in D-space

The poly autoencoder is fit transductively on the corpus (no labels), then
applied to the queries. This matches the production setting: the operator
fits the AE once on the index they want to compress.

Usage:
    python beir_eval.py --model arctic-embed-m-v2.0 --dataset scifact --d 256
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from poly_pca import (
    fit_pca,
    project,
    polynomial_lift,
    fit_ridge,
    lift_dim,
)


# ============================================================ BEIR datasets

BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

BEIR_DATASETS = {
    # name      -> (url, default split)
    "scifact":  (f"{BEIR_BASE}/scifact.zip",  "test"),
    "nfcorpus": (f"{BEIR_BASE}/nfcorpus.zip", "test"),
    "arguana":  (f"{BEIR_BASE}/arguana.zip",  "test"),
    "fiqa":     (f"{BEIR_BASE}/fiqa.zip",     "test"),
    "trec-covid": (f"{BEIR_BASE}/trec-covid.zip", "test"),
}

DATA_DIR = Path(__file__).parent / "beir_data"
CACHE_DIR = Path(__file__).parent / "cache"


def _download(url: str, dest: Path) -> None:
    print(f"[beir] downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def _ensure_dataset(name: str) -> Path:
    if name not in BEIR_DATASETS:
        raise KeyError(f"unknown BEIR dataset {name!r}; "
                       f"choose from {list(BEIR_DATASETS)}")
    url, _ = BEIR_DATASETS[name]
    target = DATA_DIR / name
    if target.exists():
        return target
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / f"{name}.zip"
    if not archive.exists():
        _download(url, archive)
    print(f"[beir] extracting {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(DATA_DIR)
    archive.unlink()
    return target


def load_beir(name: str) -> Tuple[List[str], List[str], List[str], List[str],
                                  Dict[str, Dict[str, int]]]:
    """Returns (corpus_ids, corpus_texts, query_ids, query_texts, qrels).

    qrels[query_id][doc_id] = relevance score (1+ = relevant).
    """
    path = _ensure_dataset(name)
    _, split = BEIR_DATASETS[name]

    corpus_ids: List[str] = []
    corpus_texts: List[str] = []
    with open(path / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus_ids.append(doc["_id"])
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            corpus_texts.append((title + " " + text).strip())

    query_ids: List[str] = []
    query_texts: List[str] = []
    with open(path / "queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            query_ids.append(q["_id"])
            query_texts.append(q["text"])

    qrels: Dict[str, Dict[str, int]] = {}
    with open(path / "qrels" / f"{split}.tsv", "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.strip().split("\t")
            score = int(score)
            if score <= 0:
                continue
            qrels.setdefault(qid, {})[did] = score

    # filter queries to those that have qrels in the requested split
    keep = set(qrels)
    paired = [(qid, qt) for qid, qt in zip(query_ids, query_texts) if qid in keep]
    query_ids = [qid for qid, _ in paired]
    query_texts = [qt for _, qt in paired]

    return corpus_ids, corpus_texts, query_ids, query_texts, qrels


# ============================================================ encoders

# Models we care about for Phase 0 / Phase 1. All accessed via
# sentence-transformers; matryoshka-capable models permit sliced retrieval.
MODELS = {
    "arctic-embed-m-v2.0": {
        # Custom RoPE code requires xformers; disabling its fast paths
        # produces uninitialized position_ids on standard attention.
        # Kept here for reference; not usable on Apple Silicon.
        "hf": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "matryoshka": True,
        "trust_remote_code": True,
        "query_prefix": "query: ",
        "doc_prefix": "",
        "max_seq_length": 512,
        "config_kwargs": {
            "use_memory_efficient_attention": False,
            "unpad_inputs": False,
        },
        "broken_on_apple_silicon": True,
    },
    "mxbai-embed-large-v1": {
        # Bert-large backbone; matryoshka 64-1024. Substitute for arctic
        # when running on MPS / CPU without xformers.
        "hf": "mixedbread-ai/mxbai-embed-large-v1",
        "matryoshka": True,
        "trust_remote_code": False,
        "query_prefix": (
            "Represent this sentence for searching relevant passages: "
        ),
        "doc_prefix": "",
        "max_seq_length": 512,
    },
    "nomic-embed-text-v1.5": {
        "hf": "nomic-ai/nomic-embed-text-v1.5",
        "matryoshka": True,
        "trust_remote_code": True,
        "query_prefix": "search_query: ",
        "doc_prefix": "search_document: ",
        "max_seq_length": 512,
    },
    "bge-m3": {
        "hf": "BAAI/bge-m3",
        "matryoshka": False,
        "trust_remote_code": False,
        "query_prefix": "",
        "doc_prefix": "",
        "max_seq_length": 1024,
    },
    "bge-large-en-v1.5": {
        # No matryoshka training; the "matryoshka" column on this model is
        # just naive top-d truncation, intended as a strawman. Same backbone
        # family as mxbai (bert-large 1024d) for cleaner comparison.
        "hf": "BAAI/bge-large-en-v1.5",
        "matryoshka": True,
        "trust_remote_code": False,
        "query_prefix": (
            "Represent this sentence for searching relevant passages: "
        ),
        "doc_prefix": "",
        "max_seq_length": 512,
    },
    "bge-base-en-v1.5": {
        # 110M-param no-matryoshka cousin of bge-large. Stand-in when
        # bge-large is too slow on MPS to be practical for an interactive
        # session. 768d output.
        "hf": "BAAI/bge-base-en-v1.5",
        "matryoshka": True,
        "trust_remote_code": False,
        "query_prefix": (
            "Represent this sentence for searching relevant passages: "
        ),
        "doc_prefix": "",
        "max_seq_length": 512,
    },
    "e5-base-v2": {
        # 110M-param popular MTEB model, no matryoshka training.
        # Different family than bge — uses simpler "query:" / "passage:"
        # prefixes per its model card.
        "hf": "intfloat/e5-base-v2",
        "matryoshka": True,
        "trust_remote_code": False,
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "max_seq_length": 512,
    },
}


def encode_corpus(model_name: str, dataset_name: str,
                  query_texts: List[str], corpus_texts: List[str],
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Encode (queries, corpus) with the chosen model. Cached on disk."""
    spec = MODELS[model_name]
    cache_q = CACHE_DIR / f"{model_name}__{dataset_name}__queries.npy"
    cache_c = CACHE_DIR / f"{model_name}__{dataset_name}__corpus.npy"
    if cache_q.exists() and cache_c.exists():
        print(f"[encode] loading cached {model_name} on {dataset_name}")
        return np.load(cache_q), np.load(cache_c)

    from sentence_transformers import SentenceTransformer
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[encode] loading model {spec['hf']} on {device}")
    kwargs = {}
    if spec["trust_remote_code"]:
        kwargs["trust_remote_code"] = True
    if "config_kwargs" in spec:
        kwargs["config_kwargs"] = spec["config_kwargs"]
    model = SentenceTransformer(spec["hf"], device=device, **kwargs)
    if "max_seq_length" in spec:
        model.max_seq_length = spec["max_seq_length"]

    qt = [spec["query_prefix"] + t for t in query_texts]
    ct = [spec["doc_prefix"] + t for t in corpus_texts]

    print(f"[encode] queries: {len(qt)}")
    Vq = model.encode(qt, batch_size=32, show_progress_bar=True,
                      convert_to_numpy=True, normalize_embeddings=False)
    print(f"[encode] corpus: {len(ct)}")
    Vc = model.encode(ct, batch_size=32, show_progress_bar=True,
                      convert_to_numpy=True, normalize_embeddings=False)

    Vq = Vq.astype(np.float32)
    Vc = Vc.astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_q, Vq)
    np.save(cache_c, Vc)
    return Vq, Vc


# ============================================================ retrieval

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
    return X / n


def _topk_scores(Vq: np.ndarray, Vc: np.ndarray, k: int = 100,
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Cosine top-k. Assumes Vq, Vc already L2-normalized."""
    sims = Vq @ Vc.T
    # argpartition is faster than argsort but doesn't preserve order; we
    # still sort the k selected because NDCG cares about ranking.
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    top_sims = sims[rows, idx]
    order = np.argsort(-top_sims, axis=1)
    idx = idx[rows, order]
    top_sims = top_sims[rows, order]
    return idx, top_sims


def retrieve_raw(Vq: np.ndarray, Vc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _topk_scores(_l2_normalize(Vq), _l2_normalize(Vc))


def retrieve_matryoshka(Vq: np.ndarray, Vc: np.ndarray, d: int):
    return _topk_scores(_l2_normalize(Vq[:, :d]), _l2_normalize(Vc[:, :d]))


def retrieve_pca(Vq: np.ndarray, Vc: np.ndarray, d: int):
    """PCA fit on corpus. Retrieve cosine in d-dim PCA space."""
    stats = fit_pca(Vc, d)
    Pc = (Vc - stats.mean) @ stats.Q
    Pq = (Vq - stats.mean) @ stats.Q
    return _topk_scores(_l2_normalize(Pq), _l2_normalize(Pc))


def retrieve_poly(Vq: np.ndarray, Vc: np.ndarray, d: int,
                  lam: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Polynomial autoencoder: PCA -> quad lift -> ridge -> V_hat in D-dim.

    Fit AE on the corpus only (queries are unseen). Retrieve cosine on V_hat.
    """
    stats = fit_pca(Vc, d)
    Pc = project(Vc, stats)
    Pq = project(Vq, stats)
    Lc = polynomial_lift(Pc, degree=2)
    W = fit_ridge(Lc, Vc, lam=lam)
    Lq = polynomial_lift(Pq, degree=2)
    Vc_hat = Lc @ W
    Vq_hat = Lq @ W
    return _topk_scores(_l2_normalize(Vq_hat), _l2_normalize(Vc_hat))


# ============================================================ metrics

def ndcg_at_k(qrels: Dict[str, Dict[str, int]],
              query_ids: List[str], corpus_ids: List[str],
              top_idx: np.ndarray, top_sims: np.ndarray, k: int = 10) -> float:
    """Standard NDCG@k with binary or graded relevance from qrels."""
    log2 = np.log2(np.arange(2, k + 2))  # discount factors for ranks 1..k
    scores = []
    for q_pos, qid in enumerate(query_ids):
        rel = qrels.get(qid, {})
        if not rel:
            continue
        ranked = top_idx[q_pos, :k]
        gains = np.array([rel.get(corpus_ids[i], 0) for i in ranked],
                         dtype=np.float64)
        dcg = ((2 ** gains - 1) / log2).sum()
        ideal = sorted(rel.values(), reverse=True)[:k]
        ideal = np.array(ideal, dtype=np.float64)
        if len(ideal) < k:
            ideal = np.pad(ideal, (0, k - len(ideal)))
        idcg = ((2 ** ideal - 1) / log2).sum()
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(qrels: Dict[str, Dict[str, int]],
                query_ids: List[str], corpus_ids: List[str],
                top_idx: np.ndarray, k: int = 10) -> float:
    scores = []
    for q_pos, qid in enumerate(query_ids):
        rel_set = set(qrels.get(qid, {}))
        if not rel_set:
            continue
        retrieved = {corpus_ids[i] for i in top_idx[q_pos, :k]}
        scores.append(len(retrieved & rel_set) / len(rel_set))
    return float(np.mean(scores)) if scores else 0.0


# ============================================================ main

@dataclass
class MethodResult:
    method: str
    ndcg10: float
    recall10: float
    bytes_per_vec: int
    notes: str = ""


def run(model_name: str, dataset_name: str, d: int,
        ks: Tuple[int, ...] = (10,)) -> List[MethodResult]:
    print(f"[run] {model_name} on {dataset_name}, d={d}")
    corpus_ids, corpus_texts, query_ids, query_texts, qrels = load_beir(dataset_name)
    print(f"[run] corpus={len(corpus_texts)}, queries={len(query_texts)}, "
          f"qrels={sum(len(v) for v in qrels.values())}")

    Vq, Vc = encode_corpus(model_name, dataset_name, query_texts, corpus_texts)
    D = Vq.shape[1]
    print(f"[run] embedding dim D={D}")

    spec = MODELS[model_name]
    results: List[MethodResult] = []

    print("[run] retrieve: raw")
    t0 = time.time()
    idx, _ = retrieve_raw(Vq, Vc)
    results.append(MethodResult(
        method=f"raw ({D}d)",
        ndcg10=ndcg_at_k(qrels, query_ids, corpus_ids, idx, None, 10),
        recall10=recall_at_k(qrels, query_ids, corpus_ids, idx, 10),
        bytes_per_vec=D * 2,  # fp16
    ))
    print(f"  done in {time.time()-t0:.1f}s")

    if spec["matryoshka"] and d < D:
        print(f"[run] retrieve: matryoshka top-{d}")
        t0 = time.time()
        idx, _ = retrieve_matryoshka(Vq, Vc, d)
        results.append(MethodResult(
            method=f"matryoshka ({d}d)",
            ndcg10=ndcg_at_k(qrels, query_ids, corpus_ids, idx, None, 10),
            recall10=recall_at_k(qrels, query_ids, corpus_ids, idx, 10),
            bytes_per_vec=d * 2,
        ))
        print(f"  done in {time.time()-t0:.1f}s")

    print(f"[run] retrieve: PCA top-{d}")
    t0 = time.time()
    idx, _ = retrieve_pca(Vq, Vc, d)
    results.append(MethodResult(
        method=f"PCA ({d}d)",
        ndcg10=ndcg_at_k(qrels, query_ids, corpus_ids, idx, None, 10),
        recall10=recall_at_k(qrels, query_ids, corpus_ids, idx, 10),
        bytes_per_vec=d * 2,
    ))
    print(f"  done in {time.time()-t0:.1f}s")

    M = lift_dim(d)
    print(f"[run] retrieve: poly autoencoder d={d} (M={M} lift features)")
    t0 = time.time()
    idx, _ = retrieve_poly(Vq, Vc, d)
    results.append(MethodResult(
        method=f"poly-AE ({d}d -> {D}d V_hat)",
        ndcg10=ndcg_at_k(qrels, query_ids, corpus_ids, idx, None, 10),
        recall10=recall_at_k(qrels, query_ids, corpus_ids, idx, 10),
        bytes_per_vec=d * 2,
        notes=f"M={M}",
    ))
    print(f"  done in {time.time()-t0:.1f}s")

    return results


def print_table(model_name: str, dataset_name: str, d: int,
                results: List[MethodResult]) -> None:
    print()
    print(f"=== {model_name} on {dataset_name}, latent budget d={d} ===")
    print(f"{'method':<32} {'NDCG@10':>9} {'Recall@10':>10} "
          f"{'bytes/vec':>10}  {'notes':<20}")
    print("-" * 90)
    for r in results:
        print(f"{r.method:<32} {r.ndcg10:>9.4f} {r.recall10:>10.4f} "
              f"{r.bytes_per_vec:>10d}  {r.notes:<20}")
    # delta lines
    raw = next((r for r in results if r.method.startswith("raw")), None)
    if raw is not None:
        print()
        for r in results:
            if r is raw:
                continue
            print(f"  {r.method:<30} NDCG drop vs raw: "
                  f"{(r.ndcg10 - raw.ndcg10) * 100:+.2f} pp")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--dataset", required=True, choices=list(BEIR_DATASETS))
    parser.add_argument("--d", type=str, default="256",
                        help="comma-separated latent dims for matryoshka/PCA/poly")
    args = parser.parse_args()
    ds = [int(x) for x in args.d.split(",")]
    for d in ds:
        results = run(args.model, args.dataset, d)
        print_table(args.model, args.dataset, d, results)


if __name__ == "__main__":
    main()
