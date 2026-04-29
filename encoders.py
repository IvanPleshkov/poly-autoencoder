"""
Transformer-based encoders for the four non-classical embeddings in the
post. Each encoder takes DBpedia text (loaded via datasets.load_dbpedia_text)
and produces embeddings cached as .npy under cache/.

  - sentence-t5-base       768d   sentence-transformers
  - all-mpnet-base-v2      768d   sentence-transformers
  - all-MiniLM-L6-v2       384d   sentence-transformers
  - clip-text-vit-b32      512d   transformers (HuggingFace)

Encoding 20K texts on a M-series MacBook (CPU): ~1–6 minutes per model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from dataloaders import load_dbpedia_text

CACHE_DIR = Path(__file__).parent / "cache"


SENTENCE_TRANSFORMER_MODELS = {
    "sentence-t5-base":   "sentence-transformers/sentence-t5-base",
    "all-mpnet-base-v2":  "sentence-transformers/all-mpnet-base-v2",
    "all-MiniLM-L6-v2":   "sentence-transformers/all-MiniLM-L6-v2",
}


def _normalize(V: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(V, axis=1, keepdims=True).clip(min=1e-12)
    return (V / nrm).astype(np.float32)


def _encode_sentence_transformer(texts: Sequence[str], model_name: str,
                                 batch_size: int = 32) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODELS[model_name])
    V = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,    # we normalize ourselves
    ).astype(np.float32)
    return V


def _encode_clip_text(texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    """CLIP text encoder (ViT-B/32). Truncates to model max length."""
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    name = "openai/clip-vit-base-patch32"
    tok = CLIPTokenizer.from_pretrained(name)
    model = CLIPModel.from_pretrained(name).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            enc = tok(batch, padding=True, truncation=True, max_length=77,
                      return_tensors="pt")
            res = model.get_text_features(**enc)
            # transformers <5: returns Tensor; transformers >=5: returns
            # BaseModelOutputWithPooling whose pooler_output is the
            # text feature tensor.
            feats = (res if isinstance(res, torch.Tensor)
                     else res.pooler_output)
            out.append(feats.cpu().numpy())
            if i % (10 * batch_size) == 0:
                print(f"    [clip] {i}/{len(texts)}")
    V = np.vstack(out).astype(np.float32)
    return V


SUPPORTED_ENCODERS = (
    "sentence-t5-base",
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "clip-text-vit-b32",
)


def encode(name: str, n: int) -> np.ndarray:
    """Encode the first n DBpedia texts with the given model.

    Cached under cache/{name}_n{n}.npy. Returns (n, D) unit-normalized f32.
    """
    if name not in SUPPORTED_ENCODERS:
        raise KeyError(f"unknown encoder {name!r}; "
                       f"available: {SUPPORTED_ENCODERS}")
    cache = CACHE_DIR / f"{name}_n{n}.npy"
    if cache.exists():
        return np.load(cache)

    print(f"[encode] {name}: loading {n} texts from DBpedia...")
    texts = load_dbpedia_text(n)
    print(f"[encode] {name}: encoding {n} texts (this can take minutes)...")
    if name == "clip-text-vit-b32":
        V = _encode_clip_text(texts)
    else:
        V = _encode_sentence_transformer(texts, name)
    V = _normalize(V)
    np.save(cache, V)
    return V
