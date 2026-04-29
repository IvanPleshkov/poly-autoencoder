"""
Loaders for the four classical embeddings used in the post.

Each loader returns:
    (vectors: np.ndarray of shape (n, D), float32, unit-normalized,
     texts: list[str] | None — only for HF text datasets)

Caching: vectors are saved as .npy under cache/ on first call; subsequent
calls hit the cache.

  - dbpedia-openai     KShivendu/dbpedia-entities-openai-1M  (HF)   1536d
  - glove-100          ann-benchmarks  HDF5                          100d
  - nytimes-256        ann-benchmarks  HDF5                          256d
  - deep-image-96      ann-benchmarks  HDF5                           96d
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


_HDF5_URL = "http://ann-benchmarks.com/{name}-angular.hdf5"

CLASSICAL_DATASETS = {
    "dbpedia-openai":   dict(D=1536, default_n=100_000),
    "glove-100":        dict(D=100,  default_n=100_000),
    "nytimes-256":      dict(D=256,  default_n=100_000),
    "deep-image-96":    dict(D=96,   default_n=100_000),
}


def _normalize(V: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(V, axis=1, keepdims=True).clip(min=1e-12)
    return (V / nrm).astype(np.float32)


def _load_dbpedia_openai(n: int) -> Tuple[np.ndarray, list[str]]:
    from datasets import load_dataset
    ds = load_dataset(
        "KShivendu/dbpedia-entities-openai-1M",
        split=f"train[:{n}]",
    )
    V = np.asarray(ds["openai"], dtype=np.float32)
    texts = list(ds["text"])
    return V, texts


def _load_hdf5(name: str, n: int) -> Tuple[np.ndarray, None]:
    """Load first n vectors from a local ann-benchmarks HDF5 file."""
    import h5py
    path = CACHE_DIR / f"{name}-angular.hdf5"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download with:\n"
            f"  curl -sL --output {path} {_HDF5_URL.format(name=name)}"
        )
    with h5py.File(path, "r") as f:
        return np.asarray(f["train"][:n], dtype=np.float32), None


def load(name: str, n: int | None = None) -> Tuple[np.ndarray, list[str] | None]:
    """Load a classical embedding by name.

    Returns (V (n, D) unit-normalized float32, texts or None).
    """
    if name not in CLASSICAL_DATASETS:
        raise KeyError(f"unknown dataset {name!r}; "
                       f"available: {list(CLASSICAL_DATASETS)}")
    if n is None:
        n = CLASSICAL_DATASETS[name]["default_n"]

    cache = CACHE_DIR / f"{name}_n{n}.npy"
    if cache.exists():
        V = np.load(cache)
        # texts are not cached — encoders.py uses dbpedia text via HF
        return V, None

    if name == "dbpedia-openai":
        V, texts = _load_dbpedia_openai(n)
    else:
        V, texts = _load_hdf5(name, n)
    V = _normalize(V)
    np.save(cache, V)
    return V, texts


def load_dbpedia_text(n: int) -> list[str]:
    """Just the text fields from DBpedia, used by encoders.py."""
    cache = CACHE_DIR / f"dbpedia-text_n{n}.txt"
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]
    from datasets import load_dataset
    ds = load_dataset(
        "KShivendu/dbpedia-entities-openai-1M",
        split=f"train[:{n}]",
    )
    texts = list(ds["text"])
    # newlines in text would break our line-by-line cache; replace with spaces
    flat = [t.replace("\n", " ").replace("\r", " ") for t in texts]
    with open(cache, "w", encoding="utf-8") as f:
        f.write("\n".join(flat))
    return flat
