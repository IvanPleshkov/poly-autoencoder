"""
Reproduce the headline table from the post: eight embeddings, one row each,
optimal d for each model.

Classical embeddings are loaded from cache or downloaded on first call.
Transformer-encoded embeddings require encoding the first 20K DBpedia
texts; this is done lazily and cached.

Run:
    python reproduce.py                    # all 8 rows
    python reproduce.py --classical-only   # skip transformer encoders
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from dataloaders import load, CLASSICAL_DATASETS
from encoders import encode
from poly_pca import run_experiment


# (name, kind, n, d) chosen as in the post.
# kind ∈ {"classical", "encode"}.
TABLE = [
    ("dbpedia-openai",    "classical", 100_000, 100),
    ("sentence-t5-base",  "encode",     20_000,  50),
    ("deep-image-96",     "classical", 100_000,  20),
    ("all-MiniLM-L6-v2",  "encode",     20_000,  40),
    ("all-mpnet-base-v2", "encode",     20_000,  50),
    ("clip-text-vit-b32", "encode",     20_000,  40),
    ("glove-100",         "classical", 100_000,  50),
    ("nytimes-256",       "classical", 100_000, 100),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical-only", action="store_true",
                        help="skip transformer encoders")
    args = parser.parse_args()

    rows = TABLE if not args.classical_only else \
        [r for r in TABLE if r[1] == "classical"]

    results = []
    for name, kind, n, d in rows:
        t0 = time.time()
        print(f"\n=== {name} (n={n}, d={d}) ===")
        if kind == "classical":
            V, _ = load(name, n=n)
        else:
            V = encode(name, n=n)
        print(f"    V: shape={V.shape}")
        res = run_experiment(V, d, name=name)
        results.append((res, V.shape[1]))
        print(f"    {res.as_row()}  [{time.time() - t0:.1f}s]")

    print()
    print("=" * 100)
    print("Headline table (R² on 80/20 holdout, optimal d per model):")
    print("=" * 100)
    header = (f"{'Embedding':<22} {'D':>5} {'N':>8} {'cond':>6} "
              f"{'d':>4} {'PCA R²':>7} {'Poly R²':>8} {'Δ p.p.':>7}")
    print(header)
    print("-" * len(header))
    for r, D in results:
        print(f"{r.name:<22} "
              f"{D:>5} "
              f"{r.n_train + r.n_test:>8} "
              f"{r.cond_orig:>6.1f} "
              f"{r.d:>4} "
              f"{r.pca_r2_test:>7.3f} "
              f"{r.poly_r2_test:>8.3f} "
              f"{(r.poly_r2_test - r.pca_r2_test) * 100:>+7.1f}")


if __name__ == "__main__":
    main()
