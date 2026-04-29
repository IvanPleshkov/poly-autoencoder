"""
Run the polynomial-PCA experiment on a single embedding × d.

Examples:

    # classical pre-computed embeddings (load + run)
    python run.py --embedding dbpedia-openai --d 100

    # transformer-encoded (encode DBpedia text first, then run)
    python run.py --embedding all-MiniLM-L6-v2 --d 40 --n 20000

    # sweep multiple d values
    python run.py --embedding deep-image-96 --d 20,50,100
"""
from __future__ import annotations

import argparse

import numpy as np

from dataloaders import CLASSICAL_DATASETS, load
from encoders import SUPPORTED_ENCODERS, encode
from poly_pca import run_experiment


ALL_EMBEDDINGS = list(CLASSICAL_DATASETS) + list(SUPPORTED_ENCODERS)


def load_embedding(name: str, n: int | None) -> np.ndarray:
    if name in CLASSICAL_DATASETS:
        V, _ = load(name, n=n)
    elif name in SUPPORTED_ENCODERS:
        if n is None:
            n = 20_000
        V = encode(name, n=n)
    else:
        raise KeyError(f"unknown embedding {name!r}; "
                       f"available: {ALL_EMBEDDINGS}")
    return V


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Polynomial-PCA decomposition: PCA top-d → quadratic "
                     "lift → Ridge OLS, with 80/20 holdout."),
    )
    parser.add_argument(
        "--embedding", required=True, choices=ALL_EMBEDDINGS,
        help="which embedding to test",
    )
    parser.add_argument(
        "--d", type=str, default="50,100",
        help="comma-separated list of PCA dimensions to try (default: 50,100)",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="number of vectors to use (default: dataset's default)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lam", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"[run] loading {args.embedding} ...")
    V = load_embedding(args.embedding, args.n)
    print(f"[run] V shape={V.shape}, dtype={V.dtype}, "
          f"norm range=[{np.linalg.norm(V, axis=1).min():.4f}, "
          f"{np.linalg.norm(V, axis=1).max():.4f}]")

    ds = [int(x) for x in args.d.split(",")]
    print(f"[run] sweeping d ∈ {ds}")
    print()
    print(f"{'d':>4} {'M':>6} {'PCA R²':>8} {'Poly R²_train':>14} "
          f"{'Poly R²_test':>14} {'Δ over PCA':>11} "
          f"{'cond_orig':>10} {'cond_resid':>11}")
    print("-" * 84)
    for d in ds:
        res = run_experiment(V, d, name=args.embedding, seed=args.seed,
                             lam=args.lam)
        print(f"{res.d:>4} {res.M:>6} {res.pca_r2_test:>8.3f} "
              f"{res.poly_r2_train:>14.3f} {res.poly_r2_test:>14.3f} "
              f"{res.poly_r2_test - res.pca_r2_test:>+11.3f} "
              f"{res.cond_orig:>10.2f} {res.cond_resid_test:>11.3f}")


if __name__ == "__main__":
    main()
