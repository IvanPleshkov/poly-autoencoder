"""
Polynomial PCA decomposition for vector embeddings.

Pipeline (closed-form, no SGD):

    V (N, D)
        ├── fit_pca on V         -> (mean, Q (D, d), eigvals (d,))
        ├── project (centered, std-normalized, scale to ‖p‖ <= 0.9)
        │                        -> P (N, d)
        ├── polynomial_lift      -> L (N, M),  M = 1 + d + d(d+1)/2
        ├── ridge OLS            -> W (M, D)
        └── decode: V_hat = L @ W
            residual:    V_resid = V - V_hat

This file exposes the primitives. End-to-end retrieval evaluation lives
in beir_eval.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------- PCA fitting
@dataclass
class PCAStats:
    mean: np.ndarray            # (D,) f32
    Q: np.ndarray               # (D, d) f32, columns are top-d eigenvectors
    eigvals: np.ndarray         # (d,) f64, descending
    ball_scale: float           # global multiplier for ‖p‖ <= 0.9


def fit_pca(V_train: np.ndarray, d: int, ball_radius: float = 0.9) -> PCAStats:
    """Fit PCA on training set; freeze stats for application to held-out data."""
    mean = V_train.mean(axis=0)
    Vc = V_train - mean
    cov = (Vc.T @ Vc) / len(V_train)
    e_full, U_full = np.linalg.eigh(cov.astype(np.float64))
    # descending
    e_full = e_full[::-1]
    U_full = U_full[:, ::-1]
    Q = U_full[:, :d]
    eigvals = e_full[:d]

    # ball_scale chosen on training projection: largest projected norm
    # after per-axis std normalization should fit within ball_radius.
    proj = Vc @ Q
    proj_normed = proj / np.sqrt(eigvals)
    max_norm = float(np.linalg.norm(proj_normed, axis=1).max())
    ball_scale = ball_radius / max_norm

    return PCAStats(
        mean=mean.astype(np.float32),
        Q=Q.astype(np.float32),
        eigvals=eigvals,
        ball_scale=ball_scale,
    )


def project(V: np.ndarray, stats: PCAStats) -> np.ndarray:
    """Apply frozen PCA stats to V. Returns p in R^{N x d}, ‖p‖ <= 0.9."""
    centered = V - stats.mean
    proj = centered @ stats.Q
    normed = proj / np.sqrt(stats.eigvals.astype(np.float32))
    return (normed * stats.ball_scale).astype(np.float32)


# ----------------------------------------------------------- polynomial lift
def polynomial_lift(P: np.ndarray, degree: int = 2) -> np.ndarray:
    """Lift P (N, d) to polynomial features up to given degree.

    degree=2: features are [1, p_i, p_i*p_j (i<=j)]; M = 1 + d + d(d+1)/2.
    Only degree=2 implemented; degree=3 is feasible but explodes M.
    """
    if degree != 2:
        raise NotImplementedError(f"degree={degree} not supported")
    Pf = P.astype(np.float32)
    N_, d = Pf.shape
    iu, ju = np.triu_indices(d)
    M = 1 + d + len(iu)
    L = np.empty((N_, M), dtype=np.float32)
    L[:, 0] = 1.0
    L[:, 1:1 + d] = Pf
    L[:, 1 + d:] = Pf[:, iu] * Pf[:, ju]
    return L


def lift_dim(d: int, degree: int = 2) -> int:
    """Number of features after polynomial lift."""
    if degree != 2:
        raise NotImplementedError
    return 1 + d + d * (d + 1) // 2


# ----------------------------------------------------------- ridge OLS
def fit_ridge(L: np.ndarray, V: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Closed-form Ridge regression: minimize ‖L W - V‖² + lam·tr(L^T L)/M·‖W‖².

    Returns W (M, D) f32. Done in f64 for numerical stability with large M.
    """
    L64 = L.astype(np.float64)
    V64 = V.astype(np.float64)
    M = L64.shape[1]
    G = L64.T @ L64
    eye_scale = lam * float(np.trace(G)) / M
    G += eye_scale * np.eye(M)
    rhs = L64.T @ V64
    W = np.linalg.solve(G, rhs)
    return W.astype(np.float32)
