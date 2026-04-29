"""
Polynomial PCA decomposition for vector embeddings.

Pipeline (closed-form, no SGD):

    V (N, D)
        ├── fit_pca on train         -> (mean, Q (D, d), eigvals (d,))
        ├── project (centered, std-normalized, scale to ‖p‖ <= 0.9)
        │                            -> P (N, d)
        ├── polynomial_lift          -> L (N, M),  M = 1 + d + d(d+1)/2
        ├── ridge OLS                -> W (M, D)
        └── decode: V_struct = L @ W
            residual:    V_resid = V - V_struct

Reports R² (train, test, overfit gap) on a 80/20 split, plus cond before
and after — see post.md for context. This is essentially PPCR (Polynomial
Principal Component Regression, Vong, Geladi, Wold & Esbensen 1989) with
the target set to the input itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ----------------------------------------------------------- PCA fitting
@dataclass
class PCAStats:
    mean: np.ndarray            # (D,) f32
    Q: np.ndarray               # (D, d) f32, columns are top-d eigenvectors
    eigvals: np.ndarray         # (d,) f64, descending
    eigvals_full: np.ndarray    # (D,) f64, all eigvals (descending)
    ball_scale: float           # global multiplier for ‖p‖ <= 0.9


def fit_pca(V_train: np.ndarray, d: int, ball_radius: float = 0.9) -> PCAStats:
    """Fit PCA on training set; freeze stats for application to held-out."""
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
        eigvals_full=e_full,
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


# ----------------------------------------------------------- metrics
def r2(V_true: np.ndarray, V_pred: np.ndarray) -> float:
    """Coefficient of determination, vector-form (R² over all coords)."""
    diff = V_true.astype(np.float64) - V_pred.astype(np.float64)
    sse = float((diff * diff).sum())
    Vc = V_true.astype(np.float64) - V_true.astype(np.float64).mean(axis=0)
    sst = float((Vc * Vc).sum())
    return 1.0 - sse / sst


def anisotropy(X: np.ndarray, top_k: int = 200,
               noise_floor: float = 1e-4) -> float:
    """cond = lambda_max / lambda_min over centered covariance.

    Restricts to the top-K eigenvalues *and* drops the numerical-noise
    tail (eigenvalues below noise_floor × lambda_max). Without the
    second clip, residuals from a strong decoder produce eigenvalues
    indistinguishable from FP noise, and cond explodes spuriously.

    The default floor of 1e-4 × λ_max keeps eigen-directions that
    carry at least 0.01% of the largest direction's variance. That's
    above the round-off accumulated through f32 matmuls and below all
    signal modes we've seen (top-50 directions on deep-image-96
    sit between 0.30 × λ_max and 1.0 × λ_max — comfortably kept). If
    you have ultra-clean data with f64 throughout, you can tighten
    to 1e-8 and recover sub-percent variance modes.
    """
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(axis=0)
    cov = (Xc.T @ Xc) / len(X)
    e = np.linalg.eigvalsh(cov)[::-1]
    e = e[:min(top_k, len(e))]
    floor = e[0] * noise_floor
    e = e[e > floor]
    return float(e[0] / e[-1])


# ----------------------------------------------------------- one experiment
@dataclass
class ExperimentResult:
    name: str
    d: int
    M: int
    n_train: int
    n_test: int
    cond_orig: float
    pca_r2_test: float          # explained variance by linear PCA on test
    poly_r2_train: float        # polynomial decoder R² on training set
    poly_r2_test: float         # polynomial decoder R² on holdout
    overfit: float              # poly_r2_train - poly_r2_test
    cond_resid_test: float      # cond of residual on holdout

    def as_row(self) -> str:
        return (f"{self.name:<22} d={self.d:<4} M={self.M:<6} "
                f"PCA={self.pca_r2_test:.3f} "
                f"Poly_train={self.poly_r2_train:.3f} "
                f"Poly_test={self.poly_r2_test:.3f} "
                f"Δ={self.poly_r2_test - self.pca_r2_test:+.3f} "
                f"cond_orig={self.cond_orig:.1f} "
                f"cond_resid={self.cond_resid_test:.2f}")


def run_experiment(
    V: np.ndarray,
    d: int,
    *,
    name: str = "embedding",
    train_frac: float = 0.8,
    seed: int = 42,
    degree: int = 2,
    lam: float = 1e-3,
) -> ExperimentResult:
    """Full poly-PCA pipeline with held-out evaluation."""
    rng = np.random.default_rng(seed)
    N = len(V)
    perm = rng.permutation(N)
    n_train = int(N * train_frac)
    V_train = V[perm[:n_train]]
    V_test = V[perm[n_train:]]

    stats = fit_pca(V_train, d)
    P_train = project(V_train, stats)
    P_test = project(V_test, stats)

    # PCA-only baseline R² evaluated on the held-out set
    V_test_centered = V_test - stats.mean
    V_pca_test = V_test_centered @ stats.Q @ stats.Q.T + stats.mean
    pca_r2_test = r2(V_test, V_pca_test)

    # polynomial decoder
    L_train = polynomial_lift(P_train, degree=degree)
    L_test = polynomial_lift(P_test, degree=degree)
    M = L_train.shape[1]
    W = fit_ridge(L_train, V_train, lam=lam)

    V_pred_train = L_train @ W
    V_pred_test = L_test @ W
    poly_r2_train = r2(V_train, V_pred_train)
    poly_r2_test = r2(V_test, V_pred_test)

    V_resid_test = V_test - V_pred_test
    cond_orig = anisotropy(V_test)
    cond_resid_test = anisotropy(V_resid_test)

    return ExperimentResult(
        name=name,
        d=d,
        M=M,
        n_train=n_train,
        n_test=N - n_train,
        cond_orig=cond_orig,
        pca_r2_test=pca_r2_test,
        poly_r2_train=poly_r2_train,
        poly_r2_test=poly_r2_test,
        overfit=poly_r2_train - poly_r2_test,
        cond_resid_test=cond_resid_test,
    )
