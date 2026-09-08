"""Nyström approximation with explicit numerical controls, landmark strategies and error metrics."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus


# ----------------------------------------------------------------------------- landmark selection
def select_landmarks(X: np.ndarray, m: int, strategy: str, rng: np.random.Generator,
                     K_full: np.ndarray | None = None, ridge: float = 1e-3):
    """Return (Z, idx_or_None).  idx is given when landmarks are training points.

    strategies
      uniform          : m training points sampled uniformly without replacement
      kmeanspp         : m training points chosen by k-means++ D^2 seeding (no Lloyd iterations)
      kmeans_centroids : centroids of k-means (k=m) — NOT data points (must be encoded as new circuits)
      kmeans_nearest   : the training point closest to each k-means centroid (data points)
      leverage         : ridge-leverage-score sampling (requires the FULL kernel — oracle, budget-infeasible)
    """
    N = X.shape[0]
    m = min(m, N)
    seed = int(rng.integers(0, 2**31 - 1))
    if strategy == "uniform":
        idx = rng.choice(N, m, replace=False)
        return X[idx], idx
    if strategy == "kmeanspp":
        _, idx = kmeans_plusplus(X, n_clusters=m, random_state=seed)
        return X[idx], np.asarray(idx)
    if strategy in ("kmeans_centroids", "kmeans_nearest"):
        km = KMeans(n_clusters=m, n_init=3, random_state=seed).fit(X)
        if strategy == "kmeans_centroids":
            return km.cluster_centers_, None
        d = ((X[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
        idx = np.unique(d.argmin(0))
        if len(idx) < m:  # fill duplicates with uniform picks
            rest = np.setdiff1d(np.arange(N), idx)
            idx = np.concatenate([idx, rng.choice(rest, m - len(idx), replace=False)])
        return X[idx], idx
    if strategy == "leverage":
        if K_full is None:
            raise ValueError("leverage-score sampling needs the full kernel (oracle strategy)")
        lev = np.diag(K_full @ np.linalg.solve(K_full + ridge * N * np.eye(N), np.eye(N)))
        p = np.clip(lev, 1e-12, None); p /= p.sum()
        idx = rng.choice(N, m, replace=False, p=p)
        return X[idx], idx
    raise ValueError(strategy)


# ----------------------------------------------------------------------------- Nyström core
class Nystrom:
    """K_tilde = C W_eps^+ C^T with W_eps = W + eps I and pseudo-inverse cut-off rcond."""

    def __init__(self, eps: float = 1e-6, rcond: float = 1e-10, normalize: bool = False):
        self.eps, self.rcond, self.normalize = eps, rcond, normalize

    def fit(self, W: np.ndarray):
        self.W = W
        Weps = W + self.eps * np.eye(W.shape[0])
        self.W_pinv = np.linalg.pinv(Weps, rcond=self.rcond, hermitian=True)
        ev = np.linalg.eigvalsh(Weps)
        # cond_W: raw ratio lambda_max / lambda_min of W_eps (inf when W_eps is singular or indefinite, i.e. lambda_min <= 0).
        # cond_W_eff: condition number of the regularised inverse actually applied, lambda_max * ||W_eps^+||_2 (always finite).
        self.min_eig_W = float(ev.min())
        self.indefinite = bool(ev.min() <= 0)
        self.cond_W = float(ev.max() / ev.min()) if ev.min() > 0 else float("inf")
        self.cond_W_eff = float(ev.max() * np.linalg.norm(self.W_pinv, 2))
        return self

    def transform(self, C_rows: np.ndarray, C_cols: np.ndarray | None = None) -> np.ndarray:
        """Approximate K(rows, cols) from C_rows = K(rows, Z), C_cols = K(cols, Z)."""
        if C_cols is None:
            C_cols = C_rows
        Kt = C_rows @ self.W_pinv @ C_cols.T
        if self.normalize:
            d_r = np.sqrt(np.clip(np.einsum('ij,jk,ik->i', C_rows, self.W_pinv, C_rows), 1e-12, None))
            d_c = np.sqrt(np.clip(np.einsum('ij,jk,ik->i', C_cols, self.W_pinv, C_cols), 1e-12, None))
            Kt = Kt / d_r[:, None] / d_c[None, :]
        return Kt


# ----------------------------------------------------------------------------- error metrics
def approximation_metrics(K: np.ndarray, Kt: np.ndarray) -> dict:
    D = K - Kt
    fro = np.linalg.norm(D) / np.linalg.norm(K)
    spec = np.linalg.norm(D, 2) / np.linalg.norm(K, 2)
    diag = np.abs(np.diag(D)).mean()
    off = (np.abs(D).sum() - np.abs(np.diag(D)).sum()) / (K.shape[0] * (K.shape[0] - 1))
    return dict(E_F=float(fro), E_spec=float(spec), E_diag=float(diag), E_offdiag=float(off))
