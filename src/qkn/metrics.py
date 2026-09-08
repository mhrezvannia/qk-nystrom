"""Evaluation metrics, kernel diagnostics, confidence intervals, and quantum-vs-classical analysis."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def kta(K: np.ndarray, y: np.ndarray, centered: bool = True) -> float:
    """(Centered) kernel-target alignment <K, yy^T>_F / (||K||_F ||yy^T||_F), y in {-1,+1}."""
    y = np.where(np.asarray(y) > 0, 1.0, -1.0)
    Y = np.outer(y, y)
    if centered:
        n = K.shape[0]; H = np.eye(n) - np.ones((n, n)) / n
        K = H @ K @ H; Y = H @ Y @ H
    return float((K * Y).sum() / (np.linalg.norm(K) * np.linalg.norm(Y) + 1e-300))


def classification_metrics(y_true, y_pred) -> dict:
    return dict(acc=float(accuracy_score(y_true, y_pred)),
                bal_acc=float(balanced_accuracy_score(y_true, y_pred)),
                f1=float(f1_score(y_true, y_pred)))


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def kernel_spectrum_stats(K: np.ndarray) -> dict:
    ev = np.sort(np.linalg.eigvalsh((K + K.T) / 2))[::-1]
    ev_pos = np.clip(ev, 0, None)
    cum = np.cumsum(ev_pos) / ev_pos.sum()
    off = K[~np.eye(K.shape[0], dtype=bool)]
    return dict(rank_1e_8=int((ev > 1e-8 * ev[0]).sum()),
                eff_rank=float(np.exp(-(ev_pos / ev_pos.sum() * np.log(ev_pos / ev_pos.sum() + 1e-300)).sum())),
                energy_top20=float(cum[min(19, len(cum) - 1)]), energy_top60=float(cum[min(59, len(cum) - 1)]),
                energy_top120=float(cum[min(119, len(cum) - 1)]),
                min_eig=float(ev[-1]), max_eig=float(ev[0]), cond=float(ev[0] / max(abs(ev[-1]), 1e-300)),
                offdiag_mean=float(off.mean()), offdiag_std=float(off.std()), offdiag_median=float(np.median(off)))


def geometric_difference(K_C: np.ndarray, K_Q: np.ndarray, lam: float = 0.0) -> float:
    """g(K_C || K_Q) = sqrt(|| sqrt(K_Q) (K_C + lam I)^-1 sqrt(K_Q) ||_inf), Huang et al. (2021), with kernels
    normalised to trace N.  g ~ 1 means the classical kernel can reproduce every function of the quantum kernel;
    large g (~ sqrt(N)) is a necessary condition for any prediction advantage of K_Q over K_C."""
    N = K_C.shape[0]
    K_C = K_C * N / np.trace(K_C); K_Q = K_Q * N / np.trace(K_Q)
    w, V = np.linalg.eigh((K_Q + K_Q.T) / 2); w = np.clip(w, 0, None)
    sqQ = (V * np.sqrt(w)) @ V.T
    inv = np.linalg.solve(K_C + lam * np.eye(N) + 1e-10 * np.eye(N), np.eye(N))
    M = sqQ @ inv @ sqQ
    return float(np.sqrt(np.linalg.norm((M + M.T) / 2, 2)))


def kernel_distance(K1: np.ndarray, K2: np.ndarray) -> float:
    """Normalised Frobenius distance between two kernel matrices after trace normalisation."""
    N = K1.shape[0]
    A = K1 * N / np.trace(K1); B = K2 * N / np.trace(K2)
    return float(np.linalg.norm(A - B) / np.linalg.norm(A))
