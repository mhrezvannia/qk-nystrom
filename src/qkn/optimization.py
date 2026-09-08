"""Training of feature-map parameters by centred kernel-target alignment (KTA).

Objective: maximise KTA(K_theta(X_fit), y_fit).  Optimiser: L-BFGS-B with finite-difference gradients
(the parameter count is small, 8–16).  Everything is deterministic given the seed.  The training set is a
sub-sample of the *fit* split only; validation and test data are never used here.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .metrics import kta
from .kernels import fidelity_kernel


def effective_rank(K: np.ndarray) -> float:
    """Entropy effective rank exp(H(p)), p_i = lambda_i / sum lambda (same statistic as metrics.kernel_spectrum_stats)."""
    ev = np.clip(np.linalg.eigvalsh((K + K.T) / 2), 0, None)
    p = ev / (ev.sum() + 1e-300)
    return float(np.exp(-(p * np.log(p + 1e-300)).sum()))


def train_kta(fmap, X_fit: np.ndarray, y_fit: np.ndarray, rng: np.random.Generator,
              n_sub: int = 256, maxiter: int = 60, init: str = "uniform", restarts: int = 1, rank_lambda: float = 0.0):
    """Maximise J = KTA_c(K) + rank_lambda * log(erank(K)) / log(n_sub) on a fit-split sub-sample.
    rank_lambda = 0 is plain centred KTA (the main-study objective); rank_lambda > 0 is the pre-registered
    regularised objective of docs/alternative_training_protocol.md.
    Return dict(theta, kta_init, kta_final, trajectory, n_evals, theta_init[, restart])."""
    idx = rng.choice(len(y_fit), min(n_sub, len(y_fit)), replace=False)
    Xs, ys = X_fit[idx], y_fit[idx]
    best = None
    for r in range(restarts):
        theta0 = fmap.init_params(rng, init)
        traj = []

        def neg_kta(th):
            S = fmap.states(Xs, th)
            K = fidelity_kernel(S, S)
            v = -kta(K, ys)
            if rank_lambda > 0:
                v -= rank_lambda * np.log(effective_rank(K)) / np.log(len(ys))
            traj.append(-v)
            return v

        if fmap.n_params == 0:
            return dict(theta=theta0, kta_init=-neg_kta(theta0), kta_final=-neg_kta(theta0), trajectory=traj,
                        n_evals=len(traj), theta_init=theta0)
        k0 = -neg_kta(theta0)
        res = minimize(neg_kta, theta0, method="L-BFGS-B", options=dict(maxiter=maxiter, eps=1e-4))
        out = dict(theta=res.x, kta_init=k0, kta_final=-res.fun, trajectory=traj, n_evals=len(traj),
                   theta_init=theta0, restart=r)
        if best is None or out["kta_final"] > best["kta_final"]:
            best = out
    return best
