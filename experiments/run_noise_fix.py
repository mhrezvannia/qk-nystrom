"""Noise-robust Nyström variants, motivated by Proposition 3 (noise amplification through W_eps^+).
Same protocol as run_noise.py (2 tasks x 10 seeds, depolarising = 0), comparing at S in {100, 1000, 10000}:
  plain-pinv       : W_eps^+ via Moore-Penrose pseudo-inverse (rcond 1e-10, eps = 1e-6) exactly as in run_noise.py
  psd-project      : eigen-decompose noisy W_eps, discard non-positive eigenvalues, invert the rest (tau = 0)
  jitter-0.1       : eps = 0.1
  truncate-tau     : eigen-decompose the noisy W, keep eigenvalues > tau * lambda_max (tau in {1e-3, 1e-2, 1e-1}), invert only those
  adaptive         : eps = c * sigma_W * sqrt(m) with sigma_W^2 = mean_pq W_pq(1-W_pq)/S (the expected noise scale of the
                     estimated W, from the binomial variance) and c = 2 -- an eps that grows with the noise level
  adaptive+trunc   : adaptive eps and truncation at tau = 1e-2
Also records the noise-free accuracy of every variant (the bias each regulariser introduces when there is no noise)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from qkn.pipeline import load_config, prepare, tune_quantum_kernel, quantum_states, svm_precomputed, ROOT
from qkn.kernels import fidelity_kernel, shot_noise
from qkn.nystrom import select_landmarks, approximation_metrics
from qkn.metrics import kta, wilson_ci
from qkn.data import Preprocessor

cfg = load_config(); nc = cfg["noise"]
task_specs = [t for t in cfg["data"]["tasks"] if "classes" in t and f"{t['dataset']}_{t['classes'][0]}v{t['classes'][1]}" in nc["tasks"]]
SHOTS = [100, 1000, 10000, None]; TAUS = [1e-3, 1e-2, 1e-1]; C_ADAPT = 2.0


def inv_trunc(W, eps, tau):
    Weps = (W + W.T) / 2 + eps * np.eye(W.shape[0])
    w, V = np.linalg.eigh(Weps)
    if tau is None:  # Moore-Penrose pseudo-inverse as used in the main study (inverts negative eigenvalues too)
        return np.linalg.pinv(Weps, rcond=1e-10, hermitian=True), int((np.abs(w) > 1e-10 * np.abs(w).max()).sum()), float(w.min())
    keep = w > tau * w.max()
    return (V[:, keep] / w[keep]) @ V[:, keep].T, int(keep.sum()), float(w.min())


rows = []
for tspec in task_specs:
    for seed in cfg["seeds"]:
        P = prepare(tspec, seed, cfg); ytr, yte, n_te = P["y_train"], P["y_test"], P["n_test"]
        tq = tune_quantum_kernel(P, cfg, cfg["feature_map"]["primary"], cfg["feature_map"]["layers"], cfg["feature_map"]["entangle"])
        fm, th, a, C = tq["fmap"], tq["theta"], tq["a"], tq["C"]
        S_tr, S_te = quantum_states(P, fm, th, a); K_tr = fidelity_kernel(S_tr, S_tr)
        Xtr_a = Preprocessor.to_angles(P["U_train"], a)
        for m in nc["m_values"]:
            Z, _ = select_landmarks(Xtr_a, m, nc["strategy"], np.random.default_rng(seed * 7919 + m)); S_Z = fm.states(Z, th)
            C_trZ, W, C_teZ = fidelity_kernel(S_tr, S_Z), fidelity_kernel(S_Z, S_Z), fidelity_kernel(S_te, S_Z)
            for shots in SHOTS:
                rng = np.random.default_rng(seed * 100003 + (shots or 0))
                Cn, Wn, Cn_te = shot_noise(C_trZ, shots, rng), shot_noise(W, shots, rng, symmetric=True), shot_noise(C_teZ, shots, rng)
                sigma_W = np.sqrt(np.mean(np.clip(Wn, 0, 1) * (1 - np.clip(Wn, 0, 1))) / shots) if shots else 0.0
                eps_ad = C_ADAPT * sigma_W * np.sqrt(m)
                variants = [("plain-pinv", 1e-6, None), ("psd-project", 1e-6, 0.0), ("jitter-0.1-pinv", 0.1, None), ("adaptive", eps_ad, 0.0), ("adaptive+trunc", eps_ad, 1e-2)] + [(f"truncate-{tau:g}", 1e-6, tau) for tau in TAUS]
                for name, eps, tau in variants:
                    t0 = time.time(); Wi, rank, wmin = inv_trunc(Wn, eps, tau)
                    Kt_tr, Kt_te = Cn @ Wi @ Cn.T, Cn_te @ Wi @ Cn.T
                    res = svm_precomputed(Kt_tr, ytr, Kt_te, yte, C); met = approximation_metrics(K_tr, Kt_tr)
                    lo, hi = wilson_ci(int(round(res["acc"] * n_te)), n_te)
                    rows.append(dict(task=P["task"].key, seed=seed, m=m, shots=shots, variant=name, eps=eps, tau=tau, rank_kept=rank, min_eig_Weps=wmin, sigma_W=sigma_W,
                                     acc=res["acc"], wilson_lo=lo, wilson_hi=hi, kta_train=kta(Kt_tr, ytr), **met, Wpinv_norm=float(np.linalg.norm(Wi, 2)), runtime=time.time() - t0))
        pd.DataFrame(rows).to_csv(ROOT / "results" / "noise_fix_per_seed.csv", index=False)
        d = pd.DataFrame([r for r in rows if r["seed"] == seed and r["task"] == P["task"].key and r["m"] == 60])
        print(f"[{P['task'].key} s{seed}] m=60: " + " | ".join(f"{v} S={s}: {a:.3f}" for (v, s), a in d.groupby(["variant", "shots"], dropna=False)["acc"].mean().items() if v in ("plain-pinv", "psd-project", "adaptive", "truncate-0.01")), flush=True)
print("done", len(rows))
