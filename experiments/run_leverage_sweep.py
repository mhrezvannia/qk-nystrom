"""Ridge-parameter sweep for ridge-leverage-score landmark sampling (Reviewer 2a).
For every main task and seed, with the trained covariant kernel at its selected (a, theta, C):
  leverage scores l_i(lambda) = [K (K + lambda I)^-1]_ii for lambda in LAMBDAS (absolute, N = 600 -> lambda/N in the table),
  m in {10, 20, 60}, plus uniform and k-means-centroid references with the same RNG stream.
Records E_F, E_spec, accuracy and the entropy of the sampling distribution (uniform: log N)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from qkn.pipeline import load_config, prepare, tune_quantum_kernel, quantum_states, svm_precomputed, ROOT
from qkn.kernels import fidelity_kernel
from qkn.nystrom import Nystrom, select_landmarks, approximation_metrics
from qkn.data import Preprocessor

cfg = load_config(); LAMBDAS = [1e-3, 1e-2, 1e-1, 0.6, 6.0, 60.0]; MS = [10, 20, 60]  # 0.6 = the main-study value (1e-3 * N)
rows = []
for tspec in cfg["data"]["tasks"]:
    for seed in cfg["seeds"]:
        P = prepare(tspec, seed, cfg); ytr, yte = P["y_train"], P["y_test"]; N = P["n_train"]
        tq = tune_quantum_kernel(P, cfg, cfg["feature_map"]["primary"], cfg["feature_map"]["layers"], cfg["feature_map"]["entangle"])
        fm, th, a, C = tq["fmap"], tq["theta"], tq["a"], tq["C"]
        S_tr, S_te = quantum_states(P, fm, th, a); K_tr = fidelity_kernel(S_tr, S_tr); Xtr_a = Preprocessor.to_angles(P["U_train"], a)
        def evaluate(Z, label, lam, m, entropy):
            S_Z = fm.states(Z, th); ny = Nystrom(cfg["nystrom"]["eps"], cfg["nystrom"]["rcond"]).fit(fidelity_kernel(S_Z, S_Z))
            C_trZ = fidelity_kernel(S_tr, S_Z); Kt_tr = ny.transform(C_trZ); Kt_te = ny.transform(fidelity_kernel(S_te, S_Z), C_trZ)
            res = svm_precomputed(Kt_tr, ytr, Kt_te, yte, C); met = approximation_metrics(K_tr, Kt_tr)
            rows.append(dict(task=P["task"].key, seed=seed, strategy=label, ridge=lam, ridge_over_N=(lam / N if lam is not None else None), m=m,
                             sampling_entropy=entropy, acc=res["acc"], **met, cond_W_eff=ny.cond_W_eff, min_eig_W=ny.min_eig_W))
        for m in MS:
            rng_u = np.random.default_rng(seed * 7919 + m)
            Z, _ = select_landmarks(Xtr_a, m, "uniform", rng_u); evaluate(Z, "uniform", None, m, float(np.log(N)))
            Z, _ = select_landmarks(Xtr_a, m, "kmeans_centroids", np.random.default_rng(seed * 7919 + m)); evaluate(Z, "kmeans_centroids", None, m, np.nan)
            for lam in LAMBDAS:
                lev = np.diag(K_tr @ np.linalg.solve(K_tr + lam * np.eye(N), np.eye(N)))
                p = np.clip(lev, 1e-12, None); p /= p.sum(); H = float(-(p * np.log(p)).sum())
                idx = np.random.default_rng(seed * 7919 + m).choice(N, m, replace=False, p=p)
                evaluate(Xtr_a[idx], "leverage", lam, m, H)
        pd.DataFrame(rows).to_csv(ROOT / "results" / "leverage_sweep_per_seed.csv", index=False)
        d = pd.DataFrame([r for r in rows if r["seed"] == seed and r["task"] == P["task"].key and r["m"] == 20])
        print(f"[{P['task'].key} s{seed}] m=20 E_F: " + " | ".join(f"{r.strategy}{'' if r.ridge is None else f'(λ={r.ridge:g})'}={100*r.E_F:.1f}%" for r in d.itertuples()), flush=True)
print("done", len(rows))
