"""Finite-shot and depolarising-noise study of the exact and Nyström-compressed quantum kernel.

Hyper-parameters (a, theta, C) are tuned NOISE-FREE on the validation split (stated limitation); noise is then
injected into every fidelity that a device would have to estimate: train-landmark, landmark-landmark and
test-landmark blocks (Nyström) or train-train (symmetric, diagonal known) and test-train blocks (exact).
For each configuration we record accuracy, kernel-estimation error, Nyström error w.r.t. the noise-free exact
kernel, cond(W_eps), ||W_eps^+||_2, and the noise amplification ratio  ||K~_noisy - K~_clean||_F / ||C_noisy - C||_F.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from qkn.pipeline import load_config, prepare, tune_quantum_kernel, quantum_states, svm_precomputed, ROOT
from qkn.kernels import fidelity_kernel, shot_noise, depolarize, CircuitCounter
from qkn.nystrom import Nystrom, select_landmarks, approximation_metrics
from qkn.metrics import kta, wilson_ci
from qkn.data import Preprocessor

cfg = load_config(); nc = cfg["noise"]; n_q = cfg["data"]["n_qubits"]
task_specs = [t for t in cfg["data"]["tasks"] if "classes" in t and f"{t['dataset']}_{t['classes'][0]}v{t['classes'][1]}" in nc["tasks"]]
rows = []


def noisy(K, shots, lam, rng, symmetric=False):
    Kd = depolarize(K, lam, n_q) if lam > 0 else K
    Kh = shot_noise(Kd, shots, rng, symmetric=symmetric and lam == 0)
    if symmetric and lam > 0:  # symmetric estimation but the diagonal must now be measured too
        iu = np.triu_indices_from(Kh, k=1); Kh[(iu[1], iu[0])] = Kh[iu]
    return Kh


for tspec in task_specs:
    for seed in cfg["seeds"]:
        P = prepare(tspec, seed, cfg); ytr, yte, n_te = P["y_train"], P["y_test"], P["n_test"]
        tq = tune_quantum_kernel(P, cfg, cfg["feature_map"]["primary"], cfg["feature_map"]["layers"], cfg["feature_map"]["entangle"])
        fm, th, a, C = tq["fmap"], tq["theta"], tq["a"], tq["C"]
        S_tr, S_te = quantum_states(P, fm, th, a); K_tr, K_te = fidelity_kernel(S_tr, S_tr), fidelity_kernel(S_te, S_tr)
        Xtr_a = Preprocessor.to_angles(P["U_train"], a)
        landmarks = {m: select_landmarks(Xtr_a, m, nc["strategy"], np.random.default_rng(seed * 7919 + m))[0] for m in nc["m_values"]}
        for shots in nc["shots"]:
            for lam in nc["depolarizing"]:
                rng = np.random.default_rng(seed * 100003 + (shots or 0) + int(lam * 1000))
                # ---- exact kernel under noise
                t0 = time.time(); Kn_tr = noisy(K_tr, shots, lam, rng, symmetric=True); Kn_te = noisy(K_te, shots, lam, rng)
                res = svm_precomputed(Kn_tr, ytr, Kn_te, yte, C)
                lo, hi = wilson_ci(int(round(res["acc"] * n_te)), n_te)
                rows.append(dict(task=P["task"].key, seed=seed, method="q_exact", shots=shots, depolarizing=lam, m=None, eps=None, normalize=None,
                                 acc=res["acc"], wilson_lo=lo, wilson_hi=hi, kta_train=kta(Kn_tr, ytr),
                                 est_err=np.linalg.norm(Kn_tr - K_tr) / np.linalg.norm(K_tr), E_F=np.linalg.norm(Kn_tr - K_tr) / np.linalg.norm(K_tr),
                                 min_eig_Ktr=float(np.linalg.eigvalsh(Kn_tr).min()), runtime=time.time() - t0,
                                 **CircuitCounter.exact(P["n_train"], n_te, shots).as_dict()))
                # ---- Nyström under noise
                for m, Z in landmarks.items():
                    S_Z = fm.states(Z, th)
                    C_trZ, W, C_teZ = fidelity_kernel(S_tr, S_Z), fidelity_kernel(S_Z, S_Z), fidelity_kernel(S_te, S_Z)
                    Cn_trZ, Wn, Cn_teZ = noisy(C_trZ, shots, lam, rng), noisy(W, shots, lam, rng, symmetric=True), noisy(C_teZ, shots, lam, rng)
                    for eps in nc["eps_values"]:
                        for norm in nc["normalize_values"]:
                            t0 = time.time()
                            ny_clean = Nystrom(eps, cfg["nystrom"]["rcond"], norm).fit(W); Kt_clean = ny_clean.transform(C_trZ)
                            ny = Nystrom(eps, cfg["nystrom"]["rcond"], norm).fit(Wn)
                            Kt_tr, Kt_te = ny.transform(Cn_trZ), ny.transform(Cn_teZ, Cn_trZ)
                            res = svm_precomputed(Kt_tr, ytr, Kt_te, yte, C); met = approximation_metrics(K_tr, Kt_tr)
                            lo, hi = wilson_ci(int(round(res["acc"] * n_te)), n_te)
                            dC = np.linalg.norm(Cn_trZ - C_trZ); dW = np.linalg.norm(Wn - W)
                            rows.append(dict(task=P["task"].key, seed=seed, method="q_nystrom", shots=shots, depolarizing=lam, m=m, eps=eps, normalize=norm,
                                             acc=res["acc"], wilson_lo=lo, wilson_hi=hi, kta_train=kta(Kt_tr, ytr), **met,
                                             E_F_clean=approximation_metrics(K_tr, Kt_clean)["E_F"],
                                             noise_only_err=np.linalg.norm(Kt_tr - Kt_clean) / np.linalg.norm(K_tr),
                                             amplification=(np.linalg.norm(Kt_tr - Kt_clean) / (dC + dW)) if (dC + dW) > 0 else 0.0,
                                             est_err_C=dC / np.linalg.norm(C_trZ), est_err_W=dW / np.linalg.norm(W),
                                             cond_W=ny.cond_W, min_eig_W=ny.min_eig_W, Wpinv_norm=float(np.linalg.norm(ny.W_pinv, 2)),
                                             min_eig_Ktr=float(np.linalg.eigvalsh((Kt_tr + Kt_tr.T) / 2).min()), runtime=time.time() - t0,
                                             **CircuitCounter.nystrom(P["n_train"], n_te, m, False, shots).as_dict()))
        pd.DataFrame(rows).to_csv(ROOT / "results" / "noise_per_seed.csv", index=False)
        sub = [r for r in rows if r["seed"] == seed and r["task"] == P["task"].key and r["depolarizing"] == 0 and r["eps"] in (None, 1e-6) and r["normalize"] in (None, False)]
        print(f"[{P['task'].key} s{seed}] " + " | ".join(f"{r['method']}{'' if r['m'] is None else '(m='+str(r['m'])+')'} S={r['shots']}: {r['acc']:.3f}" for r in sub), flush=True)
print("done", len(rows))
