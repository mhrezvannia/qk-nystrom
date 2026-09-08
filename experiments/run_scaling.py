"""Scaling study on fashion_0v1: N in {600, ..., 9600}.  theta, a, C are tuned once at N = 600 (same seed) and held
fixed, so that only N changes.  We measure wall-clock of every stage, memory of the Gram matrix, circuit-evaluation
counts, E_F (while the exact Gram matrix fits in memory) and test accuracy on the full official test split.
Nyström is evaluated in its scalable form: explicit features Phi = C W_eps^{-1/2} and a linear SVM, so the N x N
matrix is never formed."""
import sys, os, time, copy, tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from qkn.pipeline import load_config, prepare, tune_quantum_kernel, ROOT
from qkn.kernels import fidelity_kernel, CircuitCounter
from qkn.nystrom import select_landmarks, approximation_metrics
from qkn.data import Preprocessor, make_task, load_split
from qkn.metrics import wilson_ci

cfg = load_config(); sc = cfg["scaling"]; rows = []
tspec = [t for t in cfg["data"]["tasks"] if "classes" in t and f"{t['dataset']}_{t['classes'][0]}v{t['classes'][1]}" == sc["task"]][0]
for seed in sc["seeds"]:
    P = prepare(tspec, seed, cfg)
    tq = tune_quantum_kernel(P, cfg, cfg["feature_map"]["primary"], cfg["feature_map"]["layers"], cfg["feature_map"]["entangle"])
    fm, th, a, C = tq["fmap"], tq["theta"], tq["a"], tq["C"]
    for N in sc["n_values"]:
        cfgN = copy.deepcopy(cfg); cfgN["data"]["n_train"] = N
        PN = prepare(tspec, seed, cfgN)  # PCA/scaler re-fitted on the N training points (train only)
        Xtr, Xte, ytr, yte = Preprocessor.to_angles(PN["U_train"], a), Preprocessor.to_angles(PN["U_test"], a), PN["y_train"], PN["y_test"]
        n_te = len(yte)
        t0 = time.time(); S_tr = fm.states(Xtr, th); S_te = fm.states(Xte, th); t_states = time.time() - t0
        # ---- exact
        if N <= sc["exact_max_n"]:
            tracemalloc.start(); t0 = time.time(); K_tr = fidelity_kernel(S_tr, S_tr); K_te = fidelity_kernel(S_te, S_tr); t_gram = time.time() - t0
            t0 = time.time(); clf = SVC(kernel="precomputed", C=C).fit(K_tr, ytr); acc = accuracy_score(yte, clf.predict(K_te)); t_svm = time.time() - t0
            peak = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
            lo, hi = wilson_ci(int(round(acc * n_te)), n_te)
            rows.append(dict(task=sc["task"], seed=seed, N=N, method="q_exact", m=None, acc=acc, wilson_lo=lo, wilson_hi=hi, t_states=t_states, t_kernel=t_gram, t_svm=t_svm,
                             t_total=t_states + t_gram + t_svm, gram_bytes=K_tr.nbytes, peak_mem_MB=peak, **CircuitCounter.exact(N, n_te).as_dict()))
        else:
            K_tr = None
        # ---- Nyström (features)
        for label, m in (("m_fixed", sc["m_fixed"]), ("m_sqrtN", int(round(sc["m_sqrt_factor"] * np.sqrt(N))))):
            tracemalloc.start(); t0 = time.time(); Z, _ = select_landmarks(Xtr, m, "kmeans_centroids", np.random.default_rng(seed)); t_lm = time.time() - t0
            t0 = time.time(); S_Z = fm.states(Z, th); C_tr = fidelity_kernel(S_tr, S_Z); W = fidelity_kernel(S_Z, S_Z); C_te = fidelity_kernel(S_te, S_Z)
            w, V = np.linalg.eigh(W + cfg["nystrom"]["eps"] * np.eye(m)); w = np.clip(w, 1e-12, None)
            Wm12 = (V / np.sqrt(w)) @ V.T; Phi_tr, Phi_te = C_tr @ Wm12, C_te @ Wm12; t_ny = time.time() - t0
            t0 = time.time(); clf = SVC(kernel="linear", C=C).fit(Phi_tr, ytr); acc = accuracy_score(yte, clf.predict(Phi_te)); t_svm = time.time() - t0
            peak = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
            met = approximation_metrics(K_tr, Phi_tr @ Phi_tr.T) if K_tr is not None else {}
            lo, hi = wilson_ci(int(round(acc * n_te)), n_te)
            rows.append(dict(task=sc["task"], seed=seed, N=N, method=f"q_nystrom_{label}", m=m, acc=acc, wilson_lo=lo, wilson_hi=hi, t_states=t_states, t_landmarks=t_lm,
                             t_kernel=t_ny, t_svm=t_svm, t_total=t_states + t_lm + t_ny + t_svm, gram_bytes=Phi_tr.nbytes, peak_mem_MB=peak, **met,
                             **CircuitCounter.nystrom(N, n_te, m, False).as_dict()))
        pd.DataFrame(rows).to_csv(ROOT / "results" / "scaling_per_seed.csv", index=False)
        print(f"seed {seed} N={N}: " + " | ".join(f"{r['method']}(m={r['m']}) acc={r['acc']:.4f} t={r['t_total']:.1f}s circ={r['circ_train']:,}" for r in rows if r['N'] == N and r['seed'] == seed), flush=True)
print("done")
