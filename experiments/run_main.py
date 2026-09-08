"""Main campaign: every task x seed x method x landmark strategy x m.  Writes results/main_per_seed.csv
(one row per configuration; nothing aggregated here).  All hyper-parameters come from configs/main.yaml.

Methods
  svm_linear, svm_rbf, logreg                       classical, same PCA-8 unit features
  q_exact[map]                                      exact fidelity kernel SVM (map in {covariant_torus, product_cosine})
  q_nystrom[map]/strategy/m                         Nyström-compressed fidelity kernel
  rbf_nystrom/strategy/m                            classical RBF kernel with the SAME landmarks
  rff_rbf/m                                         random Fourier features (D = m) for the RBF kernel
C is selected on the validation split for every method and every m (landmarks for tuning drawn from the fit split).
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from sklearn.svm import SVC
from qkn.pipeline import (load_config, prepare, classical_baselines, tune_quantum_kernel, quantum_states, select_C,
                          svm_precomputed, env_info, ROOT)
from qkn.feature_maps import make_map
from qkn.kernels import fidelity_kernel, kernel_matrix, CircuitCounter, random_fourier_features, rff_transform
from qkn.nystrom import Nystrom, select_landmarks, approximation_metrics
from qkn.metrics import kta, wilson_ci, kernel_spectrum_stats, geometric_difference
from qkn.data import Preprocessor

ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default="all"); ap.add_argument("--seeds", default="all")
ap.add_argument("--out", default="main_per_seed.csv"); args = ap.parse_args()
cfg = load_config(); OUT = ROOT / "results" / args.out
tasks = cfg["data"]["tasks"] if args.tasks == "all" else [cfg["data"]["tasks"][int(i)] for i in args.tasks.split(",")]
seeds = cfg["seeds"] if args.seeds == "all" else [int(s) for s in args.seeds.split(",")]
MAPS = [("covariant_torus", cfg["feature_map"]["layers"], True), ("product_cosine", 1, False)]
rows = []


def base_row(P, **kw):
    r = dict(dataset=P["task"].dataset, task=P["task"].key, task_name=P["task"].name, seed=P["seed"], n_train=P["n_train"],
             n_test=P["n_test"], pca_evr=P["pca_evr"])
    r.update(kw); return r


def add_acc(r, res, n_test):
    r.update(res); lo, hi = wilson_ci(int(round(res["acc"] * n_test)), n_test); r.update(wilson_lo=lo, wilson_hi=hi)


def nystrom_eval(K_tr_Z, K_Z, K_te_Z, K_full_tr, y_tr, y_te, C, ny_cfg):
    ny = Nystrom(ny_cfg["eps"], ny_cfg["rcond"], ny_cfg["normalize"]).fit(K_Z)
    Kt_tr = ny.transform(K_tr_Z); Kt_te = ny.transform(K_te_Z, K_tr_Z)
    res = svm_precomputed(Kt_tr, y_tr, Kt_te, y_te, C)
    met = approximation_metrics(K_full_tr, Kt_tr) if K_full_tr is not None else {}
    return res, met, dict(cond_W=ny.cond_W, min_eig_W=ny.min_eig_W, kta_train=kta(Kt_tr, y_tr))


def tune_C_nystrom(kfun, Xfit, Xval, yfit, yval, m, strategy, rng, ny_cfg, C_grid, K_fit_full=None):
    Z, _ = select_landmarks(Xfit, m, strategy, rng, K_full=K_fit_full)
    ny = Nystrom(ny_cfg["eps"], ny_cfg["rcond"], ny_cfg["normalize"]).fit(kfun(Z, Z))
    Cf, Cv = kfun(Xfit, Z), kfun(Xval, Z)
    return select_C(ny.transform(Cf), yfit, ny.transform(Cv, Cf), yval, C_grid)[0]


for tspec in tasks:
    for seed in seeds:
        T0 = time.time(); P = prepare(tspec, seed, cfg)
        ytr, yte, i_fit, i_val = P["y_train"], P["y_test"], P["i_fit"], P["i_val"]
        n_tr, n_te = P["n_train"], P["n_test"]
        rng = np.random.default_rng(seed)
        # ------------------------------------------------ classical baselines
        cb = classical_baselines(P, cfg); gamma_rbf = cb.pop("_rbf_gamma")
        for meth, r in cb.items():
            row = base_row(P, method=meth, map="classical", strategy=None, m=None, C=r["C"], gamma=r.get("gamma"), runtime=r["runtime"])
            add_acc(row, {k: r[k] for k in ("acc", "bal_acc", "f1")}, n_te); rows.append(row)
        U_tr, U_te = P["U_train"], P["U_test"]
        K_rbf_tr = kernel_matrix("rbf", U_tr, U_tr, gamma_rbf); K_lin_tr = kernel_matrix("linear", U_tr, U_tr)
        # ------------------------------------------------ quantum maps
        for map_name, L, ent in MAPS:
            t0 = time.time(); tq = tune_quantum_kernel(P, cfg, map_name, L, ent, train_theta=True)
            fm, th, a, C_q = tq["fmap"], tq["theta"], tq["a"], tq["C"]
            S_tr, S_te = quantum_states(P, fm, th, a)
            K_tr, K_te = fidelity_kernel(S_tr, S_tr), fidelity_kernel(S_te, S_tr)
            t_exact = time.time() - t0
            res = svm_precomputed(K_tr, ytr, K_te, yte, C_q)
            spec = kernel_spectrum_stats(K_tr)
            gd = {f"g_vs_rbf_lam{lam}": geometric_difference(K_rbf_tr, K_tr, lam) for lam in cfg["geometric_difference"]["lambdas"]}
            gd.update({f"g_vs_linear_lam{lam}": geometric_difference(K_lin_tr, K_tr, lam) for lam in cfg["geometric_difference"]["lambdas"]})
            row = base_row(P, method=f"q_exact[{map_name}]", map=map_name, strategy=None, m=None, a=a, C=C_q, theta=json.dumps(np.round(th, 5).tolist()),
                           kta_init=tq["kta_init"], kta_final=tq["kta_final"], kta_evals=tq["n_evals"], val_acc=tq["val_acc"], kta_train=kta(K_tr, ytr),
                           runtime=t_exact, **CircuitCounter.exact(n_tr, n_te).as_dict(), **spec, **gd)
            add_acc(row, res, n_te); rows.append(row)
            Xtr_a, Xte_a = Preprocessor.to_angles(U_tr, a), Preprocessor.to_angles(U_te, a)
            kq = lambda A, B: fidelity_kernel(fm.states(A, th), fm.states(B, th))
            kr = lambda A, B: kernel_matrix("rbf", A / a, B / a, gamma_rbf)  # classical RBF on the same landmarks (unit space)
            K_fit_full = K_tr[np.ix_(i_fit, i_fit)]
            for strategy in cfg["nystrom"]["strategies"]:
                for m in cfg["nystrom"]["m_grid"]:
                    if m > n_tr: continue
                    if m > len(i_fit) and strategy != "uniform": continue
                    lm_rng = np.random.default_rng(seed * 7919 + m)
                    # --- C selection on validation (landmarks from the fit split only)
                    C_ny = tune_C_nystrom(kq, Xtr_a[i_fit], Xtr_a[i_val], ytr[i_fit], ytr[i_val], min(m, len(i_fit)), strategy,
                                          np.random.default_rng(seed * 7919 + m), cfg["nystrom"], cfg["svm"]["C_grid"], K_fit_full)
                    # --- final landmarks from the full training set
                    t0 = time.time()
                    Z, idx = select_landmarks(Xtr_a, m, strategy, lm_rng, K_full=K_tr)
                    S_Z = fm.states(Z, th)
                    K_trZ, K_Z, K_teZ = fidelity_kernel(S_tr, S_Z), fidelity_kernel(S_Z, S_Z), fidelity_kernel(S_te, S_Z)
                    res, met, extra = nystrom_eval(K_trZ, K_Z, K_teZ, K_tr, ytr, yte, C_ny, cfg["nystrom"])
                    cc = CircuitCounter.nystrom(n_tr, n_te, m, landmarks_are_data=idx is not None)
                    row = base_row(P, method=f"q_nystrom[{map_name}]", map=map_name, strategy=strategy, m=m, a=a, C=C_ny,
                                   landmarks_are_data=idx is not None, runtime=time.time() - t0, **met, **extra, **cc.as_dict())
                    add_acc(row, res, n_te); rows.append(row)
                    # --- classical RBF Nyström with the SAME landmarks (only once, for the primary map loop)
                    if map_name == MAPS[0][0]:
                        C_r = tune_C_nystrom(kr, Xtr_a[i_fit], Xtr_a[i_val], ytr[i_fit], ytr[i_val], min(m, len(i_fit)), strategy,
                                             np.random.default_rng(seed * 7919 + m), cfg["nystrom"], cfg["svm"]["C_grid"], K_rbf_tr[np.ix_(i_fit, i_fit)])
                        t0 = time.time()
                        res, met, extra = nystrom_eval(kr(Xtr_a, Z), kr(Z, Z), kr(Xte_a, Z), K_rbf_tr, ytr, yte, C_r, cfg["nystrom"])
                        row = base_row(P, method="rbf_nystrom", map="classical", strategy=strategy, m=m, C=C_r, gamma=gamma_rbf,
                                       landmarks_are_data=idx is not None, runtime=time.time() - t0, **met, **extra)
                        add_acc(row, res, n_te); rows.append(row)
            print(f"  {map_name} done ({time.time()-T0:.0f}s)", flush=True)
        # ------------------------------------------------ random Fourier features (RBF), D = m
        for m in cfg["nystrom"]["rff_dims"]:
            rr = np.random.default_rng(seed * 31 + m)
            Ff, (Wf, b) = random_fourier_features(U_tr[i_fit], m, gamma_rbf, rr)
            Fv = rff_transform(U_tr[i_val], Wf, b)
            C_f = select_C(Ff @ Ff.T, ytr[i_fit], Fv @ Ff.T, ytr[i_val], cfg["svm"]["C_grid"])[0]
            t0 = time.time(); Ftr, Fte = rff_transform(U_tr, Wf, b), rff_transform(U_te, Wf, b)
            res = svm_precomputed(Ftr @ Ftr.T, ytr, Fte @ Ftr.T, yte, C_f)
            met = approximation_metrics(K_rbf_tr, Ftr @ Ftr.T)
            row = base_row(P, method="rff_rbf", map="classical", strategy="rff", m=m, C=C_f, gamma=gamma_rbf, runtime=time.time() - t0, **met)
            add_acc(row, res, n_te); rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        q = [r for r in rows if r["seed"] == seed and r["task"] == P["task"].key]
        summ = {r["method"]: round(r["acc"], 4) for r in q if r["m"] is None}
        print(f"[{P['task'].key} seed {seed}] {time.time()-T0:.0f}s  {summ}", flush=True)

meta = dict(env=env_info(cfg), config=cfg, n_rows=len(rows))
with open(OUT.with_suffix(".meta.json"), "w") as f:
    json.dump(meta, f, indent=1, default=str)
print("written", OUT, len(rows), "rows")
