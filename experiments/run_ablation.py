"""Ablation: which component of the feature map causes the observed behaviour?
Variants (all at the scaling range a fixed to pi so that only the map changes):
  covariant_torus  L in {1,2} x entangle in {T,F} x trained in {T,F(random theta)}
  covariant_torus  theta = pi/2 everywhere, no entanglement  (== product_cosine, the previous construction)
  reuploading      L in {1,2}, entangle in {T,F}, trained
  zz_reuploading   L = 2, trained
For each variant: exact-kernel test accuracy, KTA, spectrum, E_F/accuracy at m = 60 (kmeans_centroids), runtime,
kernel distance to the primary trained kernel.  Tasks: first three of config; all seeds.
"""
import sys, os, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from qkn.pipeline import load_config, prepare, tune_quantum_kernel, quantum_states, svm_precomputed, select_C, ROOT
from qkn.kernels import fidelity_kernel
from qkn.nystrom import Nystrom, select_landmarks, approximation_metrics
from qkn.metrics import kta, wilson_ci, kernel_spectrum_stats, kernel_distance
from qkn.data import Preprocessor
from qkn.feature_maps import CovariantTorus

ap = argparse.ArgumentParser(); ap.add_argument("--tasks", default=None); ap.add_argument("--seeds", default=None); ap.add_argument("--out", default="ablation_per_seed.csv"); ap.add_argument("--variants", default="base", choices=["base", "alt", "all"]); args = ap.parse_args()
cfg = load_config(); A_FIXED = float(cfg["ablation"]["scaling_range"]); M_ABL = cfg["ablation"]["m"]
TASKS = [cfg["data"]["tasks"][int(i)] for i in (args.tasks.split(",") if args.tasks else cfg["ablation"]["tasks"])]
SEEDS = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["ablation"]["seeds"])]
VARIANTS = []
for L in (1, 2):
    for ent in (True, False):
        for trained in (True, False):
            VARIANTS.append(dict(map="covariant_torus", layers=L, entangle=ent, trained=trained))
VARIANTS.append(dict(map="covariant_torus", layers=1, entangle=False, trained=False, theta_fixed="pi/2"))  # == product cosine
for L in (1, 2):
    for ent in (True, False):
        VARIANTS.append(dict(map="reuploading", layers=L, entangle=ent, trained=True))
VARIANTS.append(dict(map="zz_reuploading", layers=2, entangle=True, trained=True))
# --- alternative training objective (docs/alternative_training_protocol.md), pre-registered before running
ALT = [dict(map="covariant_torus", layers=2, entangle=True, trained="regKTA"),   # KTA + lambda*log(erank)/log(n), lambda on validation
       dict(map="covariant_torus", layers=2, entangle=True, trained="valsel")]    # validation-selected among {untrained, KTA, regKTA}
RANK_LAMBDAS = cfg["ablation"].get("rank_lambdas", [0.1, 0.3, 1.0])
if args.variants == "alt":
    VARIANTS = [dict(map="covariant_torus", layers=2, entangle=True, trained=False), dict(map="covariant_torus", layers=2, entangle=True, trained=True)] + ALT
elif args.variants == "all":
    VARIANTS = VARIANTS + ALT
rows = []
for tspec in TASKS:
    for seed in SEEDS:
        P = prepare(tspec, seed, cfg); ytr, yte = P["y_train"], P["y_test"]
        K_primary = None; tq_store = {}
        for v in VARIANTS:
            t0 = time.time()
            theta_init = None; rank_lambda = 0.0; selected_from = None
            if v.get("theta_fixed") == "pi/2":
                theta_init = np.full(cfg["data"]["n_qubits"] * v["layers"], np.pi / 2)
            if v["trained"] == "regKTA":
                cands = []
                for lam in RANK_LAMBDAS:
                    tq_l = tune_quantum_kernel(P, cfg, v["map"], v["layers"], v["entangle"], train_theta=True, scaling_ranges=[A_FIXED], rank_lambda=lam)
                    tq_l["rank_lambda"] = lam; cands.append(tq_l)
                tq = max(cands, key=lambda c: (c["val_acc"], c["rank_lambda"]))  # ties -> larger lambda (pre-registered)
                rank_lambda = tq["rank_lambda"]; tq_store["regKTA"] = tq
            elif v["trained"] == "valsel":
                pool = {k: tq_store[k] for k in ("untrained", "KTA", "regKTA") if k in tq_store}
                order = ["untrained", "KTA", "regKTA"]
                selected_from = max(order, key=lambda k: (pool[k]["val_acc"], -order.index(k)))  # ties -> untrained (pre-registered)
                tq = pool[selected_from]; rank_lambda = tq.get("rank_lambda", 0.0)
            else:
                tq = tune_quantum_kernel(P, cfg, v["map"], v["layers"], v["entangle"], train_theta=bool(v["trained"]),
                                         scaling_ranges=[A_FIXED], theta_init=theta_init)
                if v["map"] == "covariant_torus" and v["layers"] == 2 and v["entangle"] and not v.get("theta_fixed"):
                    tq_store["KTA" if v["trained"] else "untrained"] = tq
            fm, th = tq["fmap"], tq["theta"]
            S_tr, S_te = quantum_states(P, fm, th, A_FIXED)
            K_tr, K_te = fidelity_kernel(S_tr, S_tr), fidelity_kernel(S_te, S_tr)
            res = svm_precomputed(K_tr, ytr, K_te, yte, tq["C"])
            if v["map"] == "covariant_torus" and v["layers"] == 2 and v["entangle"] and v["trained"] is True:
                K_primary = K_tr
            # Nyström at m = 60 with k-means centroids
            Xtr_a = Preprocessor.to_angles(P["U_train"], A_FIXED)
            Z, _ = select_landmarks(Xtr_a, M_ABL, "kmeans_centroids", np.random.default_rng(seed))
            S_Z = fm.states(Z, th); ny = Nystrom(cfg["nystrom"]["eps"], cfg["nystrom"]["rcond"]).fit(fidelity_kernel(S_Z, S_Z))
            Kt_tr = ny.transform(fidelity_kernel(S_tr, S_Z)); Kt_te = ny.transform(fidelity_kernel(S_te, S_Z), fidelity_kernel(S_tr, S_Z))
            res_ny = svm_precomputed(Kt_tr, ytr, Kt_te, yte, tq["C"]); met = approximation_metrics(K_tr, Kt_tr)
            lo, hi = wilson_ci(int(round(res["acc"] * P["n_test"])), P["n_test"])
            tr_label = v["trained"] if isinstance(v["trained"], str) else int(v["trained"])
            rows.append(dict(task=P["task"].key, seed=seed, variant=f"{v['map']}|L={v['layers']}|ent={int(v['entangle'])}|trained={tr_label}" + ("|theta=pi/2" if v.get("theta_fixed") else ""),
                             **{k: v.get(k) for k in ("map", "layers", "entangle")}, trained=str(tr_label), n_params=fm.n_params, a=A_FIXED, C=tq["C"],
                             rank_lambda=rank_lambda, selected_from=selected_from, theta=json.dumps(np.round(np.asarray(th), 5).tolist()), kta_evals=tq.get("n_evals"),
                             acc=res["acc"], bal_acc=res["bal_acc"], f1=res["f1"], wilson_lo=lo, wilson_hi=hi, val_acc=tq["val_acc"],
                             kta_train=kta(K_tr, ytr), kta_init=tq["kta_init"], kta_final=tq["kta_final"],
                             acc_nystrom_m60=res_ny["acc"], **met, cond_W=ny.cond_W, cond_W_eff=ny.cond_W_eff, min_eig_W=ny.min_eig_W,
                             dist_to_primary=kernel_distance(K_tr, K_primary) if K_primary is not None else np.nan,
                             runtime=time.time() - t0, **kernel_spectrum_stats(K_tr)))
            print(f"{P['task'].key} s{seed} {rows[-1]['variant']:52s} acc={res['acc']:.4f} kta={rows[-1]['kta_train']:.3f} E_F={met['E_F']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results" / args.out, index=False)
print("done", len(rows))
