"""Bandwidth / feature-scaling study: a in {pi/4, pi/2, pi, 3pi/2, 2pi}.
For the product-cosine kernel (untrained) and the covariant torus kernel (trained per a):
kernel spectrum, off-diagonal similarity distribution, KTA, exact test accuracy, and Nyström E_F / accuracy at m = 60.
Tasks: fashion_0v1 and fashion_2v4; all seeds."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from qkn.pipeline import load_config, prepare, tune_quantum_kernel, quantum_states, svm_precomputed, ROOT
from qkn.kernels import fidelity_kernel
from qkn.nystrom import Nystrom, select_landmarks, approximation_metrics
from qkn.metrics import kta, kernel_spectrum_stats, wilson_ci
from qkn.data import Preprocessor

cfg = load_config(); rows = []
for tspec in cfg["data"]["tasks"][:2]:
    for seed in cfg["seeds"]:
        P = prepare(tspec, seed, cfg); ytr, yte = P["y_train"], P["y_test"]
        for a in cfg["preprocessing"]["bandwidth_study_ranges"]:
            for map_name, L, ent in [("product_cosine", 1, False), ("covariant_torus", 2, True)]:
                t0 = time.time()
                tq = tune_quantum_kernel(P, cfg, map_name, L, ent, train_theta=True, scaling_ranges=[a])
                fm, th = tq["fmap"], tq["theta"]
                S_tr, S_te = quantum_states(P, fm, th, a); K_tr, K_te = fidelity_kernel(S_tr, S_tr), fidelity_kernel(S_te, S_tr)
                res = svm_precomputed(K_tr, ytr, K_te, yte, tq["C"])
                Z, _ = select_landmarks(Preprocessor.to_angles(P["U_train"], a), 60, "kmeans_centroids", np.random.default_rng(seed))
                S_Z = fm.states(Z, th); ny = Nystrom(cfg["nystrom"]["eps"], cfg["nystrom"]["rcond"]).fit(fidelity_kernel(S_Z, S_Z))
                Kt_tr = ny.transform(fidelity_kernel(S_tr, S_Z)); Kt_te = ny.transform(fidelity_kernel(S_te, S_Z), fidelity_kernel(S_tr, S_Z))
                res_ny = svm_precomputed(Kt_tr, ytr, Kt_te, yte, tq["C"]); met = approximation_metrics(K_tr, Kt_tr)
                lo, hi = wilson_ci(int(round(res["acc"] * P["n_test"])), P["n_test"])
                rows.append(dict(task=P["task"].key, seed=seed, map=map_name, a=a, a_over_pi=a / np.pi, C=tq["C"], acc=res["acc"], wilson_lo=lo, wilson_hi=hi,
                                 val_acc=tq["val_acc"], kta_train=kta(K_tr, ytr), acc_nystrom_m60=res_ny["acc"], cond_W=ny.cond_W, **met,
                                 runtime=time.time() - t0, **kernel_spectrum_stats(K_tr)))
                print(f"{P['task'].key} s{seed} {map_name:16s} a/pi={a/np.pi:.2f} acc={res['acc']:.4f} kta={rows[-1]['kta_train']:.3f} offdiag={rows[-1]['offdiag_mean']:.3f} effrank={rows[-1]['eff_rank']:.1f} E_F={met['E_F']:.3f}", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "results" / "bandwidth_per_seed.csv", index=False)
print("done", len(rows))
