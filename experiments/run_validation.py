"""Phase D — numerical validation of candidate feature maps on real data (selection gate).

For each candidate map (and the previous product-cosine construction) on every task, seed 0..2:
  theta-dependence, entanglement-dependence, PSD, spectrum, KTA before/after training, distance from
  classical kernels (linear / RBF / product-cosine), geometric difference vs RBF, validation accuracy.
Writes results/validation_feature_maps.csv.  The primary map is chosen by the rule documented in
docs/phaseD_selection.md (highest mean validation accuracy, ties broken by KTA), subject to passing all tests.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd
from qkn.pipeline import load_config, prepare, tune_quantum_kernel, ROOT
from qkn.feature_maps import make_map
from qkn.kernels import fidelity_kernel, kernel_matrix
from qkn.metrics import kta, kernel_spectrum_stats, kernel_distance, geometric_difference
from qkn.data import Preprocessor

cfg = load_config()
rows = []
maps = [("product_cosine", 1, False), ("covariant_torus", 2, True), ("reuploading", 2, True), ("zz_reuploading", 2, True)]
for tspec in cfg["data"]["tasks"]:
    for seed in cfg["seeds"][:3]:
        P = prepare(tspec, seed, cfg)
        i_fit, i_val = P["i_fit"], P["i_val"]; ytr = P["y_train"]
        g0 = 1.0 / (P["U_train"].shape[1] * P["U_train"].var())
        for name, L, ent in maps:
            t0 = time.time()
            res = tune_quantum_kernel(P, cfg, name, L, ent, train_theta=True)
            fm, th, a = res["fmap"], res["theta"], res["a"]
            Xfit = Preprocessor.to_angles(P["U_train"][i_fit], a)
            S = fm.states(Xfit, th); K = fidelity_kernel(S, S)
            # dependence tests at the SELECTED a and trained theta
            rng = np.random.default_rng(99)
            if fm.n_params:
                K_th = fidelity_kernel(*(2 * [fm.states(Xfit, th + rng.normal(0, 0.5, th.shape))]))
                fm_ne = make_map(name, fm.n, L, False); K_ne = fidelity_kernel(*(2 * [fm_ne.states(Xfit, th)]))
                K_rand = fidelity_kernel(*(2 * [fm.states(Xfit, fm.init_params(rng))]))
                dth, dent, kta_rand = np.abs(K - K_th).max(), np.abs(K - K_ne).max(), kta(K_rand, ytr[i_fit])
            else:
                dth = dent = 0.0; kta_rand = np.nan
            spec = kernel_spectrum_stats(K)
            K_lin = kernel_matrix("linear", P["U_train"][i_fit], P["U_train"][i_fit])
            K_rbf = kernel_matrix("rbf", P["U_train"][i_fit], P["U_train"][i_fit], g0)
            K_pc = kernel_matrix("product_cosine", Xfit, Xfit)
            rows.append(dict(task=P["task"].key, seed=seed, map=name, layers=L, entangle=ent, a=a, n_params=fm.n_params,
                             val_acc=res["val_acc"], kta_trained=res["kta_fit"], kta_init=res["kta_init"], kta_random_theta=kta_rand,
                             kta_rbf=kta(K_rbf, ytr[i_fit]), kta_linear=kta(K_lin, ytr[i_fit]), kta_product_cosine=kta(K_pc, ytr[i_fit]),
                             max_dK_theta=dth, max_dK_entangle=dent,
                             dist_linear=kernel_distance(K, K_lin), dist_rbf=kernel_distance(K, K_rbf), dist_product_cosine=kernel_distance(K, K_pc),
                             g_vs_rbf=geometric_difference(K_rbf, K, 0.01), g_vs_linear=geometric_difference(K_lin, K, 0.01),
                             kta_evals=res["n_evals"], runtime=time.time() - t0, **spec))
            print(f"{P['task'].key:18s} s{seed} {name:16s} a={a:.2f} val_acc={res['val_acc']:.3f} KTA {res['kta_init']:.3f}->{res['kta_fit']:.3f} "
                  f"dK_theta={dth:.3f} dK_ent={dent:.3f} effrank={spec['eff_rank']:.1f} g_rbf={rows[-1]['g_vs_rbf']:.2f} ({time.time()-t0:.1f}s)", flush=True)
df = pd.DataFrame(rows); (ROOT / "results").mkdir(exist_ok=True)
df.to_csv(ROOT / "results" / "validation_feature_maps.csv", index=False)
print("\n=== mean over tasks & seeds ===")
print(df.groupby("map")[["val_acc", "kta_trained", "kta_init", "max_dK_theta", "max_dK_entangle", "eff_rank", "dist_rbf", "dist_product_cosine", "g_vs_rbf", "runtime"]].mean().round(3).to_string())
