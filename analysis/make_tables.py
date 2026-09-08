"""Generate every manuscript table (CSV/MD/LaTeX) and the statistical-test tables from results/*.csv.
No number in the manuscript is typed by hand: manuscript/build_manuscript.py pulls values from tables/*.csv."""
from common import *
from qkn.statistics import compare_table, paired_comparison, holm_correction
from qkn.metrics import wilson_ci

SCRIPT = "analysis/make_tables.py"
main = load("main_per_seed.csv"); SRC_MAIN = RES / "main_per_seed.csv"
main["m"] = main["m"].astype("Int64")


def cond_clean(x):
    """Condition-number convention: cond(W_eps) = lambda_max/lambda_min if lambda_min > 0 else inf.  Legacy CSVs stored
    lambda_max/max(lambda_min, 1e-300) (values up to ~1e302) for singular/indefinite W_eps; map those to inf."""
    x = np.asarray(x, float); x = np.where(x > 1e15, np.inf, x); return x


def fmt_cond(x):
    return "∞ (singular/indefinite)" if not np.isfinite(x) else f"{x:.1e}"


for _df in (main,):
    if "cond_W" in _df: _df["cond_W"] = cond_clean(_df["cond_W"])


# ---------------------------------------------------------------- Table 2: main results (m = 60 for approximate methods)
def agg(df, cols=("acc",)):
    g = df.groupby(["task", "method"])
    out = g["acc"].agg(["mean", "std", "count"]).reset_index()
    out["acc_pm"] = [fmt_pm(100 * m, 100 * s) for m, s in zip(out["mean"], out["std"])]
    n_test = df.groupby("task")["n_test"].first()
    out["wilson"] = [f"[{100*wilson_ci(int(round(m*n_test[t])), int(n_test[t]))[0]:.1f}, {100*wilson_ci(int(round(m*n_test[t])), int(n_test[t]))[1]:.1f}]" for t, m in zip(out["task"], out["mean"])]
    return out


M_MAIN, STRAT_MAIN = 60, "kmeans_centroids"
sel = main[(main["m"].isna()) | ((main["m"] == M_MAIN) & (main["strategy"].isin([STRAT_MAIN, "rff"])))]
methods_order = ["svm_linear", "svm_rbf", "logreg", "q_exact[product_cosine]", "q_exact[covariant_torus]",
                 "q_nystrom[product_cosine]", "q_nystrom[covariant_torus]", "rbf_nystrom", "rff_rbf"]
t2 = agg(sel)
ef = sel.groupby(["task", "method"])["E_F"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "EF_mean", "std": "EF_std"})
circ = sel.groupby(["task", "method"])["circ_total"].mean().reset_index()
rt = sel.groupby(["task", "method"])["runtime"].mean().reset_index()
t2 = t2.merge(ef, on=["task", "method"], how="left").merge(circ, on=["task", "method"], how="left").merge(rt, on=["task", "method"], how="left")
t2["method"] = pd.Categorical(t2["method"], methods_order, ordered=True); t2 = t2.sort_values(["task", "method"])
t2["E_F (%)"] = [("—" if pd.isna(m) else fmt_pm(100 * m, 100 * s, 1)) for m, s in zip(t2["EF_mean"], t2["EF_std"])]
t2["circuit evals"] = [("—" if pd.isna(c) else f"{int(c):,}") for c in t2["circ_total"]]
t2["Task"] = t2["task"].map(TASK_LABELS); t2["Method"] = t2["method"].map(LABELS)
wide = t2.pivot_table(index="Method", columns="Task", values="acc_pm", aggfunc="first", observed=True)
wide = wide.reindex([LABELS[m] for m in methods_order])
pooled = t2.groupby("Method", observed=True)["mean"].mean() * 100  # mean of the 7 task means (each over 10 seeds)
wide["Pooled mean (7 tasks)"] = [f"{pooled[mth]:.2f}" for mth in wide.index]
save_table(wide.reset_index(), "table2_main_accuracy", f"Test accuracy (%) mean ± SD over 10 seeds; approximate methods at m = {M_MAIN} landmarks ({STRAT_LABELS[STRAT_MAIN]} / D = {M_MAIN} random features). Full official test splits (n_test given in Table 1).", [SRC_MAIN], SCRIPT)
save_table(t2[["Task", "Method", "acc_pm", "wilson", "E_F (%)", "circuit evals", "runtime", "count"]].rename(columns={"acc_pm": "Acc (%) mean±SD", "wilson": "Wilson 95% CI of mean acc", "runtime": "runtime (s)", "count": "seeds"}),
           "table2_main_long", "Main results, long form.", [SRC_MAIN], SCRIPT)

# ---------------------------------------------------------------- Table 1: protocol (per task n_train, n_test, class counts, PCA EVR)
prot = main[main["method"] == "svm_linear"].groupby("task").agg(n_train=("n_train", "first"), n_test=("n_test", "first"), pca_evr=("pca_evr", "mean"), seeds=("seed", "nunique")).reset_index()
prot["Task"] = prot["task"].map(TASK_LABELS); prot["PCA-8 explained variance"] = (100 * prot["pca_evr"]).round(1).astype(str) + "%"
save_table(prot[["Task", "n_train", "n_test", "PCA-8 explained variance", "seeds"]], "table1_tasks", "Tasks, sample sizes and PCA-8 explained variance (mean over seeds).", [SRC_MAIN], SCRIPT)

# ---------------------------------------------------------------- statistical comparisons (paired over seeds, per task, Holm within family)
fam = []
q = main[(main["m"] == M_MAIN) & (main["strategy"] == STRAT_MAIN)]
for cand, ref in [("q_nystrom[covariant_torus]", "rbf_nystrom"), ("q_nystrom[covariant_torus]", "q_nystrom[product_cosine]")]:
    fam.append(compare_table(q, ["task"], "method", "acc", cand, ref))
ex = main[main["m"].isna()]
for cand, ref in [("q_exact[covariant_torus]", "svm_rbf"), ("q_exact[covariant_torus]", "svm_linear"), ("q_exact[covariant_torus]", "q_exact[product_cosine]")]:
    fam.append(compare_table(ex, ["task"], "method", "acc", cand, ref))
# exact vs Nyström (same map): cost of compression
mix = pd.concat([ex[ex["method"] == "q_exact[covariant_torus]"], q[q["method"] == "q_nystrom[covariant_torus]"]])
fam.append(compare_table(mix, ["task"], "method", "acc", "q_nystrom[covariant_torus]", "q_exact[covariant_torus]"))
stats = pd.concat(fam, ignore_index=True)
stats["p_t_holm_all"] = holm_correction(stats["p_t"].values)
stats["Task"] = stats["task"].map(TASK_LABELS)
stats["Δacc (pts)"] = (100 * stats["mean_diff"]).round(2); stats["95% CI"] = [f"[{100*a:.2f}, {100*b:.2f}]" for a, b in zip(stats["ci_lo"], stats["ci_hi"])]
stats_out = stats[["comparison", "Task", "n", "Δacc (pts)", "95% CI", "cohen_dz", "p_t", "p_wilcoxon", "p_t_holm_all"]].rename(columns={"p_t": "p (paired t)", "p_wilcoxon": "p (Wilcoxon)", "p_t_holm_all": "p Holm", "cohen_dz": "Cohen d_z"})
save_table(stats_out, "table_stats_main", f"Paired comparisons across seeds (n = 10) at m = {M_MAIN}; Holm correction over all {len(stats)} tests.", [SRC_MAIN], SCRIPT, floatfmt=".3f")

# landmark strategies: paired vs uniform at each m (E_F and acc), pooled over tasks
ls_rows = []
qn = main[main["method"] == "q_nystrom[covariant_torus]"]
for m, g in qn.groupby("m"):
    for s in [x for x in g["strategy"].unique() if x != "uniform"]:
        for metric in ("E_F", "acc"):
            piv = g.pivot_table(index=["task", "seed"], columns="strategy", values=metric)
            if s in piv and "uniform" in piv:
                pv = piv[[s, "uniform"]].dropna(); r = paired_comparison(pv[s].values, pv["uniform"].values, f"{s} - uniform")
                ls_rows.append(dict(m=m, strategy=s, metric=metric, n=r["n"], mean_diff=r["mean_diff"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_t=r["p_t"], p_wilcoxon=r["p_wilcoxon"], cohen_dz=r["cohen_dz"]))
ls = pd.DataFrame(ls_rows); ls["p_t_holm"] = holm_correction(ls["p_t"].values)
save_table(ls, "table_landmark_strategies", "Landmark strategies vs uniform sampling (paired over task × seed): difference in E_F and accuracy.", [SRC_MAIN], SCRIPT, floatfmt=".4f")

# ---------------------------------------------------------------- Nyström vs m summary (E_F, acc) per method/strategy pooled
ny = main[main["m"].notna()]
vs_m = ny.groupby(["method", "strategy", "m"]).agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"), EF_mean=("E_F", "mean"), EF_std=("E_F", "std"),
                                                    Espec_mean=("E_spec", "mean"), condW_med=("cond_W", "median"), frac_singular=("cond_W", lambda x: float(np.mean(~np.isfinite(x)))),
                                                    kta_mean=("kta_train", "mean"), circ=("circ_total", "mean"), n=("acc", "count")).reset_index()
vs_m["condW_med"] = [fmt_cond(x) for x in vs_m["condW_med"]]
save_table(vs_m, "table_vs_m_pooled", "Nyström metrics vs landmark budget m, pooled over tasks and seeds.", [SRC_MAIN], SCRIPT, floatfmt=".4f")

# ---------------------------------------------------------------- geometric difference table
gcols = [c for c in main.columns if c.startswith("g_vs_") and not c.endswith("lam0.0")]  # lambda = 0 is numerically unstable and omitted
gd = ex[ex["method"].str.startswith("q_exact")].groupby(["task", "method"])[gcols + ["eff_rank", "rank_1e_8", "offdiag_mean", "kta_train"]].agg(["mean", "std"])
gd.columns = ["_".join(c) for c in gd.columns]; gd = gd.reset_index(); gd["sqrtN"] = np.sqrt(main["n_train"].iloc[0])
save_table(gd, "table_geometric_difference", "Geometric difference g(K_C‖K_Q) (Huang et al. 2021) of the exact quantum kernels w.r.t. tuned RBF and linear kernels (N = 600 training points; sqrt(N) ≈ 24.5 is the scale at which advantage becomes possible), plus kernel spectrum statistics.", [SRC_MAIN], SCRIPT, floatfmt=".3f")

# ---------------------------------------------------------------- ablation / bandwidth / noise / scaling tables (if present)
if (RES / "ablation_per_seed.csv").exists():
    ab = load("ablation_per_seed.csv")
    ab["trained"] = ab["trained"].astype(str)
    t3 = ab.groupby(["variant", "map", "layers", "entangle", "trained", "n_params"]).agg(acc=("acc", "mean"), acc_sd=("acc", "std"), kta=("kta_train", "mean"), effrank=("eff_rank", "mean"),
                                                                                       acc_m60=("acc_nystrom_m60", "mean"), EF_m60=("E_F", "mean"), dist_primary=("dist_to_primary", "mean"), n=("acc", "count")).reset_index()
    if "rank_lambda" in ab:
        lam = ab[ab["variant"].str.contains("regKTA")].groupby("task")["rank_lambda"].value_counts().rename("count").reset_index()
        save_table(lam, "table3_reg_lambda_selected", "Regularised-KTA: λ selected on validation, per task (counts over seeds).", [RES / "ablation_per_seed.csv"], SCRIPT)
        sel = ab[ab["variant"].str.contains("valsel")].groupby("task")["selected_from"].value_counts().rename("count").reset_index()
        save_table(sel, "table3_valsel_choice", "Validation-selected variant: which candidate was chosen, per task (counts over seeds).", [RES / "ablation_per_seed.csv"], SCRIPT)
    t3["Acc (%)"] = [fmt_pm(100 * m, 100 * s) for m, s in zip(t3["acc"], t3["acc_sd"])]
    save_table(t3.sort_values(["map", "layers", "entangle", "trained"]), "table3_ablation", "Ablation of feature-map components and training objectives (a = π, 3 tasks × 10 seeds pooled): exact accuracy, KTA, effective rank, Nyström (m = 60) accuracy and E_F, kernel distance to the primary trained kernel.", [RES / "ablation_per_seed.csv"], SCRIPT, floatfmt=".3f")
    per_task = ab.groupby(["task", "variant"]).agg(acc=("acc", "mean"), acc_sd=("acc", "std"), kta=("kta_train", "mean")).reset_index()
    save_table(per_task, "table3_ablation_per_task", "Ablation per task.", [RES / "ablation_per_seed.csv"], SCRIPT, floatfmt=".4f")
    # paired tests: entanglement and training effects for covariant L=2
    abl_rows = []
    for task, g in ab.groupby("task"):
        piv = g.pivot_table(index="seed", columns="variant", values="acc")
        pairs = [("covariant_torus|L=2|ent=1|trained=1", "covariant_torus|L=2|ent=0|trained=1", "entanglement (trained)"),
                 ("covariant_torus|L=2|ent=1|trained=1", "covariant_torus|L=2|ent=1|trained=0", "training (entangled)"),
                 ("covariant_torus|L=2|ent=1|trained=1", "covariant_torus|L=1|ent=0|trained=0|theta=pi/2", "vs product-cosine (old construction)"),
                 ("covariant_torus|L=2|ent=1|trained=1", "reuploading|L=2|ent=1|trained=1", "covariant vs re-uploading"),
                 ("covariant_torus|L=2|ent=1|trained=1", "zz_reuploading|L=2|ent=1|trained=1", "covariant vs ZZ re-uploading"),
                 # alternative objective (docs/alternative_training_protocol.md)
                 ("covariant_torus|L=2|ent=1|trained=regKTA", "covariant_torus|L=2|ent=1|trained=0", "regularised KTA vs untrained"),
                 ("covariant_torus|L=2|ent=1|trained=regKTA", "covariant_torus|L=2|ent=1|trained=1", "regularised KTA vs plain KTA"),
                 ("covariant_torus|L=2|ent=1|trained=valsel", "covariant_torus|L=2|ent=1|trained=0", "validation-selected vs untrained"),
                 ("covariant_torus|L=2|ent=1|trained=valsel", "covariant_torus|L=2|ent=1|trained=1", "validation-selected vs plain KTA")]
        for a_, b_, lab in pairs:
            if a_ in piv and b_ in piv:
                r = paired_comparison(piv[a_].values, piv[b_].values, lab); r["task"] = task; abl_rows.append(r)
    abs_ = pd.DataFrame(abl_rows); abs_["p_t_holm"] = holm_correction(abs_["p_t"].values)
    save_table(abs_[["task", "comparison", "n", "mean_diff", "ci_lo", "ci_hi", "cohen_dz", "p_t", "p_wilcoxon", "p_t_holm"]], "table3_ablation_stats", "Paired tests for ablation contrasts (accuracy).", [RES / "ablation_per_seed.csv"], SCRIPT, floatfmt=".4f")

if (RES / "bandwidth_per_seed.csv").exists():
    bw = load("bandwidth_per_seed.csv")
    bw["cond_W"] = cond_clean(bw["cond_W"])
    tb = bw.groupby(["task", "map", "a_over_pi"]).agg(acc=("acc", "mean"), acc_sd=("acc", "std"), kta=("kta_train", "mean"), effrank=("eff_rank", "mean"), offdiag=("offdiag_mean", "mean"),
                                                        EF_m60=("E_F", "mean"), acc_m60=("acc_nystrom_m60", "mean"), condW=("cond_W", "median")).reset_index()
    tb["condW"] = [fmt_cond(x) for x in tb["condW"]]
    save_table(tb, "table_bandwidth", "Feature-scaling (bandwidth) study.", [RES / "bandwidth_per_seed.csv"], SCRIPT, floatfmt=".3f")

if (RES / "noise_per_seed.csv").exists():
    nz = load("noise_per_seed.csv"); nz["shots"] = nz["shots"].fillna(0).astype(int)
    t5 = nz[(nz["depolarizing"] == 0) & ((nz["eps"].isna()) | (nz["eps"] == 1e-6)) & ((nz["normalize"].isna()) | (nz["normalize"] == False))]
    nz["cond_W"] = cond_clean(nz["cond_W"])
    t5g = t5.groupby(["task", "method", "m", "shots"], dropna=False).agg(acc=("acc", "mean"), acc_sd=("acc", "std"), EF=("E_F", "mean"), noise_only=("noise_only_err", "mean"), amp=("amplification", "median"),
                                                                        frac_indefinite_W=("min_eig_W", lambda x: float(np.mean(x <= 0)) if x.notna().any() else np.nan), min_eig_W_med=("min_eig_W", "median"),
                                                                        Wpinv_med=("Wpinv_norm", "median"), min_eig_Ktr=("min_eig_Ktr", "median")).reset_index()
    save_table(t5g, "table5_noise_shots", "Idealised finite-shot study (no depolarising noise, eps = 1e-6, pseudo-inverse, unnormalised; 2 tasks × 10 seeds): mean accuracy ± SD, mean total Nyström error E_F, mean noise-only error, MEDIAN empirical amplification ratio, fraction of seeds with indefinite W_eps, median smallest eigenvalue of W_eps, median ||W_eps^+||_2, median smallest eigenvalue of the estimated Gram/Nyström matrix.", [RES / "noise_per_seed.csv"], SCRIPT, floatfmt=".4f")
    t5e = nz[(nz["method"] == "q_nystrom") & (nz["depolarizing"] == 0)].groupby(["m", "shots", "eps", "normalize"]).agg(acc=("acc", "mean"), EF=("E_F", "mean"), noise_only=("noise_only_err", "mean"), Wpinv_med=("Wpinv_norm", "median")).reset_index()
    save_table(t5e, "table5_noise_eps_normalize", "Effect of jitter eps and normalisation under shot noise (pooled over tasks/seeds).", [RES / "noise_per_seed.csv"], SCRIPT, floatfmt=".4f")
    t5d = nz[(nz["eps"].isna()) | ((nz["eps"] == 1e-6) & (nz["normalize"] == False))].groupby(["method", "m", "shots", "depolarizing"], dropna=False).agg(acc=("acc", "mean"), EF=("E_F", "mean")).reset_index()
    save_table(t5d, "table5_noise_depolarizing", "Depolarising noise × shots (eps = 1e-6, unnormalised).", [RES / "noise_per_seed.csv"], SCRIPT, floatfmt=".4f")

if (RES / "scaling_per_seed.csv").exists():
    scl = load("scaling_per_seed.csv")
    ts = scl.groupby(["method", "N", "m"], dropna=False).agg(acc=("acc", "mean"), acc_sd=("acc", "std"), t_total=("t_total", "mean"), t_kernel=("t_kernel", "mean"), t_svm=("t_svm", "mean"), mem_MB=("gram_bytes", lambda x: x.mean() / 1e6), peak_MB=("peak_mem_MB", "mean"), circ_train=("circ_train", "mean"), circ_test=("circ_test", "mean"), EF=("E_F", "mean")).reset_index()
    save_table(ts, "table_scaling", "Scaling study (fashion_0v1, 3 seeds): wall-clock, memory, circuit evaluations, E_F, accuracy vs N.", [RES / "scaling_per_seed.csv"], SCRIPT, floatfmt=".3f")

if (RES / "leverage_sweep_per_seed.csv").exists():
    lv = load("leverage_sweep_per_seed.csv"); lv["ridge"] = lv["ridge"].fillna(-1)
    tl = lv.groupby(["strategy", "ridge", "m"]).agg(EF_mean=("E_F", "mean"), EF_med=("E_F", "median"), acc=("acc", "mean"), acc_sd=("acc", "std"), entropy=("sampling_entropy", "mean"), n=("acc", "count")).reset_index()
    tl["ridge"] = tl["ridge"].replace(-1, np.nan)
    save_table(tl, "table_leverage_sweep", "Ridge-leverage-score sampling vs ridge parameter λ (absolute; N = 600), with uniform and k-means-centroid references (7 tasks × 10 seeds): mean/median E_F, accuracy, entropy of the sampling distribution (uniform = ln 600 = 6.40).", [RES / "leverage_sweep_per_seed.csv"], SCRIPT, floatfmt=".4f")
    # paired vs uniform at each (ridge, m)
    lrows = []
    for (lam, m), g in lv[lv["strategy"] == "leverage"].groupby(["ridge", "m"]):
        u = lv[(lv["strategy"] == "uniform") & (lv["m"] == m)].set_index(["task", "seed"])["E_F"]; l = g.set_index(["task", "seed"])["E_F"]
        j = pd.concat([l, u], axis=1, keys=["lev", "uni"]).dropna(); r = paired_comparison(j["lev"].values, j["uni"].values, f"leverage(λ={lam:g}) - uniform")
        lrows.append(dict(ridge=lam, m=m, n=r["n"], EF_diff=r["mean_diff"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_t=r["p_t"], p_wilcoxon=r["p_wilcoxon"]))
    lst = pd.DataFrame(lrows); lst["p_t_holm"] = holm_correction(lst["p_t"].values)
    save_table(lst, "table_leverage_sweep_stats", "Leverage sampling vs uniform (paired over task × seed): E_F difference by ridge λ and m; Holm-corrected.", [RES / "leverage_sweep_per_seed.csv"], SCRIPT, floatfmt=".4f")

if (RES / "validation_feature_maps.csv").exists():
    va = load("validation_feature_maps.csv")
    tv = va.groupby("map").agg(val_acc=("val_acc", "mean"), kta_init=("kta_init", "mean"), kta_trained=("kta_trained", "mean"), dK_theta=("max_dK_theta", "mean"), dK_ent=("max_dK_entangle", "mean"),
                              eff_rank=("eff_rank", "mean"), min_eig=("min_eig", "min"), dist_rbf=("dist_rbf", "mean"), dist_pc=("dist_product_cosine", "mean"), g_rbf=("g_vs_rbf", "mean"), runtime=("runtime", "mean"), n=("val_acc", "count")).reset_index()
    save_table(tv, "table_validation_maps", "Phase-D validation of candidate feature maps (7 tasks × 3 seeds): validation accuracy, KTA before/after training, max kernel change under θ perturbation and entanglement removal, effective rank, minimum eigenvalue, distance to RBF and product-cosine kernels, geometric difference vs RBF.", [RES / "validation_feature_maps.csv"], SCRIPT, floatfmt=".3f")
print("tables written to", TAB)
