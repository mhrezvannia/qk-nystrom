"""Generate every manuscript figure (PDF + PNG + provenance JSON) from results/*.csv."""
from common import *
SCRIPT = "analysis/make_figures.py"
main = load("main_per_seed.csv"); SRC = RES / "main_per_seed.csv"
tasks = [t for t in TASK_LABELS if t in set(main["task"])]


def mean_sd(df, by, col):
    g = df.groupby(by)[col].agg(["mean", "std", "count"]).reset_index(); g["sem"] = g["std"] / np.sqrt(g["count"]); return g


# ---------------------------------------------------------------- Fig: E_F and accuracy vs m (primary map, strategies), one panel per task
def fig_vs_m(metric, ylabel, name, method="q_nystrom[covariant_torus]", logy=False):
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True); axes = axes.ravel()
    df = main[main["method"] == method]
    for ax, t in zip(axes, tasks):
        d = df[df["task"] == t]
        for s, c in zip(STRAT_LABELS, ["#7f7f7f", "#1b9e77", "#1b6ca8", "#7570b3", "#d62728"]):
            dd = d[d["strategy"] == s]
            if logy:  # median and inter-quartile range (SD is not meaningful on a log axis)
                g = dd.groupby("m")[metric].quantile([0.25, 0.5, 0.75]).unstack().reset_index()
                if len(g): ax.errorbar(g["m"], g[0.5], yerr=[g[0.5] - g[0.25], g[0.75] - g[0.5]], marker="o", ms=3, lw=1, capsize=2, color=c, label=STRAT_LABELS[s], alpha=0.9)
            else:
                g = mean_sd(dd, "m", metric)
                if len(g): ax.errorbar(g["m"], g["mean"], yerr=g["std"], marker="o", ms=3, lw=1, capsize=2, color=c, label=STRAT_LABELS[s], alpha=0.9)
        ex = main[(main["task"] == t) & (main["method"] == "q_exact[covariant_torus]")]
        if metric == "acc" and len(ex): ax.axhline(ex["acc"].mean(), color="k", ls="--", lw=1, label="exact quantum kernel")
        if metric == "acc":
            for mth, c in [("svm_rbf", COLORS["svm_rbf"]), ("svm_linear", COLORS["svm_linear"])]:
                b = main[(main["task"] == t) & (main["method"] == mth)]
                if len(b): ax.axhline(b["acc"].mean(), color=c, ls=":", lw=1, label=LABELS[mth])
        ax.set_xscale("log"); ax.set_title(TASK_SHORT[t]); ax.set_xlim(8, 700)
        if logy: ax.set_yscale("log")
    for ax in axes[3:7]: ax.set_xlabel("landmarks m")
    axes[0].set_ylabel(ylabel); axes[4].set_ylabel(ylabel)
    for ax in axes[len(tasks):]: ax.axis("off")
    h, l = axes[0].get_legend_handles_labels(); axes[-1].legend(h, l, loc="center", frameon=False, fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, name, [SRC], SCRIPT, f"{ylabel} versus landmark budget m for the trained covariant quantum kernel; {'median and IQR' if logy else 'mean ± SD'} over 10 seeds; one panel per task.")


fig_vs_m("E_F", "relative Frobenius error $E_F$", "fig05_EF_vs_m", logy=True)
fig_vs_m("acc", "test accuracy", "fig06_acc_vs_m")

# ---------------------------------------------------------------- Fig: quantum vs classical Nyström vs RFF at equal budget (k-means centroids), pooled panel per task
fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True); axes = axes.ravel()
for ax, t in zip(axes, tasks):
    for mth in ["q_nystrom[covariant_torus]", "q_nystrom[product_cosine]", "rbf_nystrom", "rff_rbf"]:
        d = main[(main["task"] == t) & (main["method"] == mth) & (main["strategy"].isin(["kmeans_centroids", "rff"]))]
        g = mean_sd(d, "m", "acc")
        if len(g): ax.errorbar(g["m"], g["mean"], yerr=g["std"], marker="o", ms=3, lw=1, capsize=2, color=COLORS[mth], label=SHORT[mth])
    for mth in ["svm_rbf", "q_exact[covariant_torus]"]:
        b = main[(main["task"] == t) & (main["method"] == mth)]
        if len(b): ax.axhline(b["acc"].mean(), color=COLORS[mth], ls="--", lw=1, label=SHORT[mth])
    ax.set_xscale("log"); ax.set_title(TASK_SHORT[t]); ax.set_xlim(8, 700)
for ax in axes[3:7]: ax.set_xlabel("budget m")
axes[0].set_ylabel("test accuracy"); axes[4].set_ylabel("test accuracy")
for ax in axes[len(tasks):]: ax.axis("off")
h, l = axes[0].get_legend_handles_labels(); axes[-1].legend(h, l, loc="center", frameon=False, fontsize=7.5); fig.tight_layout()
save_fig(fig, "fig09_quantum_vs_classical_budget", [SRC], SCRIPT, "Test accuracy versus budget m for quantum Nyström (trained covariant and product-cosine kernels), classical RBF Nyström with identical k-means landmarks, and RBF random Fourier features with D = m; dashed: exact kernels.")

# ---------------------------------------------------------------- Fig: kernel spectra / conditioning (cond W vs m; eff rank of exact kernels)
fig, axes = plt.subplots(1, 3, figsize=(FIG_W_MAX, 3.0))
d = main[(main["method"].str.startswith("q_nystrom")) & (main["strategy"] == "kmeans_centroids")]
for mth in ["q_nystrom[covariant_torus]", "q_nystrom[product_cosine]"]:
    g = d[d["method"] == mth].groupby("m")["cond_W"].median().reset_index(); axes[0].plot(g["m"], g["cond_W"], marker="o", ms=3, color=COLORS[mth], label=SHORT[mth])
g = main[(main["method"] == "rbf_nystrom") & (main["strategy"] == "kmeans_centroids")].groupby("m")["cond_W"].median().reset_index()
axes[0].plot(g["m"], g["cond_W"], marker="s", ms=3, color=COLORS["rbf_nystrom"], label=SHORT["rbf_nystrom"])
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("landmarks m"); axes[0].set_ylabel("median cond($W_\\epsilon$)"); axes[0].set_title("(a) conditioning of $W_\\epsilon$")
ex = main[main["method"].str.startswith("q_exact")]
for mth, c in [("q_exact[covariant_torus]", COLORS["q_exact[covariant_torus]"]), ("q_exact[product_cosine]", COLORS["q_exact[product_cosine]"])]:
    g = ex[ex["method"] == mth].groupby("task")[["energy_top20", "energy_top60", "energy_top120"]].mean()
    x = np.arange(len(g)) + (0.2 if "product" in mth else -0.2)
    axes[1].bar(x, g["energy_top60"], width=0.38, color=c, label=SHORT[mth])
axes[1].set_xticks(np.arange(len(g))); axes[1].set_xticklabels([TASK_SHORT[t] for t in g.index], rotation=45, ha="right")
axes[1].set_ylabel("trace energy in top-60 eigenvalues"); axes[1].set_title("(b) low-rank content, exact kernels")
for mth, c in [("q_exact[covariant_torus]", COLORS["q_exact[covariant_torus]"]), ("q_exact[product_cosine]", COLORS["q_exact[product_cosine]"])]:
    g = ex[ex["method"] == mth].groupby("task")["eff_rank"].mean()
    x = np.arange(len(g)) + (0.2 if "product" in mth else -0.2)
    axes[2].bar(x, g.values, width=0.38, color=c, label=LABELS[mth])
axes[2].set_xticks(np.arange(len(g))); axes[2].set_xticklabels([TASK_SHORT[t] for t in g.index], rotation=45, ha="right")
axes[2].set_ylabel("effective rank (entropy)"); axes[2].set_yscale("log"); axes[2].set_title("(c) effective rank, exact kernels")
fig.tight_layout(rect=[0, 0.14, 1, 1])
_h0, _l0 = axes[0].get_legend_handles_labels(); _h1, _l1 = axes[1].get_legend_handles_labels()
fig.legend(_h0 + _h1, _l0 + _l1, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
save_fig(fig, "fig03_spectrum_conditioning", [SRC], SCRIPT, "Conditioning of the landmark kernel W versus m and spectral statistics of the exact kernels.")

# ---------------------------------------------------------------- Fig: geometric difference
gcols = ["g_vs_rbf_lam0.0", "g_vs_rbf_lam0.01", "g_vs_linear_lam0.0", "g_vs_linear_lam0.01"]
fig, ax = plt.subplots(figsize=(FIG_W_MAX, 3.0))
for i, (mth, c) in enumerate([("q_exact[covariant_torus]", COLORS["q_exact[covariant_torus]"]), ("q_exact[product_cosine]", COLORS["q_exact[product_cosine]"])]):
    g = ex[ex["method"] == mth].groupby("task")[["g_vs_rbf_lam0.01", "g_vs_linear_lam0.01"]].agg(["mean", "std"])
    x = np.arange(len(g)) + (-0.3 + 0.2 * i)
    ax.bar(x, g[("g_vs_rbf_lam0.01", "mean")], 0.18, yerr=g[("g_vs_rbf_lam0.01", "std")], color=c, label=f"{SHORT[mth]} vs RBF")
    ax.bar(x + 0.4, g[("g_vs_linear_lam0.01", "mean")], 0.18, yerr=g[("g_vs_linear_lam0.01", "std")], color=c, alpha=0.45, hatch="//", label=f"{SHORT[mth]} vs linear")
ax.axhline(np.sqrt(main["n_train"].iloc[0]), color="r", ls="--", lw=1, label="$\\sqrt{N}$ (advantage possible above)")
ax.axhline(1, color="k", ls=":", lw=1)
ax.set_xticks(np.arange(len(g)) + 0.05); ax.set_xticklabels([TASK_SHORT[t] for t in g.index], rotation=30, ha="right")
ax.set_ylabel("geometric difference $g(K_C\\|K_Q)$"); ax.set_yscale("log"); ax.set_ylim(top=ax.get_ylim()[1] * 30); ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper left")
fig.tight_layout(); save_fig(fig, "fig10_geometric_difference", [SRC], SCRIPT, "Geometric difference (Huang et al. 2021, λ = 0.01) between the exact quantum kernels and tuned classical RBF / linear kernels on the 600 training points; values far below sqrt(N) mean the classical kernel can express the quantum kernel's functions.")

# ---------------------------------------------------------------- optional figures from other experiments
if (RES / "ablation_per_seed.csv").exists():
    ab = load("ablation_per_seed.csv")
    ref = "covariant_torus|L=1|ent=0|trained=0|theta=pi/2"  # product-cosine kernel (previous construction) as within-(task,seed) reference
    piv = ab.pivot_table(index=["task", "seed"], columns="variant", values="acc")
    delta = (piv.sub(piv[ref], axis=0) * 100).drop(columns=[ref])
    order = sorted(delta.columns, key=lambda v: (v.split("|")[0], v))
    g = ab.groupby("variant").agg(kta=("kta_train", "mean"), ksd=("kta_train", "std"), ef=("E_F", "mean"), efsd=("E_F", "std")).reindex(order)
    dm, dse = delta[order].mean(), delta[order].std(ddof=1) / np.sqrt(delta.shape[0])
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_MAX, 4.2), sharey=True, gridspec_kw={"width_ratios": [1.15, 1, 1]}); y = np.arange(len(order))
    cols = ["#1b6ca8" if v.startswith("cov") else "#d95f02" if v.startswith("reup") else "#7570b3" for v in order]
    axes[0].barh(y, dm, xerr=1.96 * dse, color=cols, capsize=2); axes[0].axvline(0, color="k", lw=0.8); axes[0].set_xlabel("Δ accuracy vs product-cosine (pts, 95% CI)")
    axes[1].barh(y, g["kta"], xerr=g["ksd"], color=cols, capsize=2); axes[1].set_xlabel("centred KTA (train)")
    axes[2].barh(y, g["ef"], xerr=g["efsd"], color=cols, capsize=2); axes[2].set_xlabel("$E_F$ at m = 60"); axes[2].set_xscale("log")
    axes[0].set_yticks(y); axes[0].set_yticklabels([v.replace("covariant_torus", "cov").replace("reuploading", "reup").replace("zz_reuploading", "zz-reup") for v in order], fontsize=7.5)
    fig.tight_layout(); save_fig(fig, "fig04_ablation", [RES / "ablation_per_seed.csv"], SCRIPT, "Feature-map ablation (a = π; 3 tasks × 5 seeds): paired accuracy difference to the product-cosine kernel, KTA and Nyström error for each variant.")

if (RES / "noise_per_seed.csv").exists():
    nz = load("noise_per_seed.csv"); nz["shots"] = nz["shots"].fillna(np.inf)
    base = nz[(nz["depolarizing"] == 0) & ((nz["eps"].isna()) | (nz["eps"] == 1e-6)) & ((nz["normalize"].isna()) | (nz["normalize"] == False))]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_MAX, 2.6))
    xs = sorted(base["shots"].unique()); xl = [("∞" if np.isinf(s) else f"{int(s):,}") for s in xs]
    for m, c in zip([20, 60, 120], ["#9ecae1", "#3182bd", "#08519c"]):
        g = base[(base["method"] == "q_nystrom") & (base["m"] == m)].groupby("shots").agg(acc=("acc", "mean"), sd=("acc", "std"), ef=("E_F", "mean"), amp=("amplification", "mean")).reindex(xs)
        axes[0].errorbar(range(len(xs)), g["acc"], yerr=g["sd"], marker="o", ms=3, color=c, label=f"Nyström m={m}", capsize=2)
        axes[1].plot(range(len(xs)), g["ef"], marker="o", ms=3, color=c, label=f"Nyström m={m}")
    g = base[base["method"] == "q_exact"].groupby("shots").agg(acc=("acc", "mean"), sd=("acc", "std"), ef=("E_F", "mean")).reindex(xs)
    axes[0].errorbar(range(len(xs)), g["acc"], yerr=g["sd"], marker="s", ms=3, color="k", label="exact kernel", capsize=2); axes[1].plot(range(len(xs)), g["ef"], marker="s", ms=3, color="k", label="exact kernel (estimation error)")
    for ax in axes[:2]: ax.set_xticks(range(len(xs))); ax.set_xticklabels(xl); ax.set_xlabel("shots per fidelity")
    axes[0].set_ylabel("test accuracy"); axes[1].set_ylabel("$\\|K-\\hat K\\|_F/\\|K\\|_F$"); axes[1].set_yscale("log"); axes[0].legend(frameon=False, fontsize=7.5)
    e = nz[(nz["method"] == "q_nystrom") & (nz["depolarizing"] == 0) & (nz["shots"] == 1000)].groupby(["eps", "normalize", "m"]).agg(acc=("acc", "mean")).reset_index()
    for norm, ls in [(False, "-"), (True, "--")]:
        for m, c in zip([20, 60, 120], ["#9ecae1", "#3182bd", "#08519c"]):
            gg = e[(e["normalize"] == norm) & (e["m"] == m)].sort_values("eps"); axes[2].plot(gg["eps"], gg["acc"], ls=ls, marker="o", ms=3, color=c, label=f"m={m}{' (normalised)' if norm else ''}")
    axes[2].set_xscale("log"); axes[2].set_xlabel("jitter $\\epsilon$ (S = 1,000 shots)"); axes[2].set_ylabel("test accuracy")
    axes[0].set_title("(a) accuracy vs shots"); axes[1].set_title("(b) kernel error vs shots"); axes[2].set_title("(c) jitter / normalisation")
    fig.tight_layout(rect=[0, 0.13, 1, 1]); _h, _l = axes[2].get_legend_handles_labels(); fig.legend(_h, _l, loc="lower center", ncol=6, frameon=False, fontsize=7.5, title="(c): pseudo-inverse of $W_\\epsilon$, S = 1,000 shots", title_fontsize=7.5)
    save_fig(fig, "fig07_noise_shots", [RES / "noise_per_seed.csv"], SCRIPT, "Finite-shot robustness of exact and Nyström-compressed quantum kernels (2 tasks × 10 seeds pooled).")

if (RES / "scaling_per_seed.csv").exists():
    sc = load("scaling_per_seed.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_MAX, 2.6))
    for mth, c, lab in [("q_exact", "k", "exact kernel"), ("q_nystrom_m_fixed", "#1b6ca8", "Nyström, m = 60"), ("q_nystrom_m_sqrtN", "#d95f02", "Nyström, m ∝ √N")]:
        g = sc[sc["method"] == mth].groupby("N").agg(t=("t_total", "mean"), circ=("circ_train", "mean"), acc=("acc", "mean"), sd=("acc", "std"), mem=("gram_bytes", "mean")).reset_index()
        axes[0].plot(g["N"], g["circ"], marker="o", ms=3, color=c, label=lab); axes[1].plot(g["N"], g["t"], marker="o", ms=3, color=c, label=lab); axes[2].errorbar(g["N"], g["acc"], yerr=g["sd"], marker="o", ms=3, color=c, label=lab, capsize=2)
    for ax in axes: ax.set_xscale("log"); ax.set_xlabel("N (training points)")
    axes[0].set_yscale("log"); axes[0].set_ylabel("training-phase circuit evaluations"); axes[1].set_yscale("log"); axes[1].set_ylabel("simulation wall-clock (s)"); axes[2].set_ylabel("test accuracy (n = 2,000)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left"); axes[0].set_title("(a) fidelity evaluations"); axes[1].set_title("(b) simulation time"); axes[2].set_title("(c) accuracy")
    fig.tight_layout(); save_fig(fig, "fig08_scaling", [RES / "scaling_per_seed.csv"], SCRIPT, "Scaling with training-set size N (fashion_0v1, 3 seeds): circuit evaluations (symmetric count), wall-clock in simulation, and test accuracy.")

if (RES / "bandwidth_per_seed.csv").exists():
    bw = load("bandwidth_per_seed.csv")
    fig, axes = plt.subplots(1, 4, figsize=(FIG_W_MAX, 2.9))
    for mp, c in [("covariant_torus", "#1b6ca8"), ("product_cosine", "#7f7f7f")]:
        for t, ls in zip(sorted(bw["task"].unique()), ["-", "--"]):
            g = bw[(bw["map"] == mp) & (bw["task"] == t)].groupby("a_over_pi").agg(acc=("acc", "mean"), kta=("kta_train", "mean"), off=("offdiag_mean", "mean"), er=("eff_rank", "mean"), ef=("E_F", "mean")).reset_index()
            lab = f"{mp.replace('_',' ')} — {TASK_LABELS[t].replace('F-MNIST ','')}"
            axes[0].plot(g["a_over_pi"], g["acc"], ls=ls, marker="o", ms=3, color=c, label=lab); axes[1].plot(g["a_over_pi"], g["kta"], ls=ls, marker="o", ms=3, color=c)
            axes[2].plot(g["a_over_pi"], g["er"], ls=ls, marker="o", ms=3, color=c); axes[3].plot(g["a_over_pi"], g["ef"], ls=ls, marker="o", ms=3, color=c)
    for ax, yl in zip(axes, ["test accuracy", "centred KTA", "effective rank", "$E_F$ at m = 60"]): ax.set_xlabel("scaling range a / π"); ax.set_ylabel(yl)
    axes[2].set_yscale("log")
    fig.tight_layout(rect=[0, 0.2, 1, 1]); _h, _l = axes[0].get_legend_handles_labels(); fig.legend(_h, _l, loc="lower center", ncol=2, frameon=False, fontsize=7.5)
    save_fig(fig, "fig_bandwidth", [RES / "bandwidth_per_seed.csv"], SCRIPT, "Feature-scaling (bandwidth) study.")
print("figures written to", FIG)

# ---------------------------------------------------------------- Supplementary: leverage ridge sweep
if (RES / "leverage_sweep_per_seed.csv").exists():
    lv = load("leverage_sweep_per_seed.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_MAX, 2.7), sharey=False)
    for ax, m in zip(axes, sorted(lv["m"].unique())):
        d = lv[lv["m"] == m]
        g = d[d["strategy"] == "leverage"].groupby("ridge")["E_F"].quantile([0.25, 0.5, 0.75]).unstack().reset_index()
        ax.errorbar(g["ridge"], g[0.5], yerr=[g[0.5] - g[0.25], g[0.75] - g[0.5]], marker="o", ms=3, color="#d62728", capsize=2, label="ridge-leverage sampling")
        for s, c, lab in [("uniform", "#7f7f7f", "uniform"), ("kmeans_centroids", "#1b6ca8", "k-means centroids")]:
            q = d[d["strategy"] == s]["E_F"].quantile([0.25, 0.5, 0.75])
            ax.axhline(q[0.5], color=c, ls="--", lw=1, label=lab); ax.axhspan(q[0.25], q[0.75], color=c, alpha=0.12)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("ridge λ (absolute, N = 600)"); ax.set_title(f"m = {int(m)}")
    axes[0].set_ylabel("$E_F$ (median, IQR)")
    fig.tight_layout(rect=[0, 0.12, 1, 1]); _h, _l = axes[0].get_legend_handles_labels(); fig.legend(_h, _l, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    save_fig(fig, "figS_leverage_sweep", [RES / "leverage_sweep_per_seed.csv"], SCRIPT, "Ridge-leverage-score landmark sampling: E_F versus ridge parameter λ for m = 10/20/60 (7 tasks × 10 seeds; median and IQR), with uniform and k-means-centroid references.")
print("supplementary figures written")
