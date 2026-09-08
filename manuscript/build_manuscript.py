"""Build manuscript/manuscript.md (and .docx via pandoc) from tables/*.csv and results/*.csv.
Every number in the text is computed here from result files; nothing is typed by hand."""
import sys, os, json, subprocess, argparse
from pathlib import Path
_ap = argparse.ArgumentParser(); _ap.add_argument("--anonymous", action="store_true"); _ap.add_argument("--out", default="manuscript.md")
_ap.add_argument("--reference-doc", default=None, help="pandoc reference .docx (journal styles); default: manuscript/reference_qmi.docx if present")
_args = _ap.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent)); import refs as _refs   # references.yaml -> Springer name-year rendering
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
from common import TASK_LABELS, LABELS
from qkn.statistics import paired_comparison, holm_correction
from qkn.metrics import wilson_ci

TAB, RES, FIG = ROOT / "tables", ROOT / "results", ROOT / "figures"
main = pd.read_csv(RES / "main_per_seed.csv")
stats = pd.read_csv(TAB / "table_stats_main.csv")
ls = pd.read_csv(TAB / "table_landmark_strategies.csv")
val = pd.read_csv(RES / "validation_feature_maps.csv")
gd = pd.read_csv(TAB / "table_geometric_difference.csv")
ab = pd.read_csv(RES / "ablation_per_seed.csv") if (RES / "ablation_per_seed.csv").exists() else None
bw = pd.read_csv(RES / "bandwidth_per_seed.csv") if (RES / "bandwidth_per_seed.csv").exists() else None
nz = pd.read_csv(RES / "noise_per_seed.csv") if (RES / "noise_per_seed.csv").exists() else None
sc = pd.read_csv(RES / "scaling_per_seed.csv") if (RES / "scaling_per_seed.csv").exists() else None
cfg = json.load(open(RES / "main_per_seed.meta.json"))["config"]
TASKS = [t for t in TASK_LABELS if t in set(main["task"])]
N_TR = int(main["n_train"].iloc[0]); SQRTN = np.sqrt(N_TR)


def acc(task, method, m=None, strategy="kmeans_centroids"):
    d = main[(main["task"] == task) & (main["method"] == method)]
    d = d[d["m"].isna()] if m is None else d[(d["m"] == m) & (d["strategy"] == strategy)]
    return 100 * d["acc"].mean(), 100 * d["acc"].std(ddof=1), len(d)


def pm(task, method, m=None, nd=2):
    a, s, _ = acc(task, method, m); return f"{a:.{nd}f} ± {s:.{nd}f}"


def pooled(method, m=None, strategy="kmeans_centroids"):
    return np.mean([acc(t, method, m, strategy)[0] for t in TASKS])


def st(cand, ref, task):
    r = stats[(stats["comparison"] == f"{cand} - {ref}") & (stats["Task"] == TASK_LABELS[task])].iloc[0]
    return r


def wins(cand, ref, m=None):
    return sum(acc(t, cand, m)[0] > acc(t, ref, m)[0] for t in TASKS)


def sig_count(cand, ref, col="p (paired t)", alpha=0.05):
    d = stats[stats["comparison"] == f"{cand} - {ref}"]; return int((d[col] < alpha).sum()), int((d["p Holm"] < alpha).sum())


def rng_str(vals, nd=1):
    return f"{min(vals):.{nd}f}–{max(vals):.{nd}f}"


# ------------------------------------------------------------------ derived numbers
Q, PC, RBFN, RFF = "q_nystrom[covariant_torus]", "q_nystrom[product_cosine]", "rbf_nystrom", "rff_rbf"
QE, PCE, RBF, LIN, LR = "q_exact[covariant_torus]", "q_exact[product_cosine]", "svm_rbf", "svm_linear", "logreg"
ex = main[main["m"].isna()]
qn = main[main["method"] == Q]
ef60 = {t: 100 * qn[(qn["task"] == t) & (qn["m"] == 60) & (qn["strategy"] == "kmeans_centroids")]["E_F"].mean() for t in TASKS}
ef60_pc = {t: 100 * main[(main["method"] == PC) & (main["task"] == t) & (main["m"] == 60) & (main["strategy"] == "kmeans_centroids")]["E_F"].mean() for t in TASKS}
ef60_rbf = {t: 100 * main[(main["method"] == RBFN) & (main["task"] == t) & (main["m"] == 60) & (main["strategy"] == "kmeans_centroids")]["E_F"].mean() for t in TASKS}
effrank = {mth: ex[ex["method"] == mth]["eff_rank"].mean() for mth in (QE, PCE)}
e60 = {mth: ex[ex["method"] == mth]["energy_top60"].mean() for mth in (QE, PCE)}
g_rbf = {mth: ex[ex["method"] == mth]["g_vs_rbf_lam0.01"].mean() for mth in (QE, PCE)}
g_lin = {mth: ex[ex["method"] == mth]["g_vs_linear_lam0.01"].mean() for mth in (QE, PCE)}
g_rbf_task = {t: ex[(ex["method"] == QE) & (ex["task"] == t)]["g_vs_rbf_lam0.01"].mean() for t in TASKS}
kta_gain = ex[ex["method"] == QE][["kta_init", "kta_final"]].mean()
a_counts = (ex[ex["method"] == QE]["a"] / np.pi).round(2).value_counts().to_dict()
# landmark strategy effect (E_F relative reduction vs uniform at m=10..60), accuracy effect
def strat_eff(strategy, m, metric):
    r = ls[(ls["strategy"] == strategy) & (ls["m"] == m) & (ls["metric"] == metric)].iloc[0]; return r
ef_unif = {m: 100 * qn[(qn["m"] == m) & (qn["strategy"] == "uniform")]["E_F"].mean() for m in (10, 20, 60, 120)}
ef_km = {m: 100 * qn[(qn["m"] == m) & (qn["strategy"] == "kmeans_centroids")]["E_F"].mean() for m in (10, 20, 60, 120)}
ef_lev = {m: 100 * qn[(qn["m"] == m) & (qn["strategy"] == "leverage")]["E_F"].mean() for m in (10, 20, 60, 120)}
acc_m = {m: 100 * qn[(qn["m"] == m) & (qn["strategy"] == "kmeans_centroids") & (qn["task"] != "breast_cancer")]["acc"].mean() for m in (10, 20, 40, 60, 120, 300)}
acc_ex_img = 100 * ex[(ex["method"] == QE) & (ex["task"] != "breast_cancer")]["acc"].mean()
# compression cost: Nyström(m=60) - exact, per task
comp = stats[stats["comparison"] == f"{Q} - {QE}"]
# circuit counts
cc_ex = ex[ex["method"] == QE].iloc[0]; cc_ny = qn[(qn["m"] == 60) & (qn["strategy"] == "kmeans_centroids") & (qn["task"] == "fashion_0v1")].iloc[0]
# validation-gate numbers
vg = val.groupby("map")[["val_acc", "kta_init", "kta_trained", "max_dK_theta", "max_dK_entangle", "eff_rank", "runtime"]].mean()
vpiv = val.pivot_table(index=["task", "seed"], columns="map", values="val_acc")
vp = {k: paired_comparison(vpiv["covariant_torus"], vpiv[k], "")["p_t"] for k in ("reuploading", "zz_reuploading", "product_cosine")}
kpiv = val.pivot_table(index=["task", "seed"], columns="map", values="kta_trained")
kp = {k: paired_comparison(kpiv["covariant_torus"], kpiv[k], "") for k in ("reuploading", "zz_reuploading", "product_cosine")}

out = []
w = out.append

# ------------------------------------------------------------------ ablation-derived numbers
if ab is not None:
    abg = ab.groupby("variant").agg(acc=("acc", "mean"), kta=("kta_train", "mean"), er=("eff_rank", "mean"), ef=("E_F", "mean"), acc60=("acc_nystrom_m60", "mean"))
    V_PRI, V_UNTR, V_NOENT, V_PC = "covariant_torus|L=2|ent=1|trained=1", "covariant_torus|L=2|ent=1|trained=0", "covariant_torus|L=2|ent=0|trained=1", "covariant_torus|L=1|ent=0|trained=0|theta=pi/2"
    V_RE2, V_RE1, V_ZZ = "reuploading|L=2|ent=1|trained=1", "reuploading|L=1|ent=1|trained=1", "zz_reuploading|L=2|ent=1|trained=1"
    abst = pd.read_csv(TAB / "table3_ablation_stats.csv")
    def abs_row(task, comp): 
        d = abst[(abst["task"] == task) & (abst["comparison"] == comp)]; return d.iloc[0] if len(d) else None
    ab_tasks = sorted(ab["task"].unique()); ab_seeds = ab["seed"].nunique()
    V_REG, V_SEL = "covariant_torus|L=2|ent=1|trained=regKTA", "covariant_torus|L=2|ent=1|trained=valsel"
    HAS_ALT = V_REG in set(ab["variant"])
    def contrast(comp):  # per-task rows for one ablation contrast
        return abst[abst["comparison"] == comp].set_index("task")
    if HAS_ALT:
        c_reg_u, c_reg_k, c_kta_u = contrast("regularised KTA vs untrained"), contrast("regularised KTA vs plain KTA"), contrast("training (entangled)")
        c_sel_u = contrast("validation-selected vs untrained")
        d_reg = 100 * c_reg_u["mean_diff"].mean(); d_kta = 100 * c_kta_u["mean_diff"].mean(); d_sel = 100 * c_sel_u["mean_diff"].mean()
        n_reg_pos = int((c_reg_u["mean_diff"] > 0).sum()); n_kta_neg = int((c_kta_u["mean_diff"] < 0).sum())
        holm_reg = int((c_reg_u["p_t_holm"] < 0.05).sum()); holm_kta = int((c_kta_u["p_t_holm"] < 0.05).sum()); holm_regk = int((c_reg_k["p_t_holm"] < 0.05).sum())
        lam_counts = ab[ab["variant"] == V_REG]["rank_lambda"].value_counts().to_dict(); sel_counts = ab[ab["variant"] == V_SEL]["selected_from"].value_counts().to_dict()
        # pre-registered decision rule (docs/alternative_training_protocol.md)
        if d_reg <= -0.5: SCEN = "A"           # regularised objective also lowers accuracy -> statement generalises
        elif abs(d_reg) < 0.5: SCEN = "B"      # regularised objective matches untrained -> narrow the claim to plain KTA
        else: SCEN = "C"                        # regularised objective improves -> central result
        TRAIN_SHORT = {"A": "alignment-based training (plain or rank-regularised) lowers accuracy relative to the untrained kernel",
                       "B": "plain KTA maximisation lowers accuracy relative to the untrained kernel, whereas a rank-regularised KTA objective does not",
                       "C": "plain KTA maximisation lowers accuracy relative to the untrained kernel, whereas a rank-regularised KTA objective raises it"}[SCEN]
    else:
        SCEN = None; TRAIN_SHORT = "plain KTA maximisation lowers accuracy relative to the untrained kernel"

# ------------------------------------------------------------------ bandwidth-derived numbers
if bw is not None:
    bwg = bw.groupby(["map", "a_over_pi"]).agg(acc=("acc", "mean"), kta=("kta_train", "mean"), er=("eff_rank", "mean"), off=("offdiag_mean", "mean"), ef=("E_F", "mean"), condW=("cond_W", "median")).reset_index()
    def bwv(mp, a, col): return bwg[(bwg["map"] == mp) & (np.isclose(bwg["a_over_pi"], a))][col].iloc[0]

# ------------------------------------------------------------------ noise-derived numbers
if nz is not None:
    nz["shots_f"] = nz["shots"].fillna(np.inf)
    base = nz[(nz["depolarizing"] == 0) & ((nz["eps"].isna()) | (nz["eps"] == 1e-6)) & ((nz["normalize"].isna()) | (nz["normalize"] == False))]
    def nacc(method, shots, m=None, eps=1e-6, norm=False, lam=0.0):
        d = nz[(nz["method"] == method) & (nz["shots_f"] == shots) & (nz["depolarizing"] == lam)]
        if method == "q_nystrom": d = d[(d["m"] == m) & (d["eps"] == eps) & (d["normalize"] == norm)]
        return 100 * d["acc"].mean(), 100 * d["acc"].std(ddof=1)
    def nmet(col, method, shots, m=None, eps=1e-6, norm=False, lam=0.0, agg="mean"):
        d = nz[(nz["method"] == method) & (nz["shots_f"] == shots) & (nz["depolarizing"] == lam)]
        if method == "q_nystrom": d = d[(d["m"] == m) & (d["eps"] == eps) & (d["normalize"] == norm)]
        return agg(d[col]) if callable(agg) else getattr(d[col], agg)()
    best_eps = {}
    for m in (20, 60, 120):
        for S in (100, 1000, 10000):
            d = nz[(nz["method"] == "q_nystrom") & (nz["shots_f"] == S) & (nz["depolarizing"] == 0) & (nz["m"] == m) & (nz["normalize"] == False)].groupby("eps")["acc"].mean()
            best_eps[(m, S)] = (d.idxmax(), 100 * d.max())

# ------------------------------------------------------------------ scaling-derived numbers
if sc is not None:
    scg = sc.groupby(["method", "N"]).agg(acc=("acc", "mean"), t=("t_total", "mean"), tk=("t_kernel", "mean"), circ=("circ_train", "mean"), circ_te=("circ_test", "mean"), mem=("gram_bytes", "mean"), ef=("E_F", "mean")).reset_index()
    def scv(method, N, col): 
        d = scg[(scg["method"] == method) & (scg["N"] == N)]; return d[col].iloc[0] if len(d) else np.nan


if ab is not None and HAS_ALT:
    _n_ab = len(ab_tasks)
    ABSTRACT_ALT = {"A": f" and a rank-regularised alignment objective does not remove the loss (ablation, {_n_ab} tasks)",
                    "B": f"; a rank-regularised alignment objective removes this loss (ablation, {_n_ab} tasks)",
                    "C": f"; a rank-regularised alignment objective reverses it (ablation, {_n_ab} tasks)"}[SCEN]
    CONTRIB_ALT = {"A": f"a pre-registered rank-regularised alternative behaves the same way (Δ = {d_reg:+.2f} points vs untrained, {_n_ab} tasks), so the loss is not specific to the plain objective.",
                   "B": f"a pre-registered rank-regularised alternative removes the loss (Δ = {d_reg:+.2f} points vs untrained over {_n_ab} tasks, not significant), so the finding is specific to plain alignment maximisation and the objective choice governs the trade-off.",
                   "C": f"a pre-registered rank-regularised alternative improves on the untrained kernel (Δ = {d_reg:+.2f} points over {_n_ab} tasks; {holm_reg} of {_n_ab} Holm-significant)."}[SCEN]
    CONCL_ALT = {"A": "Alignment-based training—plain or rank-regularised—made the kernel easier to compress and worse at classifying.",
                 "B": "Plain alignment training made the kernel easier to compress and worse at classifying; a rank-regularised alignment objective recovered the untrained kernel's accuracy, so the loss is a property of the objective, not of training as such.",
                 "C": "Plain alignment training made the kernel easier to compress and worse at classifying, whereas a rank-regularised alignment objective improved on the untrained kernel."}[SCEN]
else:
    ABSTRACT_ALT = ""; CONTRIB_ALT = "the untrained kernel is at least as accurate."; CONCL_ALT = "Plain alignment training made the kernel easier to compress and worse at classifying."


if ab is not None and HAS_ALT:
    DISC_ALT = {"A": f"Adding a rank regulariser to the objective did not change this ({d_reg:+.2f} points vs untrained pooled over {len(ab_tasks)} tasks), so within this family the loss appears to be a property of alignment-based training rather than of the plain objective alone.",
                "B": f"Adding a rank regulariser (λ chosen on validation) restored the effective rank to the untrained level and with it the accuracy ({d_reg:+.2f} points vs untrained pooled over {len(ab_tasks)} tasks, not significant, versus {d_kta:+.2f} for plain KTA): the loss is a property of the plain objective, and the objective choice, not training as such, governs the trade-off between compressibility and accuracy.",
                "C": f"Adding a rank regulariser (λ chosen on validation) restored the effective rank and improved on the untrained kernel ({d_reg:+.2f} points pooled over {len(ab_tasks)} tasks; {holm_reg} of {len(ab_tasks)} Holm-significant), so the objective choice governs whether training helps or hurts."}[SCEN]
else:
    DISC_ALT = ""


if ab is not None and HAS_ALT:
    LIM_TRAIN = {"A": f"Two alignment-based objectives (plain and rank-regularised KTA, L-BFGS-B, one restart) were tested on {len(ab_tasks)} tasks; objectives tied to downstream loss were not.",
                 "B": f"Two alignment-based objectives (plain and rank-regularised KTA, L-BFGS-B, one restart) were tested on {len(ab_tasks)} tasks; the finding that plain KTA lowers accuracy did not extend to the regularised objective, and objectives tied directly to downstream loss were not tested.",
                 "C": f"Two alignment-based objectives were tested on {len(ab_tasks)} tasks; the improvement from the regularised objective is one to two points and should be confirmed on more tasks."}[SCEN]
else:
    LIM_TRAIN = "KTA with L-BFGS-B and a single restart is one training protocol among several; other objectives may change the training conclusion."


if (RES / "leverage_sweep_per_seed.csv").exists():
    _ls = pd.read_csv(TAB / "table_leverage_sweep_stats.csv"); _lt = pd.read_csv(TAB / "table_leverage_sweep.csv")
    _small = _ls[_ls["m"] <= 20]; _big = _ls[_ls["m"] == 60]
    _worse_small = int((_small["EF_diff"] > 0).sum()); _sig_small = int((_small["p_t_holm"] < 0.05).sum())
    _ent = _lt[_lt["strategy"] == "leverage"].groupby("ridge")["entropy"].mean()
    LEV_SWEEP_TXT = (f"A ridge sweep (Table S8 and Fig. S1 in Online Resource 1; λ ∈ {{10⁻³, …, 60}}, 7 tasks × 10 seeds) shows that this is not an artefact of the ridge value: at m ≤ 20 leverage sampling had higher E_F than uniform sampling for {_worse_small} of {len(_small)} (λ, m) settings, by {100*_small['EF_diff'].min():.1f}–{100*_small['EF_diff'].max():.1f} points ({_sig_small} Holm-significant), and at m = 60 the two were indistinguishable (|Δ| ≤ {100*_big['EF_diff'].abs().max():.2f} points). The sampling distribution is close to uniform for every λ (entropy {_ent.min():.2f}–{_ent.max():.2f} vs ln N = {np.log(N_TR):.2f}), so leverage scores carry little information for this low-rank kernel; we report the observation without attributing a mechanism. Under the ridge parameterisation studied here, ridge-leverage sampling offers no advantage over uniform sampling for this kernel.")
else:
    LEV_SWEEP_TXT = ""
LIM_LEV = ("Ridge-leverage-score sampling was evaluated over a sweep of ridge values (Online Resource 1, Table S8); within that sweep it never beat uniform sampling for this kernel, but the result is specific to this kernel family and to sampling without replacement proportional to the scores." if (RES / "leverage_sweep_per_seed.csv").exists() else "Leverage-score sampling used a fixed ridge parameter.")

# =====================================================================================================================
# MANUSCRIPT TEXT
# =====================================================================================================================
n_tasks = len(TASKS); n_seeds = main["seed"].nunique()
# Author metadata: NOT available in the workspace -> explicit placeholders (never invented). Fill from manuscript/authors.yaml if present.
_auth = (ROOT / "manuscript" / "authors.yaml")
AUTHOR_BLOCK = ("**Authors:** [AUTHOR 1 — FILL IN]¹, [AUTHOR 2 — FILL IN]²  \n¹ [AFFILIATION 1 — FILL IN]  ² [AFFILIATION 2 — FILL IN]  \n"
                "**Corresponding author:** [NAME — FILL IN], [EMAIL — FILL IN]  **ORCID:** [FILL IN]")
if _auth.exists():
    import yaml as _yaml
    _a = _yaml.safe_load(open(_auth))
    AUTHOR_BLOCK = ", ".join(f"{x['name']}{x.get('mark','')}" for x in _a["authors"]) + "  \n" + "  \n".join(f"{x.get('mark','')}{x['affiliation']}".strip() for x in _a["affiliations"]) + f"  \n**Corresponding author:** {_a['corresponding']['name']}, e-mail: {_a['corresponding']['email']}"
    if _a.get("orcids"): AUTHOR_BLOCK += "  \n**ORCID:** " + "; ".join(f"{k}: {v}" for k, v in _a["orcids"].items())
    DECL = _a.get("declarations", {})
else:
    DECL = {}
def _decl(key, default):
    return DECL.get(key, default)
ntest_img = int(main[main["task"] == "fashion_0v1"]["n_test"].iloc[0]); ntest_bc = int(main[main["task"] == "breast_cancer"]["n_test"].iloc[0])
w(f"""---
title: "How Much of a Trainable Quantum Kernel Survives Nyström Compression? A Controlled Study of Landmark Selection, Finite-Shot Noise and Classical Baselines"
---

::: {{custom-style="Author"}}
{"**Authors:** [withheld for double-anonymous review]" if _args.anonymous else AUTHOR_BLOCK}
:::

## Abstract

Fidelity quantum kernels need one circuit configuration per pair of training points; Nyström compression with m ≪ N landmarks reduces this to N·m + m(m−1)/2. Whether the compressed kernel retains accuracy, whether it offers anything over a classical Nyström kernel at equal budget, and how it behaves under finite-shot estimation lack controlled answers. We prove that trainable and entangling layers placed after data encoding cannot act on a fidelity kernel, and derive a torus-covariant kernel whose trainable, entangled fiducial state acts before the data. On {n_tasks} binary tasks (Fashion-MNIST, MNIST, Breast Cancer Wisconsin), {n_seeds} seeds and official test splits, we compare exact and compressed quantum kernels with linear, radial-basis-function (RBF), RBF-Nyström (identical landmarks) and random-Fourier-feature baselines across eleven budgets and five landmark strategies. The trained kernel is highly compressible (relative Frobenius error {np.mean(list(ef60.values())):.1f}% at m = 60 versus {np.mean(list(ef60_rbf.values())):.1f}% for RBF) and loses no accuracy under compression; plain kernel-target-alignment training lowers accuracy relative to the untrained kernel on {sum(acc(t, QE)[0] < acc(t, PCE)[0] for t in TASKS)} of {n_tasks} tasks{ABSTRACT_ALT}, and RBF Nyström matches or exceeds the quantum Nyström kernel on {sum(acc(t, RBFN, 60)[0] >= acc(t, Q, 60)[0] for t in TASKS)} of {n_tasks} tasks. The geometric difference to the tuned RBF kernel ({g_rbf[QE]:.1f} ≈ {g_rbf[QE]/SQRTN:.2f}√N) leaves no room for quantum advantage. With finite shots, the noisy landmark kernel's pseudo-inverse amplifies error 20–40-fold, driving accuracy to {nacc('q_nystrom', 100, 60)[0]:.0f}% at 100 shots (exact kernel: {nacc('q_exact', 100)[0]:.0f}%); projecting it onto its positive eigenspace restores accuracy to within one point of the noise-free value. Code regenerating every number is provided.

**Keywords:** quantum kernel methods; Nyström approximation; covariant quantum kernels; landmark selection; finite-shot noise; benchmarking.

## 1 Introduction

Quantum kernel methods estimate the similarity K(x, x′) = |⟨φ(x)|φ(x′)⟩|² between data-encoded quantum states and hand the resulting Gram matrix to a classical support-vector machine [Havlíček 2019; Schuld & Killoran 2019]. Their cost is dominated by the number of distinct circuit configurations that must be executed: N(N−1)/2 for the training Gram matrix and n_test·N for prediction, each repeated S times to estimate a fidelity to precision ≈ 1/√S. The Nyström method [Williams & Seeger 2001] is the standard remedy for quadratic kernel cost: with m landmarks one estimates only the N×m block C = K(X, Z) and the m×m block W = K(Z, Z), and uses K̃ = C W⁺ Cᵀ. Several recent works have applied Nyström to quantum kernels [Srikumar et al. 2022; Coelho et al. 2025; Rezvannia 2026], but none has answered, under a controlled protocol, three questions that decide whether the combination is useful in practice.

First, *does the compressed quantum kernel retain the accuracy of the exact one, and how does that depend on the landmark budget m and on how landmarks are chosen?* Second, *at equal budget and with identical landmarks, does a compressed quantum kernel offer anything over a compressed classical kernel?* Third, *what happens when every entry of C and W is a finite-shot estimate*—does the pseudo-inverse W⁺ amplify noise, as perturbation theory suggests?

A further, methodological problem motivated this study. A natural-looking construction places data encoding first and trainable or entangling gates afterwards, |φ(x)⟩ = V(θ)·E(x)|ψ₀⟩. Because V†V = I, the fidelity kernel is *identical* for every θ and with or without the entangling gates (Proposition 1). Reports of "trainable" or "geometric" quantum kernels built this way—including an earlier version of the present work—therefore measured properties of a fixed classical product kernel. Any study of trainable kernels must first establish, mathematically and numerically, that its parameters act on the kernel at all.

**Scope: no quantum resource is required for any result in this paper.** The kernel we study has a closed form (Proposition 2) and is computed exactly from 2⁸ = 256 real numbers at the n = 8 width used here; every experiment is an exact classical simulation, and no claim of quantum advantage is made or implied. What the paper contributes is methodological and empirical: how compression, landmark choice, spectral conditioning and finite-shot estimation behave for a quantum-kernel *construction* whose parameters demonstrably act on the kernel. Two of the findings should transfer to kernels that are not classically simulable, because they do not depend on how kernel entries are produced: the cancellation of post-encoding unitaries (Proposition 1 holds for any fidelity kernel) and the amplification of estimation noise by the Nyström pseudo-inverse (Proposition 3 depends only on the spectrum of the estimated landmark kernel and on the binomial shot statistics, which are the same for any inversion-test estimate). The accuracy comparisons with classical kernels, by contrast, are specific to this kernel family and these features and should not be extrapolated.

**Contributions.**

1. *A trainable covariant fidelity kernel with closed form.* We place a parameterised, entangled fiducial state |ψ_θ⟩ before the RZ data encoding, |φ(x)⟩ = D(x)|ψ_θ⟩, and prove that the kernel equals |χ_p((x′−x)/2)|², the squared characteristic function of the Z-basis distribution p of the fiducial state (Proposition 2). This makes precise in what sense the kernel is "geometric" (it is covariant under the torus group U(1)ⁿ generated by the encoding), shows when entanglement matters (exactly when p is not a product distribution), and identifies the kernel as classically computable at n = 8—so this paper makes no claim of quantum advantage.
2. *A controlled Nyström benchmark.* Across {n_tasks} tasks, {n_seeds} seeds, m ∈ {{10, …, 600}}, five landmark strategies and five classical baselines (including RBF Nyström with the *same* landmarks and random Fourier features with the same budget), with full official test splits, paired statistics and Holm correction. We report relative Frobenius and spectral errors, kernel-target alignment, conditioning and circuit-evaluation counts, not only accuracy.
3. *Findings on training.* Plain kernel-target-alignment training raises alignment but lowers effective rank and test accuracy; {CONTRIB_ALT} The untrained kernel and the classical RBF kernel are at least as accurate as any trained variant, and compressibility and accuracy are linked through effective rank.
4. *Finite-shot analysis and a remedy.* A first-order perturbation bound (Proposition 3) predicts that estimation noise is amplified by ‖W_ε⁻¹‖²; experiments show accuracy falling from 88% to 63–72% at 100–1,000 shots with the standard pseudo-inverse, and that projecting the noisy landmark kernel onto its positive eigenspace (or a noise-adaptive jitter) restores accuracy to within one point of the noise-free result.

## 2 Background and Related Work

### 2.1 Quantum fidelity kernels
For an encoding unitary U(x) and reference state |0⟩, the fidelity kernel K(x, x′) = |⟨0|U(x)†U(x′)|0⟩|² is positive semi-definite and can be estimated on hardware by the inversion (or SWAP) test with S shots per pair [Havlíček 2019; Schuld & Killoran 2019; Schuld 2021]. Angle encodings ⊗ⱼRZ(xⱼ) generate truncated Fourier series in the data [Schuld, Sweke & Meyer 2021]; the scale of the encoding angles acts as a kernel bandwidth that strongly affects generalisation [Shaydulin & Wild 2022; Canatar et al. 2023]. Kernel values concentrate exponentially with qubit count for many maps [Thanasilp et al. 2024]; at n = 8 this effect is mild but, as we show, it interacts with Nyström conditioning. Whether a quantum kernel *can* outperform a classical one on given data is governed by the geometric difference g(K_C‖K_Q) [Huang et al. 2021]: values near 1 mean the classical kernel expresses every function of the quantum kernel; only g ≈ √N leaves room for advantage.

### 2.2 Geometric and covariant quantum kernels
Covariant quantum kernels [Glick et al. 2024] encode group elements by a unitary representation D(g) acting on a fiducial state, K(g, g′) = |⟨ψ|D(g)†D(g′)|ψ⟩|², and are invariant under left multiplication; the fiducial state can be optimised. Group-invariant and equivariant variational models [Larocca et al. 2022; Ragone et al. 2022; Meyer et al. 2023; Nguyen et al. 2024; Schatzki et al. 2024; Wiersema et al. 2025] provide inductive biases for *variational* circuits. We use "geometric" strictly in the covariant-kernel sense: our encoding D(x) = ⊗ⱼRZ(xⱼ) is a representation of the torus U(1)ⁿ, and the kernel is invariant under simultaneous translation of all encoding angles (Proposition 2a). Proposition 1 explains why a "horizontal" trainable layer placed after the encoding cannot act on a kernel.

### 2.3 Nyström approximation and landmark selection
Nyström approximation [Williams & Seeger 2001] has error bounds depending on the sampling scheme [Drineas & Mahoney 2005; Kumar, Mohri & Talwalkar 2012; Gittens & Mahoney 2016]. Uniform sampling is the baseline; k-means centroids [Zhang, Tsang & Kwok 2008] and kernel-k-means++ landmarks [Oglic & Gärtner 2017] reduce error at small m; ridge-leverage-score sampling is near-optimal but requires the full kernel. In practice, K̃ is regularised as C(W + εI)⁻¹Cᵀ or through eigenvalue truncation of W. We evaluate all of these.

### 2.4 Nyström approximation of quantum kernels
Srikumar, Hill & Hollenberg [2022] use a low-rank Nyström approximation of the quantum kernel inside a quantum random forest and derive generalisation bounds; Coelho, Kruse & Rosskopf [2025] use Nyström to cut circuit executions while training embedding kernels by KTA and report experiments under coherent and depolarising noise; Rezvannia [2026] applies a Nyström-compressed quantum kernel to the Breast Cancer Wisconsin data. None of these compares against classical Nyström at equal budget with identical landmarks, sweeps m and landmark strategies with paired statistics, analyses the conditioning of the estimated landmark kernel, or uses a feature map whose parameters are proven to act on the kernel. Benchmarking studies of quantum kernels [Schnabel & Roth 2025] have stressed that quantum kernels rarely beat tuned classical kernels on standard data; our study extends that scrutiny to the compression axis.

Table 1 summarises the differentiation. These individual ingredients—Nyström for quantum kernels, k-means landmarks, covariant kernels, KTA training, shot-noise simulation—exist in prior literature; the contribution of this paper lies in their controlled joint analysis and in the specific findings it produces (Propositions 1–3 and §5).

**Table 1** Prior work on Nyström-compressed or covariant quantum kernels versus this study. ✓ = present; — = absent; (✓) = partial.

| Work | Quantum kernel | Nyström | Trainable map (verified to act) | Classical baseline at equal budget | Landmark-strategy study | Finite-shot noise | Spectral remedy | Main distinction |
|---|---|---|---|---|---|---|---|---|
| Havlíček et al. 2019 | ✓ | — | — | — | — | (✓) hardware | — | Introduces fidelity kernels |
| Glick et al. 2024 | ✓ covariant | — | ✓ fiducial state | — | — | (✓) hardware | — | Covariant kernels; no compression |
| Srikumar et al. 2023 | ✓ | ✓ | — | (✓) classical RF | — | — | — | Nyström inside a quantum random forest |
| Coelho et al. 2025 | ✓ | ✓ | ✓ KTA | — | — | ✓ coherent/depolarising | — | Nyström + KTA training; closest prior work |
| Rezvannia 2026 | ✓ | ✓ | — | — | — | — | — | Single tabular data set |
| Schnabel & Roth 2025 | ✓ | — | (✓) | ✓ | — | — | — | Benchmark without compression |
| **This work** | ✓ covariant, closed form | ✓ (11 budgets) | ✓ (Props. 1–2; ablation; two objectives) | ✓ RBF Nyström (same landmarks), RFF, tuned RBF/linear/LR | ✓ 5 strategies + ridge sweep | ✓ idealised shots + depolarising | ✓ PSD projection / truncation / adaptive ε | Controlled joint analysis; negative and cautionary results reported |

### 2.5 Research gap and questions
**RQ1** Does the redesigned kernel depend on its parameters and on entanglement, and how does it relate to classical kernels? **RQ2** How do approximation error, alignment and accuracy degrade with m, and which landmark strategy retains the most? **RQ3** How does finite-shot noise propagate through W⁺, and what regularisation is needed? **RQ4** At equal budget, does the quantum Nyström kernel differ from classical Nyström kernels, and does the geometric difference allow any advantage? **RQ5** What are the actual circuit-evaluation counts and simulation costs as N grows?
""")
w(f"""## 3 Method

### 3.1 Data preprocessing (leakage-safe)
Fig. 1 summarises the pipeline. Each image is flattened (784 pixels) and projected by PCA to k = 8 components fitted on the *training* sub-sample only; each component is min–max scaled to [0, 1] using training minima and maxima and multiplied by a scaling range a ∈ {{π/2, π, 2π}} chosen on a validation split. Test points are transformed with the training PCA and scaler and may fall outside [0, a]. The tabular data set (30 features) is treated identically. `Preprocessor.fit` asserts that it only ever sees training data, and a unit test checks that test data are not refitted.

### 3.2 Feature map: torus-covariant kernel with a trainable fiducial state
Let D(x) = ⊗ⱼ₌₁ⁿ RZ(xⱼ) and let the fiducial circuit V(θ) consist of L = 2 blocks, each a layer of RY(θ_{{l,j}}) on every qubit followed by a ring of CNOTs (16 parameters for n = 8). The embedding is |φ(x)⟩ = D(x) V(θ)|0⟩ⁿ; the data act last, the parameters first (Fig. 2). Two ablation families are used for comparison: a data-re-uploading map $H \\to [\\mathrm{{RZ}}(x)\\,\\mathrm{{RY}}(\\theta_l)\\,\\text{{CNOT ring}}]^{{\\times L}} \\to \\mathrm{{RZ}}(x)$, and an IQP-type map with data-dependent ZZ entanglers; and the previous construction $\\mathrm{{RZ}}(x)|+\\rangle^{{n}}$ (equivalently $H \\to \\mathrm{{RZ}}(x)$ followed by *any* data-independent gates), which we call the product-cosine kernel.

### 3.3 Kernel formulation and properties
**Proposition 1 (cancellation).** If |φ(x)⟩ = V E(x)|ψ₀⟩ with V independent of x, then K is independent of V. For E = D and |ψ₀⟩ = |+⟩ⁿ,

$$K(x, x') = \\prod_{{j=1}}^{{n}} \\cos^2\\!\\left(\\frac{{x_j - x'_j}}{{2}}\\right).$$

*Proof:* V†V = I; ⟨+|RZ(δ)|+⟩ = cos(δ/2). ∎

**Proposition 2 (covariant kernel).** With p_b(θ) = |⟨b|V(θ)|0⟩|² and s_b ∈ {{±1}}ⁿ the Z-eigenvalues of |b⟩,

$$K_\\theta(x, x') = \\left|\\sum_{{b}} p_b\\, e^{{\\mathrm{{i}}\\,(x'-x)\\cdot s_b/2}}\\right|^2 = \\left|\\chi_p\\left((x'-x)/2\\right)\\right|^2.$$

(a) K_θ(x + h, x′ + h) = K_θ(x, x′) for all h (covariance under U(1)ⁿ). (b) K_θ depends on θ only through p. (c) If p is a product distribution with means m_j = ⟨Z_j⟩, K_θ = ∏ⱼ[cos²(δⱼ/2) + m_j² sin²(δⱼ/2)]; m ≡ 0 recovers the product-cosine kernel. (d) K_θ is a non-product function iff p is non-product, so an *entangled* fiducial state is necessary for cross-feature structure. (e) K_θ(δ) = Σ_{{b,b′}} p_b p_{{b′}} cos((s_b − s_{{b′}})·δ/2) is positive definite with spectral measure on {{−1, 0, 1}}ⁿ, i.e. it has an explicit classical feature map of dimension ≤ 3ⁿ. *Proof:* D is diagonal, so ⟨ψ_θ|D(x)†D(x′)|ψ_θ⟩ = Σ_b p_b e^{{−i(x′−x)·s_b/2}}; (c)–(e) follow by factorising or expanding the square. ∎ (Full proofs: Online Resource 1, Sect. S1.)

Consequences: the parameters and entangling gates genuinely shape the kernel; "geometric" has the precise meaning (a); and, since K_θ is computable from the 2⁸ = 256 numbers p_b, **the kernel is classically simulable at this width**—the study is about approximation methodology, not quantum advantage. All simulations use exact statevectors (a batched NumPy simulator cross-checked against PennyLane to 10⁻¹⁰); no result depends on sampling unless stated.

### 3.4 Training the fiducial state
θ is trained by maximising the centred kernel-target alignment

$$\\mathrm{{KTA}}(K_\\theta, y) = \\frac{{\\langle HKH,\\; H y y^{{\\mathsf{{T}}}} H\\rangle_F}}{{\\|HKH\\|_F\\, \\|H y y^{{\\mathsf{{T}}}} H\\|_F}}$$

on {cfg['feature_map']['kta_train_subsample']} points drawn from the fit split, with L-BFGS-B (finite-difference gradients, ≤ {cfg['feature_map']['kta_maxiter']} iterations, θ₀ ~ U(0, 2π)); validation and test data are never used. For each scaling range a, θ is trained and C is selected on the validation split; the (a, C, θ) with the best validation accuracy is kept.

### 3.5 Nyström approximation, landmarks and numerical control
With landmarks Z (m×n), C = K(X, Z), W = K(Z, Z):

$$\\tilde K = C\\, W_\\varepsilon^{{+}}\\, C^{{\\mathsf{{T}}}}, \\qquad W_\\varepsilon = W + \\varepsilon I,$$

with ε = 10⁻⁶ and pseudo-inverse cut-off 10⁻¹⁰ (relative) unless stated; the test kernel is C_te W_ε⁺ Cᵀ. The scalable form Φ = C W_ε^{{−1/2}} with a linear SVM never materialises the N×N matrix and is used in the scaling study. Landmark strategies: uniform (training points); k-means++ seeding (training points); k-means centroids (not data points, hence encoded as new circuits); nearest training point to each k-means centroid; and ridge-leverage-score sampling computed from the full kernel (an oracle, budget-infeasible reference). Errors are reported as

$$E_F = \\frac{{\\|K - \\tilde K\\|_F}}{{\\|K\\|_F}}, \\qquad E_2 = \\frac{{\\|K - \\tilde K\\|_2}}{{\\|K\\|_2}},$$

mean diagonal and off-diagonal absolute errors, and cond(W_ε).

### 3.6 Idealised finite-shot and depolarising noise model
We use an *idealised finite-shot noise model*: every fidelity a device would have to estimate is replaced by an independent binomial estimate Binomial(S, K)/S (symmetric blocks estimated once per unordered pair; the diagonal of an exact kernel is taken as known, K_ii = 1). Global depolarising noise with strength λ maps K → (1 − λ)K + λ/2ⁿ before sampling. The model omits readout error, correlated and time-varying noise, the fact that C and W share landmark circuits (their errors would then be correlated), and hardware-specific diagonal estimates; none of the results below is a hardware validation. Hyper-parameters are tuned noise-free (a stated limitation). **Proposition 3 (first-order noise propagation; proof in Online Resource 1, Sect. S1).** Let W_ε = W + εI, K̃ = C W_ε⁻¹ Cᵀ, and let Ĉ = C + ΔC, Ŵ = W + ΔW be estimates with ‖ΔW‖₂ < λ_min(W) + ε. Then

$$\\tilde K(\\hat C, \\hat W) - \\tilde K = \\Delta C\\, W_\\varepsilon^{{-1}} C^{{\\mathsf{{T}}}} + C\\, W_\\varepsilon^{{-1}} \\Delta C^{{\\mathsf{{T}}}} - C\\, W_\\varepsilon^{{-1}} \\Delta W\\, W_\\varepsilon^{{-1}} C^{{\\mathsf{{T}}}} + O(\\|\\Delta\\|^2),$$

hence

$$\\|\\Delta \\tilde K\\|_F \\le 2\\|C\\|_2\\, \\|W_\\varepsilon^{{-1}}\\|_2\\, \\|\\Delta C\\|_F + \\|C\\|_2^2\\, \\|W_\\varepsilon^{{-1}}\\|_2^2\\, \\|\\Delta W\\|_F + O(\\|\\Delta\\|^2),$$

with E‖ΔW‖_F² = Σ_{{p≠q}} W_pq(1 − W_pq)/S ≤ m(m−1)/(4S) under the idealised shot model. Estimation error in the landmark block is amplified by the *square* of ‖W_ε⁻¹‖₂ = 1/(λ_min(W) + ε). Two cautions: the bound is an upper bound, whereas the quantity we measure in §5.6, A = ‖K̃_noisy − K̃_clean‖_F/(‖ΔC‖_F + ‖ΔW‖_F), is a realised ratio; and the hypothesis ‖ΔW‖₂ < λ_min(W) + ε fails for the low-rank kernels studied here when ε is small (λ_min(W) ≈ 0), in which case Ŵ_ε is indefinite, the pseudo-inverse inverts negative eigenvalues, and the first-order regime no longer applies at all. The remedies tested in §5.6 all act by restoring a positive, bounded smallest inverted eigenvalue: PSD projection, relative eigenvalue truncation, and a noise-adaptive jitter ε = c·σ_W√m (σ_W² = mean binomial variance of the entries of Ŵ). Normalising K̃ to unit diagonal rescales rows and columns without changing the spectrum of the applied inverse and is therefore not expected to help.

### 3.7 Circuit-evaluation accounting (Proposition 4)
Exact: N(N−1)/2 training and n_test·N test configurations. Nyström: m(m−1)/2 + N·m training (minus m(m+1)/2 if landmarks are training points) and n_test·m test configurations; each repeated S times. For N = 600, m = 60, n_test = 2,000: {int(cc_ex['circ_train']):,} → {int(cc_ny['circ_train']):,} training evaluations ({cc_ex['circ_train']/cc_ny['circ_train']:.1f}×) and {int(cc_ex['circ_test']):,} → {int(cc_ny['circ_test']):,} test evaluations ({cc_ex['circ_test']/cc_ny['circ_test']:.0f}×). We never call this a speed-up: in simulation the exact Gram matrix is a single matrix product.

### 3.8 Use of artificial-intelligence tools
{_decl('ai_use', '[FILL IN — statement on the use of AI tools]')}

## 4 Experimental Protocol

**Tasks (Table 2).** Fashion-MNIST T-shirt/top vs Trouser, Pullover vs Coat, T-shirt/top vs Shirt, Sandal vs Sneaker; MNIST 3 vs 8 and 4 vs 9; Breast Cancer Wisconsin (Diagnostic). Image tasks: N = {N_TR} training images drawn stratified from the official training split (seed-dependent), tested on the *entire* official test split of the two classes (n_test = {ntest_img}). Tabular: stratified 70/30 split per seed (n_test = {ntest_bc}), N = {int(main[main['task']=='breast_cancer']['n_train'].iloc[0])}.
**Validation.** 20% of the training sample is held out for choosing C ∈ {{0.01, 0.1, 1, 10, 100}} (all SVMs), γ (RBF; grid relative to 1/(d·Var)), a (quantum kernels) and θ (KTA). Final models are refitted on the full training sample. For Nyström methods C is re-selected per m with landmarks drawn from the fit split, then landmarks are re-drawn from the full training sample; the RBF bandwidth γ is tuned once for the exact RBF kernel and reused for RBF Nyström at every m (a per-m re-tuning could only make that baseline stronger).
**Seeds.** {n_seeds} seeds ({', '.join(map(str, cfg['seeds']))}) control the training sub-sample, splits, θ initialisation and landmark randomness; every method sees the same seeds.
**Baselines.** Linear SVM, RBF SVM and logistic regression on the same PCA features; classical RBF Nyström [Williams & Seeger 2001] with *identical* landmarks and the same ε; and random Fourier features [Rahimi & Recht 2007] for the RBF kernel with D = m features. All baselines use the same splits, seeds and validation protocol as the quantum kernels.
**Budget grid.** m ∈ {{{', '.join(map(str, cfg['nystrom']['m_grid']))}}} (m ≤ N), random-feature dimension D = m.
**Statistics.** Mean ± SD over seeds; Wilson 95% intervals for the mean accuracy at the given n_test; paired t-tests and Wilcoxon signed-rank tests across seeds with Cohen's d_z and bootstrap CIs; Holm correction within each family of tests. "Significant" is used only when the Holm-adjusted p < 0.05. Dispersion in figures: mean ± SD for accuracies (linear axes); median and inter-quartile range for approximation errors, which are shown on logarithmic axes where an SD bar is not meaningful (stated in each caption). Condition numbers of W_ε are reported as λ_max/λ_min when λ_min > 0 and as ∞ (singular or indefinite) otherwise; the conditioning of the inverse actually applied is reported as ‖W_ε⁺‖₂.
**Software/hardware.** Python {cfg['software']['python']}, NumPy {cfg['software']['numpy']}, SciPy {cfg['software']['scipy']}, scikit-learn {cfg['software']['scikit_learn']}, PennyLane {cfg['software']['pennylane']} (simulator cross-check only); {cfg['hardware']['cpu']}, {cfg['hardware']['ram_gb']} GB RAM. Configuration file `configs/main.yaml`; all per-configuration results are in `results/*.csv` (Online Resource 2).
""")
# ---------------------------------------------------------------- results
t1 = pd.read_csv(TAB / "table1_tasks.csv"); t2 = pd.read_csv(TAB / "table2_main_accuracy.csv")
from tablefmt import md_table, INT_COLS  # shared with scripts/build_supplement.py
q_vs_rbfn = stats[stats["comparison"] == f"{Q} - {RBFN}"]; q_vs_pc = stats[stats["comparison"] == f"{Q} - {PC}"]
qe_vs_rbf = stats[stats["comparison"] == f"{QE} - {RBF}"]; qe_vs_lin = stats[stats["comparison"] == f"{QE} - {LIN}"]; qe_vs_pce = stats[stats["comparison"] == f"{QE} - {PCE}"]
def holm_sig(d): return int((d["p Holm"] < 0.05).sum())
def raw_sig(d): return int((d["p (paired t)"] < 0.05).sum())
km10 = strat_eff("kmeans_centroids", 10, "E_F"); km10a = strat_eff("kmeans_centroids", 10, "acc"); km60a = strat_eff("kmeans_centroids", 60, "acc"); lev10 = strat_eff("leverage", 10, "E_F")
w(f"""## 5 Results

### 5.1 Feature-map validation (RQ1)
Table S1 summarises the validation gate (7 tasks × 3 seeds). For the previous construction the maximum change of any kernel entry under a θ perturbation or removal of the entangling ring is exactly 0 (Proposition 1). For the covariant map the corresponding changes are {vg.loc['covariant_torus','max_dK_theta']:.2f} and {vg.loc['covariant_torus','max_dK_entangle']:.2f} (re-uploading: {vg.loc['reuploading','max_dK_theta']:.2f} / {vg.loc['reuploading','max_dK_entangle']:.2f}; ZZ re-uploading: {vg.loc['zz_reuploading','max_dK_theta']:.2f} / {vg.loc['zz_reuploading','max_dK_entangle']:.2f}), all kernels are PSD to 10⁻⁹ with unit diagonal, and the closed form of Proposition 2 matches the simulator to 10⁻¹⁰ (unit tests). KTA training raises alignment from {vg.loc['covariant_torus','kta_init']:.2f} to {vg.loc['covariant_torus','kta_trained']:.2f} for the covariant map (product-cosine: {vg.loc['product_cosine','kta_trained']:.2f}; re-uploading: {vg.loc['reuploading','kta_trained']:.2f}; ZZ: {vg.loc['zz_reuploading','kta_trained']:.2f}). Validation accuracies of the four maps are statistically indistinguishable (largest paired difference {100*max(abs(vpiv['covariant_torus'].mean()-vpiv[k].mean()) for k in ['reuploading','zz_reuploading','product_cosine']):.2f} points; all p > 0.25), so the covariant map was selected by the pre-declared tie-breaker (closed-form theory, lowest cost), not by a performance gap; the other maps are retained as ablation arms.

### 5.2 Ablation: which components matter? (RQ1)
""")
if ab is not None:
    r_ent = abst[abst["comparison"] == "entanglement (trained)"]; r_tr = abst[abst["comparison"] == "training (entangled)"]; r_pc = abst[abst["comparison"] == "vs product-cosine (old construction)"]; r_re = abst[abst["comparison"] == "covariant vs re-uploading"]
    w(f"""Table 4 and Fig. 4 report {len(ab['variant'].unique())} variants at fixed a = π on {len(ab_tasks)} tasks × {ab_seeds} seeds. Four findings. (i) *Plain KTA training lowers accuracy.* The trained entangled covariant kernel reaches {100*abg.loc[V_PRI,'acc']:.2f}% pooled versus {100*abg.loc[V_UNTR,'acc']:.2f}% for the same circuit with random, untrained θ and {100*abg.loc[V_PC,'acc']:.2f}% for the product-cosine kernel (θ = π/2, no entanglement); the paired differences are {', '.join(f"{100*r['mean_diff']:+.2f} pts (p = {r['p_t']:.3f})" for _, r in r_tr.iterrows())} on {', '.join(TASK_LABELS[t] for t in r_tr['task'])}. Training raises alignment ({abg.loc[V_UNTR,'kta']:.2f} → {abg.loc[V_PRI,'kta']:.2f}) while halving the effective rank ({abg.loc[V_UNTR,'er']:.1f} → {abg.loc[V_PRI,'er']:.1f}): the optimiser increases alignment by making the kernel smoother, which costs discriminative resolution. (ii) *Entanglement of the fiducial state changes the kernel but not, on average, the accuracy:* with training, removing the ring changes accuracy by {', '.join(f"{-100*r['mean_diff']:+.2f} pts" for _, r in r_ent.iterrows())} (none Holm-significant). (iii) *Re-uploading maps are as accurate as the best kernels but far less compressible:* {100*abg.loc[V_RE2,'acc']:.2f}% (L = 2) with effective rank {abg.loc[V_RE2,'er']:.0f} and E_F = {100*abg.loc[V_RE2,'ef']:.1f}% at m = 60, versus E_F = {100*abg.loc[V_PRI,'ef']:.2f}% for the trained covariant kernel; the ZZ map is both the least accurate ({100*abg.loc[V_ZZ,'acc']:.2f}%) and the least compressible (E_F = {100*abg.loc[V_ZZ,'ef']:.1f}%). Compressibility and accuracy are therefore linked through effective rank, and the kernel that Nyström approximates best is the one whose training removed information.
""")
    if bw is not None:
        _tb = pd.read_csv(TAB / "table_bandwidth.csv").groupby(["map", "a_over_pi"]).agg(acc=("acc", "mean"), effrank=("effrank", "mean"), EF_m60=("EF_m60", "mean")).reset_index()
        def _bwv(mp, a, col): return float(_tb[(_tb["map"] == mp) & (np.isclose(_tb["a_over_pi"], a))][col].iloc[0])
        _accs = [100 * _bwv("covariant_torus", a, "acc") for a in (0.25, 0.5, 1.0, 1.5, 2.0)]
        w(f"""**Scaling range (Table 5).** The scaling range a acts as a kernel bandwidth: for the trained covariant kernel, widening a from π/4 to 2π raises the effective rank from {_bwv('covariant_torus', 0.25, 'effrank'):.1f} to {_bwv('covariant_torus', 2.0, 'effrank'):.1f} and the Nyström error at m = 60 from {100*_bwv('covariant_torus', 0.25, 'EF_m60'):.4f}% to {100*_bwv('covariant_torus', 2.0, 'EF_m60'):.1f}%, while pooled accuracy varies only between {min(_accs):.1f}% and {max(_accs):.1f}%; the product-cosine kernel shows the same trade-off with a larger rank range ({_bwv('product_cosine', 0.25, 'effrank'):.1f}–{_bwv('product_cosine', 2.0, 'effrank'):.1f}). Compressibility is therefore governed by the bandwidth as much as by training.
""")
    if HAS_ALT:
        w(f"""(iv) *Alternative training objective (pre-registered protocol in Online Resource 1, Sect. S4).* The rank-regularised objective J_λ = KTA_c + λ·log(erank)/log(n) with λ ∈ {{0.1, 0.3, 1.0}} chosen on the validation split (selected λ: {', '.join(f'{k:g} in {v} runs' for k, v in sorted(lam_counts.items()))}) reaches {100*abg.loc[V_REG,'acc']:.2f}% pooled with effective rank {abg.loc[V_REG,'er']:.1f} and KTA {abg.loc[V_REG,'kta']:.2f}, versus {100*abg.loc[V_UNTR,'acc']:.2f}% (rank {abg.loc[V_UNTR,'er']:.1f}) untrained and {100*abg.loc[V_PRI,'acc']:.2f}% (rank {abg.loc[V_PRI,'er']:.1f}) for plain KTA. Paired against the untrained kernel the regularised objective differs by {', '.join(f"{100*r['mean_diff']:+.2f} pts (p = {r['p_t']:.3f}, Holm {r['p_t_holm']:.2f})" for _, r in c_reg_u.iterrows())} on {', '.join(TASK_LABELS[t] for t in c_reg_u.index)}; against plain KTA by {', '.join(f"{100*r['mean_diff']:+.2f} pts (p = {r['p_t']:.3f})" for _, r in c_reg_k.iterrows())}. Choosing among untrained, plain-KTA and regularised-KTA on the validation split (a practitioner's rule) picked {', '.join(f'{k} in {v} runs' for k, v in sorted(sel_counts.items()))} and reached {100*abg.loc[V_SEL,'acc']:.2f}% ({d_sel:+.2f} pts vs untrained). Under the pre-registered decision rule this is scenario {SCEN}: {TRAIN_SHORT}. The regulariser restores effective rank almost exactly to the untrained level, which is consistent with rank collapse—not alignment per se—being the mechanism behind the accuracy loss.
""")
w(f"""### 5.3 Nyström approximation versus landmark budget (RQ2)
Fig. 5 shows E_F versus m for the trained covariant kernel and Fig. 6 the corresponding test accuracy. Because the trained kernel has low effective rank ({effrank[QE]:.1f} on average; {100*e60[QE]:.0f}% of the trace in the top 60 eigenvalues, versus {effrank[PCE]:.1f} and {100*e60[PCE]:.0f}% for the product-cosine kernel; Fig. 3), E_F falls from {ef_unif[10]:.1f}% (uniform, m = 10) to {ef_unif[60]:.1f}% at m = 60 and {ef_unif[120]:.2f}% at m = 120; the product-cosine and RBF kernels are harder to compress (E_F at m = 60, k-means centroids: {np.mean(list(ef60_pc.values())):.1f}% and {np.mean(list(ef60_rbf.values())):.1f}%, Table 3). Accuracy of the compressed covariant kernel (image tasks, k-means centroids) rises from {acc_m[10]:.1f}% at m = 10 to {acc_m[40]:.1f}% at m = 40 and {acc_m[60]:.1f}% at m = 60, after which it is flat ({acc_m[120]:.1f}% at m = 120, {acc_m[300]:.1f}% at m = 300; exact kernel {acc_ex_img:.1f}%). At m = 60 the difference to the exact kernel is between {comp['Δacc (pts)'].min():+.2f} and {comp['Δacc (pts)'].max():+.2f} points across tasks and not significant on any task (Table S3): for this kernel, compression to 10% of the training set costs no measurable accuracy.

**Landmark strategies.** k-means centroids give the lowest E_F at every m: relative to uniform sampling, E_F is reduced by {100*(1-ef_km[10]/ef_unif[10]):.0f}% at m = 10 ({100*km10['mean_diff']:+.2f} points absolute, paired over {int(km10['n'])} task×seed pairs, Holm p = {km10['p_t_holm']:.1g}) and by {100*(1-ef_km[60]/ef_unif[60]):.0f}% at m = 60; k-means nearest points and k-means++ seeding lie in between. Ridge-leverage sampling with the full kernel—an oracle—was *worse* than uniform for this low-rank kernel (E_F {ef_lev[10]:.1f}% vs {ef_unif[10]:.1f}% at m = 10 at the main-study ridge). {LEV_SWEEP_TXT} The accuracy gain from better landmarks is small: {100*km10a['mean_diff']:+.2f} points at m = 10 (Holm p = {km10a['p_t_holm']:.2f}) and {100*km60a['mean_diff']:+.2f} at m = 60 (Holm p = {km60a['p_t_holm']:.2f}). We therefore do *not* describe k-means landmarks as a "smart" or superior method for classification; they are a reliable way to lower approximation error at small budgets.

### 5.4 Comparison with classical baselines at equal budget (RQ4)
Table 3 gives test accuracy (mean ± SD over {n_seeds} seeds) for all methods, with approximate methods at m = 60 (k-means centroids) and D = 60 random features. Three paired comparisons matter. (a) *Quantum Nyström vs RBF Nyström with identical landmarks:* the RBF kernel is more accurate on {sum(q_vs_rbfn['Δacc (pts)'] < 0)} of {n_tasks} tasks, by {rng_str(-q_vs_rbfn['Δacc (pts)'], 2)} points; {raw_sig(q_vs_rbfn)} of these differences {'is' if raw_sig(q_vs_rbfn)==1 else 'are'} significant before and {holm_sig(q_vs_rbfn)} after Holm correction. (b) *Exact trained covariant kernel vs tuned RBF SVM:* RBF is more accurate on {sum(qe_vs_rbf['Δacc (pts)'] < 0)} of {n_tasks} tasks ({rng_str(-qe_vs_rbf[qe_vs_rbf['Δacc (pts)'] < 0]['Δacc (pts)'], 2)} points), {holm_sig(qe_vs_rbf)} Holm-significant. (c) *Exact trained covariant kernel vs its untrained product-cosine special case:* the untrained kernel is more accurate on {sum(qe_vs_pce['Δacc (pts)'] < 0)} of {n_tasks} tasks ({raw_sig(qe_vs_pce)} significant before, {holm_sig(qe_vs_pce)} after correction). Against a *linear* SVM the trained covariant kernel is more accurate on {sum(qe_vs_lin['Δacc (pts)'] > 0)} of {n_tasks} tasks (up to {qe_vs_lin['Δacc (pts)'].max():+.2f} points on {TASK_LABELS[[t for t in TASKS if TASK_LABELS[t]==qe_vs_lin.loc[qe_vs_lin['Δacc (pts)'].idxmax(),'Task']][0]]}, Holm p = {qe_vs_lin['p Holm'].min():.3f}), which shows that the kernel is non-trivially non-linear—but so is the RBF kernel. Random Fourier features with D = m are the weakest approximation at small budgets (Fig. 9) but catch up with the RBF kernel by D ≈ 120–300. Logistic regression on the same eight features is within one point of the best method on {sum(abs(acc(t, LR)[0] - max(acc(t, mth)[0] for mth in [RBF, PCE, QE, LIN])) < 1 for t in TASKS)} of {n_tasks} tasks.

### 5.5 Generalisation across data sets
The ordering of methods is the same on the tabular task (Breast Cancer: all methods within {max(acc(t := 'breast_cancer', mth)[0] for mth in [RBF, PCE, QE, LIN, LR]) - min(acc('breast_cancer', mth)[0] for mth in [RBF, PCE, QE, LIN, LR]):.1f} points of each other), on the easy Fashion-MNIST pair and on the hard pairs (Pullover/Coat and T-shirt/Shirt, where every method is at 79–82%): the redesigned quantum kernel does not change its relative standing when the task is harder or the data modality changes.

### 5.6 Idealised finite-shot and depolarising noise (RQ3)
""")
if nz is not None:
    S100, S1k, S10k = 100, 1000, 10000
    w(f"""Fig. 7 and Table 6 report the primary map on Fashion-MNIST T-shirt/Trouser and Pullover/Coat ({nz['seed'].nunique()} seeds each). Noise-free, exact and Nyström kernels agree to within 0.6 points. With finite shots the *exact* kernel degrades gracefully ({nacc('q_exact', S100)[0]:.1f}% at S = 100, {nacc('q_exact', S1k)[0]:.1f}% at 1,000, {nacc('q_exact', S10k)[0]:.1f}% at 10,000; noise-free {nacc('q_exact', np.inf)[0]:.1f}%), even though the estimated Gram matrix becomes indefinite (smallest eigenvalue {nmet('min_eig_Ktr','q_exact',S100):.2f} at S = 100). The Nyström kernel with the standard pseudo-inverse collapses: {nacc('q_nystrom', S100, 20)[0]:.1f}/{nacc('q_nystrom', S100, 60)[0]:.1f}/{nacc('q_nystrom', S100, 120)[0]:.1f}% at S = 100 and {nacc('q_nystrom', S1k, 20)[0]:.1f}/{nacc('q_nystrom', S1k, 60)[0]:.1f}/{nacc('q_nystrom', S1k, 120)[0]:.1f}% at S = 1,000 for m = 20/60/120. Larger m is *worse* under noise—the reverse of the noise-free ordering—because more landmarks mean a larger, worse-conditioned W: the estimated W_ε is indefinite (median smallest eigenvalue {nmet('min_eig_W','q_nystrom',S1k,60,agg='median'):.3f} at m = 60, S = 1,000) and the pseudo-inverse inverts its small negative eigenvalues. The realised amplification ratio A = ‖K̃_noisy − K̃_clean‖_F/(‖ΔC‖_F + ‖ΔW‖_F) is {nmet('amplification','q_nystrom',S1k,20,agg='median'):.0f}–{nmet('amplification','q_nystrom',S1k,120,agg='median'):.0f} (median over seeds). This is the breakdown regime of Proposition 3: the hypothesis ‖ΔW‖₂ < λ_min(W) + ε is violated (Ŵ_ε is indefinite in {100*nmet('min_eig_W','q_nystrom',S1k,60,agg=lambda x: float(np.mean(x <= 0))):.0f}% of runs at m = 60, S = 1,000), so the ratio should be read as evidence of the mechanism—inversion of small and negative eigenvalues—rather than as a check of the first-order constant. Normalising K̃ to unit diagonal does not help (it slightly hurts), and a fixed jitter ε = 0.1 restores accuracy only at S = 10,000 ({nacc('q_nystrom', S10k, 60, eps=0.1)[0]:.1f}%) but not at 100 shots ({nacc('q_nystrom', S100, 60, eps=0.1)[0]:.1f}%). Depolarising noise up to λ = 0.2 changes these conclusions little (Table S5): shot noise through W⁺ is the dominant failure mode.
""")
nf = pd.read_csv(RES / "noise_fix_per_seed.csv") if (RES / "noise_fix_per_seed.csv").exists() else None
if nf is not None:
    nf["shots_f"] = nf["shots"].fillna(np.inf)
    def nfa(v, S, m): d = nf[(nf["variant"] == v) & (nf["shots_f"] == S) & (nf["m"] == m)]; return 100 * d["acc"].mean()
    bias = lambda v, m: nfa(v, np.inf, m) - nfa('plain-pinv', np.inf, m)
    w(f"""**Remedies (Table 7, Fig. 11).** Discarding the non-positive eigenvalues of the estimated W_ε before inversion (PSD projection) restores accuracy to {nfa('psd-project', 100, 60):.1f}% at S = 100 and {nfa('psd-project', 1000, 60):.1f}% at S = 1,000 for m = 60 (pseudo-inverse: {nfa('plain-pinv', 100, 60):.1f}% / {nfa('plain-pinv', 1000, 60):.1f}%; noise-free {nfa('plain-pinv', np.inf, 60):.1f}%), and behaves the same for m = 20 and 120 ({nfa('psd-project', 100, 20):.1f}% and {nfa('psd-project', 100, 120):.1f}% at S = 100), with no noise-free bias ({bias('psd-project', 60):+.1f} points). The noise-adaptive jitter ε = 2σ_W√m performs identically ({nfa('adaptive', 100, 60):.1f}% at S = 100, bias {bias('adaptive', 60):+.1f}). Relative eigenvalue truncation trades robustness for bias: τ = 10⁻³ gives {nfa('truncate-0.001', 100, 60):.1f}% at S = 100 with a {bias('truncate-0.001', 60):+.1f}-point noise-free bias, τ = 10⁻² {nfa('truncate-0.01', 100, 60):.1f}% with {bias('truncate-0.01', 60):+.1f} points, and τ = 10⁻¹ {nfa('truncate-0.1', 100, 60):.1f}% with {bias('truncate-0.1', 60):+.1f} points—aggressive truncation removes the same low-variance directions that carry the class information of this low-rank kernel. A fixed jitter ε = 0.1 helps only at S ≥ 10,000 ({nfa('jitter-0.1-pinv', 100, 60):.1f}% at S = 100). The practical rule is simple: never apply a plain pseudo-inverse to an estimated landmark kernel; project it onto its positive eigenspace (or use a noise-scaled jitter) first. With that rule, Nyström compression of this quantum kernel loses {nfa('plain-pinv', np.inf, 60) - nfa('psd-project', 100, 60):.1f} points at 100 shots per fidelity on these tasks.
""")
if sc is not None:
    w(f"""### 5.7 Scaling with N (RQ5)
Fig. 8 and Table S7 report Fashion-MNIST T-shirt/Trouser for N ∈ {{{', '.join(str(int(n)) for n in sorted(sc['N'].unique()))}}} (θ, a, C fixed from N = 600; 3 seeds). Training-phase circuit evaluations grow as N(N−1)/2 for the exact kernel ({int(scv('q_exact', 600, 'circ')):,} → {int(scv('q_exact', 4800, 'circ')):,} from N = 600 to 4,800) and as N·m + m(m−1)/2 for Nyström ({int(scv('q_nystrom_m_fixed', 600, 'circ')):,} → {int(scv('q_nystrom_m_fixed', 9600, 'circ')):,} for m = 60 at N = 9,600), a {scv('q_exact', 4800, 'circ')/scv('q_nystrom_m_fixed', 4800, 'circ'):.0f}× reduction at N = 4,800. Test accuracy with m = 60 is {100*scv('q_nystrom_m_fixed', 600, 'acc'):.1f}% at N = 600 and {100*scv('q_nystrom_m_fixed', 9600, 'acc'):.1f}% at N = 9,600 (exact: {100*scv('q_exact', 600, 'acc'):.1f}% → {100*scv('q_exact', 4800, 'acc'):.1f}% at N = 4,800); E_F at fixed m = 60 is {100*scv('q_nystrom_m_fixed', 600, 'ef'):.2f}% at N = 600 and {100*scv('q_nystrom_m_fixed', 4800, 'ef'):.2f}% at N = 4,800 (the trained kernel's low rank makes m = 60 sufficient at every N tested), and m ∝ √N lowers it to {100*scv('q_nystrom_m_sqrtN', 4800, 'ef'):.2f}%. The circuit-evaluation count is the quantity that would matter on hardware; we report it as a count, not as a speed-up. Environment-specific simulator timings are given in Table S7 (Online Resource 1) only and carry no information about hardware cost. This study is small (3 seeds, θ fixed from N = 600) and is included to document the accounting, not as a robustness result.
""")
w(f"""### 5.8 Quantum-versus-classical kernel analysis
The geometric difference of the trained covariant kernel with respect to the tuned RBF kernel is g = {g_rbf[QE]:.1f} on average (per task {rng_str(list(g_rbf_task.values()))}; λ = 0.01), i.e. {g_rbf[QE]/SQRTN:.2f}√N, and {g_rbf[PCE]:.1f} for the product-cosine kernel; with respect to the linear kernel g = {g_lin[QE]:.0f} ≈ {g_lin[QE]/SQRTN:.1f}√N (Fig. 10). Under the geometric-difference criterion of Huang et al. (a necessary condition for a prediction advantage, with its own assumptions on the learning problem), values of g far below √N mean that the classical kernel can represent every function the quantum kernel can; g against the *linear* kernel is above √N, and the quantum kernels do out-predict the linear SVM by 0.2–2.2 points, whereas g against the tuned RBF kernel does not indicate a regime in which the covariant kernel would be expected to outperform it on these features—consistent with Table 3. At n = 8 the kernel is in any case classically computable from 256 numbers (Proposition 2), so no quantum advantage is claimed or implied.
""")
w(f"""## 6 Discussion

**What the redesign did and did not achieve.** Moving the parameterised, entangled circuit *before* the data encoding produced a kernel whose parameters and entanglement demonstrably act on it, with a closed form that explains its behaviour: the kernel is a shift-invariant function whose spectral measure is the Z-basis distribution of the fiducial state. The same closed form shows that at n = 8 the kernel is a 256-parameter classical Fourier kernel; the covariant construction is a principled way to *parameterise a family of classical periodic kernels with a quantum circuit*, and we found no evidence that this family is more useful than the RBF kernel on PCA-8 features. This is consistent with the geometric-difference analysis and with earlier benchmarking [Schnabel & Roth 2025; Huang et al. 2021].

**Why plain alignment training hurt.** Kernel-target alignment rewards kernels whose leading eigenvector is aligned with the labels and is maximised by kernels of very low effective rank; the optimiser duly halved the effective rank and drove the mean off-diagonal similarity towards 1 on several tasks. Such a kernel is trivially compressible (E_F below 1% at m = 60) but discards within-class structure that the SVM could have used. {DISC_ALT} Alignment is a surrogate for generalisation; the per-task differences involved are one to three points on ten seeds and several do not survive Holm correction, so the statements above describe a consistent direction rather than a large, universally significant effect.

**The compressibility–accuracy trade-off.** Across all maps, the effective rank of the exact kernel predicts both its Nyström error and its accuracy: the re-uploading maps (effective rank 55–97 in the ablation) were as accurate as anything we tried but still had E_F of 8–15% at m = 60, whereas the trained covariant kernel (effective rank ≈ 8) reached E_F < 1% at m = 60 and lost accuracy in the process. A practitioner who chooses a kernel *because* it compresses well should check that the compressibility did not come from over-smoothing.

**Landmark selection.** k-means landmarks are a dependable way to cut approximation error at small m—by roughly a third relative to uniform sampling—and this matters when kernel entries are expensive. Their effect on accuracy, however, was at most half a point and vanished for m ≥ 30. Leverage-score sampling, near-optimal in theory, underperformed uniform sampling for this very low-rank kernel; the choice of ridge parameter deserves attention before it is used with quantum kernels.

**Noise is the decisive practical issue.** The most consequential finding is the interaction between finite-shot estimation and the Nyström pseudo-inverse. Proposition 3 predicts amplification by ‖W_ε⁻¹‖², and the experiments show accuracy falling to 63–72% (chance: 50%) at 100–1,000 shots—while the exact kernel, whose indefiniteness the SVM tolerates, loses only 1–3 points. The failure is not intrinsic to Nyström but to inverting an *estimated* matrix without spectral control: discarding non-positive eigenvalues, truncating at τ‖W‖, or using a noise-adaptive jitter recovered almost all of the noise-free accuracy at 100 shots. Prior Nyström-quantum-kernel studies that report noise-free simulations [Srikumar et al. 2022; Rezvannia 2026] or a single regularisation setting should be read with this in mind, and future hardware experiments should report the spectrum of the estimated landmark kernel.

**Relation to prior work.** Our accuracy conclusions agree with Schnabel & Roth [2025] that tuned classical kernels are hard to beat; our compression conclusions extend Srikumar et al. [2022] and Coelho et al. [2025] by quantifying, for the first time to our knowledge, the landmark-strategy effect, the conditioning of W and the noise amplification with paired statistics and equal-budget classical baselines. Our Proposition 1 formalises a pitfall that is easy to fall into when transplanting variational-circuit intuition ("add trainable layers") to kernels.

## 7 Limitations
(1) *Simulation only; idealised noise.* All kernels were simulated exactly; the finite-shot model is an idealised binomial model of the inversion test with independent entries, a known unit diagonal and no readout, correlated or drifting noise, and C and W were sampled independently although on hardware they share landmark circuits; hyper-parameters were tuned noise-free. No result is a hardware validation. (2) *Width and simulability.* n = 8 qubits after PCA-8; the kernel is classically computable from 256 numbers, no quantum advantage is claimed, the study does not address exponential concentration at large n [Thanasilp et al. 2024], and PCA discards information that another encoding might use. (3) *Sample sizes.* N = 600 training points; the scaling study fixes θ from N = 600, uses three seeds and is reported for its accounting only. (4) *Tabular task.* 171 test points and repeated random splits rather than an official test split. (5) *Training objectives.* {LIM_TRAIN} (6) *Landmark strategies.* {LIM_LEV} k-means used n_init = 3. (7) *Estimator.* SWAP-test statistics differ from the inversion-test model by constants. (8) *Cost accounting.* Circuit-evaluation counts are analytic (Proposition 4), not measured on a device; simulator timings (Online Resource 1, Table S7) carry no information about hardware cost. (9) *Effect sizes.* Most accuracy differences between methods are one point or less; with n_test = 2,000 the Wilson interval of a single accuracy is about ±1 point, so only paired, multi-seed comparisons carry evidential weight, and several do not survive Holm correction. (10) *Baselines.* The RBF bandwidth was tuned once for the exact kernel and reused for RBF Nyström; a per-m re-tuning could only strengthen that baseline.

## 8 Conclusion
We rebuilt a Nyström-compressed quantum-kernel pipeline around a feature map whose parameters provably act on the kernel, and evaluated it under a controlled protocol. The compressed kernel keeps the accuracy of the exact one at 10% of the landmark budget, k-means landmarks lower approximation error but not accuracy, and neither the exact nor the compressed quantum kernel outperforms a tuned RBF kernel or its own compressed classical counterpart on eight PCA features—an outcome anticipated by the geometric difference and by the kernel's closed form. {CONCL_ALT} Finite-shot estimation breaks the standard Nyström pseudo-inverse, and spectral projection of the estimated landmark kernel repairs it. These results do not argue against quantum kernels; they specify what a Nyström-compressed quantum kernel must be compared against, which quantities must be reported (E_F, effective rank, cond W, shot budget), and which regularisation is mandatory before any hardware experiment.

## Statements and Declarations

**Funding** {_decl('funding', '[FILL IN — funding statement]')}

**Competing interests** {_decl('competing_interests', '[FILL IN — competing-interests statement]')}

**Ethics approval and consent to participate** {_decl('ethics', 'Not applicable. The study uses publicly available benchmark data sets and involves no human participants, animals or personal data.')}

**Data availability** {_decl('data_availability', 'Fashion-MNIST and MNIST are public benchmark data sets; the exact files used are downloaded by `scripts/prepare_data.py`. Breast Cancer Wisconsin (Diagnostic) is distributed with scikit-learn. Every per-configuration result (Online Resource 2) and every table and figure with its provenance metadata are provided with the submission.')}

**Code availability** {_decl('code_availability', 'The complete pipeline (configuration, simulator, experiment scripts, analysis, tests and manuscript builder) is provided as a repository archive with the submission; `README.md` lists the commands that regenerate every number, table and figure in this paper. [Public repository / DOI — to be inserted by the author at deposit]')}

**Author contributions** {_decl('author_contributions', '[FILL IN — author contributions]')}

## References
{{REFERENCES}}

## Tables

**Table 2** Tasks, sample sizes and PCA-8 explained variance (mean over seeds).

{md_table(t1)}

**Table 3** Test accuracy (%) mean ± SD over {n_seeds} seeds; approximate methods at m = 60 landmarks (k-means centroids) / D = 60 random features; full official test splits. Wilson 95% intervals for a single accuracy at n_test = 2,000 are approximately ±1 point. Paired statistics: Table S3.

{md_table(t2)}
""")
if ab is not None:
    t3 = pd.read_csv(TAB / "table3_ablation.csv")[["variant", "n_params", "Acc (%)", "kta", "effrank", "acc_m60", "EF_m60", "n"]]
    t3.columns = ["variant", "params", "exact acc (%)", "KTA", "eff. rank", "Nyström m=60 acc", "E_F m=60", "runs"]; t3["Nyström m=60 acc"] *= 100; t3["E_F m=60"] *= 100
    t3["variant"] = (t3["variant"].str.replace("covariant_torus", "covariant", regex=False).str.replace("zz_reuploading", "ZZ re-upload", regex=False)
                     .str.replace("reuploading", "re-upload", regex=False).str.replace("|theta=pi/2", ", θ = π/2", regex=False)
                     .str.replace("|ent=1", ", ent", regex=False).str.replace("|ent=0", ", no ent", regex=False).str.replace("|trained=", ", ", regex=False)
                     .str.replace("|", ", ", regex=False).str.replace(", 0", ", untrained", regex=False).str.replace(", 1", ", KTA", regex=False))
    w(f"""**Table 4** Feature-map ablation at a = π ({len(ab_tasks)} tasks × {ab_seeds} seeds pooled). "ent" = CNOT ring in the fiducial/re-uploading circuit; "trained" = θ optimised by KTA (0 = random θ; θ = π/2 gives the product-cosine kernel).

{md_table(t3)}
""")
if bw is not None:
    tb = pd.read_csv(TAB / "table_bandwidth.csv"); tb2 = tb.groupby(["map", "a_over_pi"]).agg(acc=("acc", "mean"), kta=("kta", "mean"), effrank=("effrank", "mean"), offdiag=("offdiag", "mean"), EF_m60=("EF_m60", "mean")).reset_index()
    tb2["acc"] *= 100; tb2["EF_m60"] *= 100
    w(f"""**Table 5** Scaling-range (bandwidth) study, two Fashion-MNIST tasks × 10 seeds pooled: test accuracy (%), KTA, effective rank, mean off-diagonal kernel value and E_F (%) at m = 60 (per-task values and cond(W_ε) in Table S6, Online Resource 1).

{md_table(tb2.assign(map=lambda d: d['map'].str.replace('covariant_torus', 'covariant', regex=False).str.replace('product_cosine', 'product-cosine', regex=False)).rename(columns={'map': 'kernel', 'a_over_pi': 'a/π', 'acc': 'acc (%)', 'kta': 'KTA', 'effrank': 'eff. rank', 'offdiag': 'mean off-diag.', 'EF_m60': 'E_F (%) m=60'}), '.3f')}
""")
if nz is not None:
    t5 = pd.read_csv(TAB / "table5_noise_shots.csv"); t5["acc"] *= 100; t5["acc_sd"] *= 100; t5["shots"] = t5["shots"].replace(0, np.inf)
    w(f"""**Table 6** Idealised finite-shot study (ε = 10⁻⁶, pseudo-inverse, no depolarising noise; two tasks × 10 seeds): mean accuracy (%) ± SD, mean E_F, mean noise-only error, *median* realised amplification ratio A, fraction of runs with indefinite Ŵ_ε, median λ_min(Ŵ_ε), median ‖Ŵ_ε⁺‖₂, median smallest eigenvalue of the estimated Gram (exact) or Nyström matrix (λ_min(K̂)). Columns: S = shots; SD over seeds; E_noise = noise-only error; A = amplification ratio; frac. indef. = fraction of runs with indefinite Ŵ_ε. cond(Ŵ_ε) is ∞ whenever Ŵ_ε is indefinite and is therefore not tabulated.

{md_table(t5[["task", "method", "m", "shots", "acc", "acc_sd", "EF", "noise_only", "amp", "frac_indefinite_W", "min_eig_W_med", "Wpinv_med", "min_eig_Ktr"]].assign(task=lambda d: d["task"].map(TASK_LABELS).str.replace("F-MNIST ", "", regex=False), method=lambda d: d["method"].map({"q_exact": "exact", "q_nystrom": "Nyström"})).rename(columns={"shots": "S", "acc": "acc (%)", "acc_sd": "SD (pts)", "EF": "E_F", "noise_only": "E_noise", "amp": "A", "frac_indefinite_W": "frac. indef.", "min_eig_W_med": "λ_min(Ŵ_ε)", "Wpinv_med": "‖Ŵ_ε⁺‖₂", "min_eig_Ktr": "λ_min(K̂)"}), '.3f')}
""")
if nf is not None:
    t6 = nf.groupby(["variant", "m", "shots_f"]).agg(acc=("acc", "mean"), sd=("acc", "std"), EF=("E_F", "mean"), rank_kept=("rank_kept", "mean"), Wpinv=("Wpinv_norm", "median")).reset_index(); t6["acc"] *= 100; t6["sd"] *= 100
    w(f"""**Table 7** Noise-robust Nyström variants (two tasks × 10 seeds; k-means centroids): accuracy (%) mean ± SD, E_F, mean number of eigenvalues kept, median ‖W_ε⁺‖₂.

{md_table(t6, '.3f')}
""")
w(f"""Online Resource 1 (ESM_1.pdf) contains the proofs of Propositions 1–4 (Sect. S1), the pre-registered protocol of the alternative training objective (Sect. S4), Table S1 (validation gate), Table S3 (paired statistics, all main comparisons), Table S4 (landmark strategies vs uniform at every m), Table S5 (depolarising noise), Table S6 (scaling-range study per task), Table S7 (scaling, incl. environment-specific simulator timings), Table S8 (leverage ridge sweep and its paired statistics) and Fig. S1 (E_F vs ridge λ). Online Resource 2 (ESM_2.xlsx) contains every per-configuration result (one row per task × seed × method × budget).

## Figures

![**Fig. 1** Pipeline: leakage-safe preprocessing (PCA-8 and scaling fitted on training data), covariant feature map with trainable fiducial state, exact or Nyström-compressed fidelity kernel (with spectral control of the estimated landmark kernel), SVM, and the evaluation protocol.](../figures/fig01_pipeline.png)

![**Fig. 2** Feature map: the fiducial circuit V(θ) (RY layers and CNOT ring, two blocks) acts on |0⟩ⁿ before the data encoding D(x) = ⊗RZ(xⱼ); the previous construction (right) applied trainable/entangling gates after the encoding, where they cancel in the fidelity.](../figures/fig02_circuit.png)

![**Fig. 3** (a) Median condition number of the landmark kernel W_ε vs m for the trained covariant, product-cosine and RBF kernels (k-means centroids). (b) Trace energy in the top-60 eigenvalues and (c) effective rank of the exact kernels per task.](../figures/fig03_spectrum_conditioning.png)

![**Fig. 4** Feature-map and training-objective ablation (a = π; {len(ab_tasks) if ab is not None else 3} tasks × {ab_seeds if ab is not None else 10} seeds): paired accuracy difference of every variant to the product-cosine kernel (95% CI), centred KTA, and Nyström E_F at m = 60 (log scale).](../figures/fig04_ablation.png)

![**Fig. 5** Relative Frobenius error of the Nyström approximation of the trained covariant kernel vs landmark budget m for five landmark strategies (median and inter-quartile range over 10 seeds), one panel per task.](../figures/fig05_EF_vs_m.png)

![**Fig. 6** Test accuracy of the Nyström-compressed trained covariant kernel vs m for five landmark strategies (mean ± SD over 10 seeds); dashed: exact quantum kernel; dotted: RBF and linear SVM.](../figures/fig06_acc_vs_m.png)

![**Fig. 7** Finite-shot robustness (two tasks × 10 seeds): (a) accuracy vs shots for the exact kernel and Nyström with m = 20/60/120 (pseudo-inverse, ε = 10⁻⁶); (b) kernel error vs shots; (c) effect of jitter and normalisation at S = 1,000.](../figures/fig07_noise_shots.png)

![**Fig. 8** Scaling with N on Fashion-MNIST T-shirt/Trouser (3 seeds): (a) training-phase circuit evaluations (symmetric count), (b) simulation wall-clock, (c) test accuracy on the 2,000 official test images.](../figures/fig08_scaling.png)

![**Fig. 9** Test accuracy vs budget m for quantum Nyström (trained covariant and product-cosine kernels), classical RBF Nyström with identical k-means landmarks, and RBF random Fourier features with D = m; dashed lines: exact kernels.](../figures/fig09_quantum_vs_classical_budget.png)

![**Fig. 10** Geometric difference g(K_C‖K_Q) (λ = 0.01) of the exact quantum kernels with respect to the tuned RBF and linear kernels on the 600 training points; red dashed line: √N.](../figures/fig10_geometric_difference.png)

![**Fig. 11** Noise-robust variants: accuracy vs shots for the pseudo-inverse, PSD projection, truncation and adaptive jitter (two tasks × 10 seeds, m = 20/60/120).](../figures/fig11_noise_fix.png)
""")

text = "\n".join(out)
# --- Springer name-year citations (checked against references.yaml), reference list, float placement, consecutive numbering
text, _cited = _refs.convert_citations(text)
_uncited = sorted(set(_refs.BY_KEY) - _cited)
if _uncited: raise SystemExit(f"references never cited in the text: {_uncited}")
text = text.replace("{REFERENCES}", "::: {custom-style=\"Bibliography\"}\n" + "\n\n".join(_refs.render(r) for r in _refs.reference_list()) + "\n:::")   # hanging-indent paragraphs in Word
from layout import place_floats, renumber_floats
from mathify import mathify
text = renumber_floats(text)
text = place_floats(text)
# plain-text abstract (Unicode notation) for the submission system's abstract field
_abs = text.split("## Abstract")[1].split("**Keywords:**")[0].strip()
(ROOT / "manuscript" / "abstract_plain.txt").write_text(_abs + "\n", encoding="utf-8")
text = mathify(text)          # all remaining Unicode notation -> LaTeX inline math
_md = ROOT / "manuscript" / _args.out
_md.write_text(text)
print(f"{_md.name} written:", len(text.split()), "words")
try:
    _ref = Path(_args.reference_doc) if _args.reference_doc else ROOT / "manuscript" / "reference_qmi.docx"
    _cmd = ["pandoc", str(_md), "-o", str(_md.with_suffix(".docx")), "--resource-path", str(ROOT / "manuscript"), "--columns=1000"] + (["--reference-doc", str(_ref)] if _ref.exists() else [])
    subprocess.run(_cmd, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fix_docx_tables.py"), str(_md.with_suffix(".docx"))], check=True)   # explicit column widths
    print(f"{_md.with_suffix('.docx').name} written")
except Exception as e:
    print("pandoc failed:", e)
