"""Build the Springer 'Electronic Supplementary Material' for the QMI submission from the same sources as the manuscript.

  manuscript/supplement.md  ->  ESM_1.pdf  (Online Resource 1: proofs S1, supplementary tables, figures, pre-registered protocol,
                                             reproducibility)          [built by scripts/build_submission.py via pandoc + xelatex]
  ESM_2.xlsx                                (Online Resource 2: every per-configuration result, one sheet per experiment)

Every table is generated from tables/*.csv (no hand-typed numbers). Usage: python scripts/build_supplement.py [out_dir]
"""
import re, sys, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT = Path(__file__).resolve().parents[1]; TAB, RES, FIG, DOCS = ROOT / "tables", ROOT / "results", ROOT / "figures", ROOT / "docs"
sys.path.insert(0, str(ROOT / "manuscript"))
from tablefmt import md_table  # noqa: E402

TITLE = "How Much of a Trainable Quantum Kernel Survives Nyström Compression? A Controlled Study of Landmark Selection, Finite-Shot Noise and Classical Baselines"


def author_block():
    a = yaml.safe_load((ROOT / "manuscript" / "authors.yaml").read_text(encoding="utf-8"))
    return ", ".join(x["name"] for x in a["authors"]) + "  \n" + "  \n".join(x["affiliation"] for x in a["affiliations"]) + f"  \nCorresponding author: {a['corresponding']['email']}"


def shift_headings(md: str, by: int) -> str:
    return re.sub(r"^(#+) ", lambda m: "#" * (len(m.group(1)) + by) + " ", md, flags=re.M)


def read_doc(name):
    t = (DOCS / name).read_text(encoding="utf-8")
    t = re.sub(r"^# .*\n", "", t, count=1)  # drop the document title (we supply our own heading)
    return shift_headings(t, 1)


def main(out_dir=None):
    out = []; w = out.append
    w(f'---\ntitle: "Online Resource 1 — Supplementary Information"\n---\n\n**Supplementary Information for:** {TITLE}\n\n{author_block()}\n')
    w("**Contents.** Sect. S1 proofs of Propositions 1–4; Sect. S2 supplementary tables (Table S1, S3–S8; Table S2 is the per-configuration "
      "spreadsheet, Online Resource 2); Sect. S3 supplementary figures (Fig. S1, S2); Sect. S4 pre-registered protocol of the alternative "
      "training objective; Sect. S5 reproducibility and provenance.\n")
    # ---- S1 proofs
    w("## S1 Theoretical results with proofs\n"); w(read_doc("theory.md"))
    # ---- S2 tables
    w("\n## S2 Supplementary tables\n")
    tv = pd.read_csv(TAB / "table_validation_maps.csv")
    w("**Table S1** Phase-D validation gate of the candidate feature maps (7 tasks × 3 seeds): validation accuracy, centred KTA before and after training, "
      "maximum kernel change under a θ perturbation (dK_θ) and under removal of the entangling ring (dK_ent), effective rank, minimum eigenvalue, "
      "distance to the RBF and product-cosine kernels, geometric difference versus RBF, runtime per kernel (s).\n")
    w(md_table(tv.rename(columns={"map": "feature map", "val_acc": "val. acc", "kta_init": "KTA (init)", "kta_trained": "KTA (trained)", "dK_theta": "dK_θ",
                                  "dK_ent": "dK_ent", "eff_rank": "eff. rank", "min_eig": "λ_min", "dist_rbf": "dist. RBF", "dist_pc": "dist. product-cosine", "g_rbf": "g vs RBF"}), ".3f") + "\n")
    w("**Table S2** — the per-configuration results (one row per task × seed × method × budget) are provided as Online Resource 2 (ESM_2.xlsx).\n")
    st = pd.read_csv(TAB / "table_stats_main.csv")
    w("**Table S3** Paired comparisons across seeds (n = 10) for every main comparison and task at m = 60 (k-means centroids): mean accuracy difference "
      "(points), bootstrap 95% CI, Cohen's d_z, paired t-test and Wilcoxon p-values, Holm-adjusted p within the family of all tests in this table.\n")
    w(md_table(st, ".3f") + "\n")
    ls = pd.read_csv(TAB / "table_landmark_strategies.csv")
    w("**Table S4** Landmark strategies versus uniform sampling at every budget m (paired over task × seed): mean difference in E_F and in accuracy, "
      "95% CI, p-values and Holm-adjusted p.\n")
    w(md_table(ls.rename(columns={"mean_diff": "Δ", "ci_lo": "CI lo", "ci_hi": "CI hi", "p_t": "p (t)", "p_wilcoxon": "p (Wilcoxon)", "cohen_dz": "d_z", "p_t_holm": "p Holm"}), ".4f") + "\n")
    dp = pd.read_csv(TAB / "table5_noise_depolarizing.csv"); dp["acc"] *= 100
    w("**Table S5** Depolarising noise × shots (ε = 10⁻⁶, unnormalised; two tasks × 10 seeds): mean accuracy (%) and mean E_F.\n")
    w(md_table(dp.assign(method=lambda d: d["method"].map({"q_exact": "exact", "q_nystrom": "Nyström"}).fillna(d["method"])).rename(columns={"shots": "S", "depolarizing": "λ (depol.)", "acc": "acc (%)", "EF": "E_F"}), ".3f") + "\n")
    bw = pd.read_csv(TAB / "table_bandwidth.csv"); bw["acc"] *= 100; bw["acc_sd"] *= 100; bw["EF_m60"] *= 100; bw["acc_m60"] *= 100
    w("**Table S6** Scaling-range (bandwidth) study per task and kernel (10 seeds): test accuracy (%) and SD, centred KTA, effective rank, mean off-diagonal "
      "kernel value, E_F (%) and accuracy (%) at m = 60, median cond(W_ε) (∞ when W_ε is singular or indefinite).\n")
    w(md_table(bw.assign(task=lambda d: d["task"].map({"fashion_0v1": "T-shirt/Trouser", "fashion_2v4": "Pullover/Coat"}).fillna(d["task"]),
                         map=lambda d: d["map"].str.replace("covariant_torus", "covariant", regex=False).str.replace("product_cosine", "product-cosine", regex=False))
               .rename(columns={"map": "kernel", "a_over_pi": "a/π", "acc": "acc (%)", "acc_sd": "SD", "kta": "KTA", "effrank": "eff. rank", "offdiag": "mean off-diag.",
                                "EF_m60": "E_F (%) m=60", "acc_m60": "acc (%) m=60", "condW": "cond(W_ε)"}), ".3f") + "\n")
    sc = pd.read_csv(TAB / "table_scaling.csv"); sc["acc"] *= 100; sc["acc_sd"] *= 100
    w("**Table S7** Scaling study (Fashion-MNIST T-shirt/Trouser, 3 seeds, θ fixed from N = 600): accuracy (%) on the 2,000 official test images, "
      "simulator wall-clock (s; environment-specific, no information about hardware cost), memory (MB), training and test circuit-evaluation counts "
      "(Proposition 4), and E_F.\n")
    w(md_table(sc.assign(method=lambda d: d["method"].map({"q_exact": "exact", "q_nystrom_m_fixed": "Nyström m = 60", "q_nystrom_m_sqrtN": "Nyström m ∝ √N"}).fillna(d["method"]))
               .rename(columns={"acc": "acc (%)", "acc_sd": "SD", "t_total": "t total (s)", "t_kernel": "t kernel (s)", "t_svm": "t SVM (s)", "mem_MB": "Gram (MB)", "peak_MB": "peak (MB)",
                                "circ_train": "circuits (train)", "circ_test": "circuits (test)", "EF": "E_F"}), ".3f") + "\n")
    lv = pd.read_csv(TAB / "table_leverage_sweep.csv"); lvs = pd.read_csv(TAB / "table_leverage_sweep_stats.csv")
    w("**Table S8** Ridge-leverage-score sampling versus the ridge parameter λ (absolute scale, N = 600) for m ∈ {10, 20, 60}, with uniform and k-means-centroid "
      "references (7 tasks × 10 seeds): mean and median E_F, accuracy and SD, entropy of the sampling distribution (uniform = ln 600 = 6.40), number of runs; "
      "and (lower part) paired E_F difference leverage − uniform with 95% CI, p-values and Holm-adjusted p.\n")
    w(md_table(lv.rename(columns={"ridge": "λ", "EF_mean": "E_F mean", "EF_med": "E_F median", "acc": "acc", "acc_sd": "SD"}), ".4f") + "\n")
    w(md_table(lvs.rename(columns={"ridge": "λ", "EF_diff": "Δ E_F", "ci_lo": "CI lo", "ci_hi": "CI hi", "p_t": "p (t)", "p_wilcoxon": "p (Wilcoxon)", "p_t_holm": "p Holm"}), ".4f") + "\n")
    # ---- S3 figures
    w("\n## S3 Supplementary figures\n")
    w("![**Fig. S1** Ridge-leverage-score landmark sampling: E_F (median and inter-quartile range over 7 tasks × 10 seeds) versus the ridge parameter λ for m = 10, 20, 60, "
      "with the uniform and k-means-centroid references (dashed lines: medians; shaded: inter-quartile ranges).](../figures/figS_leverage_sweep.png)\n")
    w("![**Fig. S2** Scaling-range (bandwidth) study on two Fashion-MNIST tasks (10 seeds): test accuracy, centred KTA, effective rank and E_F at m = 60 versus a/π for the "
      "trained covariant and the product-cosine kernel.](../figures/fig_bandwidth.png)\n")
    # ---- S4 protocol
    w("\n## S4 Pre-registered protocol for the alternative training objective\n"); w(read_doc("alternative_training_protocol.md"))
    # ---- S5 reproducibility
    cfg = json.load(open(RES / "main_per_seed.meta.json"))["config"]
    w("\n## S5 Reproducibility and provenance\n")
    w(f"Software: Python {cfg['software']['python']}, NumPy {cfg['software']['numpy']}, SciPy {cfg['software']['scipy']}, scikit-learn {cfg['software']['scikit_learn']}, "
      f"PennyLane {cfg['software']['pennylane']} (simulator cross-check only). Hardware: {cfg['hardware']['cpu']}, {cfg['hardware']['ram_gb']} GB RAM. "
      "All experiments run with `OMP_NUM_THREADS=1`; the configuration is `configs/main.yaml` in the code repository.\n")
    w("Every table and figure in the manuscript and in this document is generated from the per-configuration result files by `analysis/make_tables.py`, "
      "`analysis/make_figures.py`, `analysis/make_figures_extra.py` and `manuscript/build_manuscript.py`; each figure carries a provenance file "
      "(`figures/*_metadata.json`) recording the source files and their SHA-256 hashes. The command sequence that regenerates everything is listed in the "
      "repository `README.md`. Trained-kernel rows reproduce exactly within one software environment and to within 0.5 accuracy points across environments "
      "(floating-point differences in L-BFGS-B); untrained rows reproduce exactly.\n")
    md = "\n".join(out)
    # (the supplement keeps Unicode notation: its proofs use constructs the automatic converter does not cover reliably)
    dst = Path(out_dir) if out_dir else ROOT / "manuscript"
    (dst / "supplement.md").write_text(md, encoding="utf-8")
    print("supplement.md written:", len(md.split()), "words")
    # ---- Online Resource 2: spreadsheet of per-seed results
    xlsx = dst / "ESM_2.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        readme = pd.DataFrame({"sheet": [], "source": [], "rows": [], "description": []})
        rows = []
        for name, desc in [("main_per_seed", "Main study: every task × seed × method × landmark strategy × budget m"), ("ablation_per_seed", "Feature-map / training-objective ablation (3 tasks × 10 seeds × 16 variants)"),
                           ("bandwidth_per_seed", "Scaling-range (bandwidth) study"), ("noise_per_seed", "Idealised finite-shot and depolarising noise study"),
                           ("noise_fix_per_seed", "Noise-robust Nyström variants"), ("scaling_per_seed", "Scaling with N"), ("leverage_sweep_per_seed", "Ridge-leverage sweep"),
                           ("validation_feature_maps", "Phase-D validation of candidate feature maps")]:
            p = RES / f"{name}.csv"
            if not p.exists(): continue
            df = pd.read_csv(p); sheet = name[:31]
            df.to_excel(xw, sheet_name=sheet, index=False); rows.append((sheet, f"results/{name}.csv", len(df), desc))
        pd.DataFrame(rows, columns=["sheet", "source", "rows", "description"]).to_excel(xw, sheet_name="README", index=False)
    print("ESM_2.xlsx written:", xlsx)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
