import json, hashlib, sys, os, subprocess, datetime
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG, TAB = ROOT / "results", ROOT / "figures", ROOT / "tables"
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))

# Journal artwork conventions (Springer/QMI): sans-serif lettering (Helvetica/Arial-compatible), 8–9 pt at final size, figures at
# most 174 mm wide (double-column text area), TrueType (Type 42) fonts embedded in vector output, 600 dpi TIFF for combination art.
FIG_W_MAX = 174 / 25.4   # inches
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"],   # Arial-metric sans serif; DejaVu Sans only as glyph fallback (⟨⟩⊗)
                     "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans"})
_orig_subplots = plt.subplots
def _journal_subplots(*a, figsize=None, **k):
    """plt.subplots with the width clamped to 174 mm; heights are scaled with a 1.15 factor so 8-pt lettering keeps its room."""
    if figsize is not None:
        w, h = figsize
        if w > FIG_W_MAX:
            s = FIG_W_MAX / w; figsize = (FIG_W_MAX, h * s * 1.15)
    return _orig_subplots(*a, figsize=figsize, **k)
plt.subplots = _journal_subplots
COLORS = {"q_nystrom[covariant_torus]": "#1b6ca8", "q_nystrom[product_cosine]": "#7f7f7f", "rbf_nystrom": "#d95f02",
          "rff_rbf": "#e6ab02", "q_exact[covariant_torus]": "#08306b", "q_exact[product_cosine]": "#4d4d4d",
          "svm_rbf": "#a6611a", "svm_linear": "#1a9850", "logreg": "#66c2a5"}
LABELS = {"q_nystrom[covariant_torus]": "Quantum Nyström (trained covariant kernel)", "q_nystrom[product_cosine]": "Quantum Nyström (product-cosine kernel)",
          "rbf_nystrom": "Classical RBF Nyström (same landmarks)", "rff_rbf": "Random Fourier features (RBF)",
          "q_exact[covariant_torus]": "Exact quantum kernel (covariant)", "q_exact[product_cosine]": "Exact quantum kernel (product-cosine)",
          "svm_rbf": "RBF SVM", "svm_linear": "Linear SVM", "logreg": "Logistic regression"}
STRAT_LABELS = {"uniform": "uniform", "kmeanspp": "k-means++ (points)", "kmeans_centroids": "k-means centroids",
                "kmeans_nearest": "k-means nearest points", "leverage": "ridge leverage (oracle)"}
# short forms for panel titles / tick labels at journal figure width (the caption names the data set)
SHORT = {"q_nystrom[covariant_torus]": "Q-Nyström (trained covariant)", "q_nystrom[product_cosine]": "Q-Nyström (product-cosine)",
         "rbf_nystrom": "RBF Nyström (same landmarks)", "rff_rbf": "Random Fourier features", "q_exact[covariant_torus]": "exact Q kernel (covariant)",
         "q_exact[product_cosine]": "exact Q kernel (product-cosine)", "svm_rbf": "RBF SVM", "svm_linear": "linear SVM", "logreg": "logistic regression"}
TASK_SHORT = {"fashion_0v1": "T-shirt/Trouser", "fashion_2v4": "Pullover/Coat", "fashion_0v6": "T-shirt/Shirt", "fashion_5v7": "Sandal/Sneaker",
              "mnist_3v8": "MNIST 3/8", "mnist_4v9": "MNIST 4/9", "breast_cancer": "Breast Cancer"}
TASK_LABELS = {"fashion_0v1": "F-MNIST T-shirt/Trouser", "fashion_2v4": "F-MNIST Pullover/Coat", "fashion_0v6": "F-MNIST T-shirt/Shirt",
               "fashion_5v7": "F-MNIST Sandal/Sneaker", "mnist_3v8": "MNIST 3/8", "mnist_4v9": "MNIST 4/9", "breast_cancer": "Breast Cancer (tabular)"}


def load(name):
    return pd.read_csv(RES / name)


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def save_fig(fig, name, sources, script, description, extra=None):
    """Save figure as PDF (vector) + PNG (300 dpi) + TIFF (600 dpi, LZW; journal 'combination art') and write provenance metadata JSON."""
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / f"{name}.tif", bbox_inches="tight", pad_inches=0.03, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    meta = dict(figure=name, description=description, sources=[str(Path(s).relative_to(ROOT)) if str(s).startswith(str(ROOT)) else str(s) for s in sources],
                source_sha256={Path(s).name: hashlib.sha256(Path(s).read_bytes()).hexdigest()[:16] for s in sources if Path(s).exists()},
                script=script, generated=datetime.datetime.utcnow().isoformat() + "Z", code_version=_git_rev(),
                config="configs/main.yaml")
    if extra: meta.update(extra)
    with open(FIG / f"{name}_metadata.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    plt.close(fig)


def save_table(df, name, caption, sources, script, floatfmt=".2f", index=False):
    """Write CSV + Markdown + LaTeX with a provenance header comment."""
    df.to_csv(TAB / f"{name}.csv", index=index)
    md = df.to_markdown(index=index, floatfmt=floatfmt)
    prov = f"<!-- generated by {script} from {', '.join(Path(s).name for s in sources)} on {datetime.datetime.utcnow().date()} -->\n"
    (TAB / f"{name}.md").write_text(prov + f"**{caption}**\n\n" + md + "\n")
    tex = df.to_latex(index=index, float_format=lambda x: f"{x:{floatfmt}}", caption=caption, label=f"tab:{name}", escape=True)
    (TAB / f"{name}.tex").write_text(f"% generated by {script} from {', '.join(Path(s).name for s in sources)}\n" + tex)
    return md


def fmt_pm(m, s, nd=2):
    return f"{m:.{nd}f} ± {s:.{nd}f}"
