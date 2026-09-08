"""Figures that are not data plots: pipeline schematic (Fig. 1), circuit diagrams (Fig. 2), plus the noise-fix figure (Fig. 11)."""
from common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
SCRIPT = "analysis/make_figures_extra.py"

# ---------------------------------------------------------------- Fig 1: pipeline schematic (174 mm x 60 mm; 8-pt lettering)
fig, ax = plt.subplots(figsize=(FIG_W_MAX, 2.8)); ax.axis("off"); fig.subplots_adjust(left=0.006, right=0.994, top=1, bottom=0)
BW = 0.18
boxes = [("Official train split\nstratified sub-sample\nN = 600", 0.01, 0.62), ("PCA-8 + min–max\nfitted on TRAIN only\nangles in [0, a]", 0.205, 0.62),
         ("Fiducial state V(θ)|0⟩\ntrained by KTA (fit split)\nthen D(x) = ⊗RZ(xⱼ)", 0.4, 0.62), ("Exact kernel\nK = |⟨φ(x)|φ(x′)⟩|²\nN(N−1)/2 evaluations", 0.595, 0.80),
         ("Nyström: landmarks Z\n(5 strategies)\nK̃ = C·W⁺·Cᵀ\n(spectral control of W)", 0.595, 0.40), ("SVM (C on validation)\nfull official test split\n10 seeds, paired tests", 0.79, 0.62)]
for i, (txt, x, y) in enumerate(boxes):
    col = "#dbe9f6" if i in (2, 3, 4) else "#eeeeee"
    hh = 0.16 if i in (3, 4) else 0.15
    ax.add_patch(FancyBboxPatch((x, y - hh), BW, 2 * hh, boxstyle="round,pad=0.01", fc=col, ec="#333333", lw=0.8, transform=ax.transAxes))
    ax.text(x + BW / 2, y, txt, ha="center", va="center", fontsize=7.3, linespacing=1.25, transform=ax.transAxes)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9, lw=0.8, color="#333333", shrinkA=0, shrinkB=0))
arrow(0.19, 0.62, 0.205, 0.62); arrow(0.385, 0.62, 0.4, 0.62); arrow(0.58, 0.66, 0.595, 0.72); arrow(0.58, 0.58, 0.595, 0.50)
arrow(0.775, 0.72, 0.79, 0.66); arrow(0.775, 0.50, 0.79, 0.58)
ax.text(0.685, 0.60, "same θ, a, C", ha="center", va="center", fontsize=7.5, style="italic", transform=ax.transAxes)
ax.text(0.5, 0.0, "Baselines on the same features and splits: linear SVM · RBF SVM · logistic regression ·\nRBF Nyström (same landmarks) · random Fourier features (D = m).\n"
        "Noise model: each estimated fidelity → Binomial(S, K)/S; depolarising K → (1 − λ)K + λ/2ⁿ.",
        ha="center", va="bottom", fontsize=7.5, linespacing=1.3, transform=ax.transAxes)
save_fig(fig, "fig01_pipeline", [ROOT / "configs" / "main.yaml"], SCRIPT, "Schematic of the leakage-safe pipeline and evaluation protocol.")

# ---------------------------------------------------------------- Fig 2: circuits
import pennylane as qml
n = 4  # drawn with 4 qubits for legibility; the experiments use 8
dev = qml.device("default.qubit", wires=n)
@qml.qnode(dev)
def covariant(x, th):
    for l in range(2):
        for q in range(n): qml.RY(th[l * n + q], wires=q)
        for q in range(n - 1): qml.CNOT(wires=[q, q + 1])
        qml.CNOT(wires=[n - 1, 0])
    qml.Barrier(wires=range(n))
    for q in range(n): qml.RZ(x[q], wires=q)
    return qml.state()
@qml.qnode(dev)
def previous(x, th):
    for q in range(n): qml.Hadamard(wires=q)
    for q in range(n): qml.RZ(x[q], wires=q)
    qml.Barrier(wires=range(n))
    for q in range(n - 1): qml.CNOT(wires=[q, q + 1])
    qml.CNOT(wires=[n - 1, 0])
    for q in range(n): qml.RY(th, wires=q)
    return qml.state()
x = [f"x{i+1}" for i in range(n)]
fig, axes = plt.subplots(1, 2, figsize=(FIG_W_MAX, 2.3))
for ax, (circ, args, title) in zip(axes, [(covariant, ([0.3] * n, [0.5] * (2 * n)), "(a) This work: V(θ) on |0⟩ⁿ before the data D(x) = ⊗RZ(xⱼ)"),
                                          (previous, ([0.3] * n, 1.2), "(b) Previous construction: RY(θ), CNOTs after the encoding (cancel)")]):
    f2, a2 = qml.draw_mpl(circ, style="black_white", decimals=None, show_all_wires=True)(*args)
    f2.canvas.draw(); img = np.asarray(f2.canvas.buffer_rgba()); plt.close(f2)
    ax.imshow(img); ax.axis("off"); ax.set_anchor("N"); ax.set_title(title, fontsize=8, pad=4)
fig.tight_layout()
save_fig(fig, "fig02_circuit", [ROOT / "src" / "qkn" / "feature_maps.py"], SCRIPT, "Feature-map circuits (drawn with 4 qubits; experiments use 8): (a) trainable fiducial state before the encoding; (b) the previous, provably inert construction.")

# ---------------------------------------------------------------- Fig 11: noise-fix
if (RES / "noise_fix_per_seed.csv").exists():
    nf = load("noise_fix_per_seed.csv"); nf["shots"] = nf["shots"].fillna(np.inf)
    xs = sorted(nf["shots"].unique()); xl = [("∞" if np.isinf(s) else f"{int(s):,}") for s in xs]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_MAX, 2.9), sharey=True)
    variants = [("plain-pinv", "k", "pseudo-inverse (ε=1e-6)"), ("jitter-0.1-pinv", "#7f7f7f", "pseudo-inverse, ε=0.1"), ("psd-project", "#1b6ca8", "PSD projection"),
                ("truncate-0.01", "#d95f02", "truncation τ=1e-2"), ("adaptive", "#1b9e77", "adaptive ε=2σ_W√m"), ("adaptive+trunc", "#7570b3", "adaptive + truncation")]
    for ax, m in zip(axes, [20, 60, 120]):
        for v, c, lab in variants:
            g = nf[(nf["variant"] == v) & (nf["m"] == m)].groupby("shots")["acc"].agg(["mean", "std"]).reindex(xs)
            ax.errorbar(range(len(xs)), g["mean"], yerr=g["std"], marker="o", ms=3, color=c, label=lab, capsize=2, lw=1)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xl); ax.set_xlabel("shots per fidelity"); ax.set_title(f"m = {m}")
    axes[0].set_ylabel("test accuracy")
    fig.tight_layout(rect=[0, 0.17, 1, 1]); _h, _l = axes[0].get_legend_handles_labels(); fig.legend(_h, _l, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    save_fig(fig, "fig11_noise_fix", [RES / "noise_fix_per_seed.csv"], SCRIPT, "Noise-robust Nyström variants under finite-shot estimation (two tasks × 10 seeds).")
print("extra figures written")
