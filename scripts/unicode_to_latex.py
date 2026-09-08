"""Replace Unicode symbols in a pandoc-generated .tex file by pdflatex-safe LaTeX macros (for the arXiv package).
Usage: python scripts/unicode_to_latex.py in.tex out.tex"""
import sys, re
M = {
    # Greek
    "α": r"\ensuremath{\alpha}", "β": r"\ensuremath{\beta}", "γ": r"\ensuremath{\gamma}", "δ": r"\ensuremath{\delta}", "ε": r"\ensuremath{\varepsilon}",
    "θ": r"\ensuremath{\theta}", "λ": r"\ensuremath{\lambda}", "μ": r"\ensuremath{\mu}", "π": r"\ensuremath{\pi}", "σ": r"\ensuremath{\sigma}",
    "τ": r"\ensuremath{\tau}", "φ": r"\ensuremath{\varphi}", "χ": r"\ensuremath{\chi}", "ψ": r"\ensuremath{\psi}", "ω": r"\ensuremath{\omega}",
    "Δ": r"\ensuremath{\Delta}", "Σ": r"\ensuremath{\Sigma}", "Φ": r"\ensuremath{\Phi}", "Ψ": r"\ensuremath{\Psi}", "Λ": r"\ensuremath{\Lambda}", "Γ": r"\ensuremath{\Gamma}",
    # operators / relations
    "∏": r"\ensuremath{\prod}", "∑": r"\ensuremath{\sum}", "√": r"\ensuremath{\surd}", "∈": r"\ensuremath{\in}", "∞": r"\ensuremath{\infty}",
    "≤": r"\ensuremath{\le}", "≥": r"\ensuremath{\ge}", "≈": r"\ensuremath{\approx}", "≪": r"\ensuremath{\ll}", "≫": r"\ensuremath{\gg}", "≳": r"\ensuremath{\gtrsim}", "≲": r"\ensuremath{\lesssim}", "¦": r"\ensuremath{|}",
    "≠": r"\ensuremath{\ne}", "≡": r"\ensuremath{\equiv}", "→": r"\ensuremath{\rightarrow}", "⇒": r"\ensuremath{\Rightarrow}", "⇔": r"\ensuremath{\Leftrightarrow}",
    "×": r"\ensuremath{\times}", "·": r"\ensuremath{\cdot}", "⋅": r"\ensuremath{\cdot}", "−": r"\ensuremath{-}", "±": r"\ensuremath{\pm}", "∓": r"\ensuremath{\mp}",
    "⊗": r"\ensuremath{\otimes}", "⊕": r"\ensuremath{\oplus}", "∂": r"\ensuremath{\partial}", "∇": r"\ensuremath{\nabla}", "∝": r"\ensuremath{\propto}",
    "⟨": r"\ensuremath{\langle}", "⟩": r"\ensuremath{\rangle}", "‖": r"\ensuremath{\|}", "†": r"\ensuremath{\dagger}", "∎": r"\ensuremath{\blacksquare}",
    "✓": r"\ensuremath{\checkmark}", "∘": r"\ensuremath{\circ}", "∼": r"\ensuremath{\sim}", "∅": r"\ensuremath{\emptyset}", "ℝ": r"\ensuremath{\mathbb{R}}",
    "ℂ": r"\ensuremath{\mathbb{C}}", "𝔼": r"\ensuremath{\mathbb{E}}", "′": r"\ensuremath{'}", "″": r"\ensuremath{''}", "ˆ": r"\^{}", "̃": "",
    # super/subscripts
    "⁰": r"\ensuremath{^{0}}", "¹": r"\ensuremath{^{1}}", "²": r"\ensuremath{^{2}}", "³": r"\ensuremath{^{3}}", "⁴": r"\ensuremath{^{4}}", "⁵": r"\ensuremath{^{5}}",
    "⁶": r"\ensuremath{^{6}}", "⁷": r"\ensuremath{^{7}}", "⁸": r"\ensuremath{^{8}}", "⁹": r"\ensuremath{^{9}}", "⁺": r"\ensuremath{^{+}}", "⁻": r"\ensuremath{^{-}}",
    "ⁿ": r"\ensuremath{^{n}}", "ᵀ": r"\ensuremath{^{\mathsf{T}}}", "ᵢ": r"\ensuremath{_{i}}", "ⱼ": r"\ensuremath{_{j}}", "ₖ": r"\ensuremath{_{k}}",
    "₀": r"\ensuremath{_{0}}", "₁": r"\ensuremath{_{1}}", "₂": r"\ensuremath{_{2}}", "₃": r"\ensuremath{_{3}}", "ₘ": r"\ensuremath{_{m}}", "ₙ": r"\ensuremath{_{n}}",
    "ₐ": r"\ensuremath{_{a}}", "ₜ": r"\ensuremath{_{t}}", "ₑ": r"\ensuremath{_{e}}", "ₓ": r"\ensuremath{_{x}}",
    # typography
    "—": "---", "–": "--", "’": "'", "‘": "`", "“": "``", "”": "''", "…": r"\ldots{}", "\u00a0": "~", "\u2009": r"\,", "₌": r"\ensuremath{_{=}}",
    "ā": r"\={a}", "ö": r"\"o", "ü": r"\"u", "é": r"\'e", "í": r"\'{\i}", "ó": r"\'o", "č": r"\v{c}", "ł": r"\l{}", "ć": r"\'c", "ä": r"\"a", "ø": r"\o{}", "ñ": r"\~n", "ş": r"\c{s}",
}
# collapse consecutive \ensuremath{^{.}} superscripts into one group (e.g. ⁻¹ -> ^{-1})
def collapse(t):
    t = re.sub(r"\\ensuremath\{\^\{([^}]*)\}\}\\ensuremath\{\^\{([^}]*)\}\}", r"\\ensuremath{^{\1\2}}", t)
    t = re.sub(r"\\ensuremath\{_\{([^}]*)\}\}\\ensuremath\{_\{([^}]*)\}\}", r"\\ensuremath{_{\1\2}}", t)
    return t

def convert(text: str) -> str:
    """Convert only the document body; the pandoc preamble (lmodern Type 1 fonts, microtype, T1 encoding) is left untouched."""
    head, sep, body = text.partition("\\begin{document}")
    if "AtBeginEnvironment{longtable}" not in head:  # standalone use (build_submission.py supplies these via header.tex)
        head = head.rstrip("\n") + "\n\\usepackage{amssymb}\\usepackage{etoolbox}\\AtBeginEnvironment{longtable}{\\scriptsize}\\setlength\\LTleft{0pt}\\setlength\\LTright{0pt}\n"
    body = re.sub("([A-Za-z])\u0302", r"\\ensuremath{\\hat{\1}}", body)     # combining circumflex (K̂)
    body = re.sub("([A-Za-z])\u0303", r"\\ensuremath{\\tilde{\1}}", body)   # combining tilde (K̃)
    t = "".join(M.get(ch, ch) for ch in body)
    for _ in range(3):
        t = collapse(t)
    return head + sep + t

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    t = open(src, encoding="utf-8").read()
    open(dst, "w", encoding="utf-8").write(convert(t))
    left = sorted({c for c in convert(t) if ord(c) > 0x24F})
    print("converted; remaining non-Latin characters:", left)
