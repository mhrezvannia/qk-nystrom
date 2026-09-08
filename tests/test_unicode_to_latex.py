"""Tests for scripts/unicode_to_latex.py (pdflatex-safe arXiv source)."""
import sys, re, shutil, subprocess
import pytest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import unicode_to_latex as u  # noqa: E402

PRE = "\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n\\usepackage{lmodern}\n\\IfFileExists{microtype.sty}{\\usepackage[]{microtype}\n\\UseMicrotypeSet[protrusion]{basicmath}}{}\n"


def test_body_only_and_preamble_cleanup():
    src = PRE + "\\begin{document}\nK = |χ_p((x′−x)/2)|² with ‖W_ε⁻¹‖₂ ≪ 1, λ ∈ ℝ — done ∎\n\\end{document}\n"
    out = u.convert(src)
    head, _, body = out.partition("\\begin{document}")
    assert "lmodern" in head and "microtype" in head           # pandoc font setup kept (arXiv TeX Live has both)
    assert "AtBeginEnvironment{longtable}{\\scriptsize}" in head          # inserted when absent
    assert "\\ensuremath{\\chi}" in body and "\\ensuremath{\\varepsilon}" in body and "\\ensuremath{\\ll}" in body
    assert "\\ensuremath{^{-1}}" in body and "\\ensuremath{_{2}}" in body  # collapsed consecutive scripts
    assert "---" in body and "\\ensuremath{\\blacksquare}" in body
    assert not [c for c in body if ord(c) > 0x24F], "non-Latin characters left in body"


@pytest.mark.skipif(not (ROOT / "manuscript" / "manuscript.md").exists(), reason="manuscript not built yet")
def test_preamble_not_duplicated_and_mapping_is_complete_for_manuscript():
    head_with = PRE + "\\usepackage{etoolbox}\\AtBeginEnvironment{longtable}{\\scriptsize}\n"
    out = u.convert(head_with + "\\begin{document}x\\end{document}")
    assert out.count("AtBeginEnvironment{longtable}") == 1
    # every non-Latin character that occurs in the generated manuscript must have a mapping
    md = (ROOT / "manuscript" / "manuscript.md").read_text(encoding="utf-8")
    chars = {c for c in md if ord(c) > 0x24F}
    unmapped = sorted(c for c in chars if c not in u.M and c not in "\u0302\u0303")  # combining hat/tilde are attached to a base letter by pandoc
    assert not unmapped, f"unmapped characters in manuscript.md: {unmapped}"


def test_converted_snippet_compiles_with_pdflatex():
    if shutil.which("pdflatex") is None:
        import pytest; pytest.skip("pdflatex not installed")
    import tempfile
    d = Path(tempfile.mkdtemp())
    src = ("\\documentclass{article}\\usepackage{amsmath,amssymb}\\usepackage{longtable}\n\\begin{document}\n"
           # pandoc escapes underscores in prose as \_ ; the snippet mimics that
           "Kernel K = |χ\\_p((x′−x)/2)|², bound ‖ΔW‖₂ \\textless{} λ\\_min(W)+ε, ratio A ≳ 20, m ≪ N, g ≈ 0.46√N, θ ∈ [0, 2π]ⁿ, W⁺ = Wᵀ, 10⁻³ ✓ ∎\n"
           "\\begin{longtable}{ll}a & b\\\\\\end{longtable}\n\\end{document}\n")
    (d / "t.tex").write_text(u.convert(src), encoding="utf-8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"], cwd=d, capture_output=True, text=True)
    log = (d / "t.log").read_text(errors="ignore")
    assert r.returncode == 0 and (d / "t.pdf").exists()
    assert not [l for l in log.splitlines() if l.startswith("!")], "LaTeX errors"
    assert not re.search(r"^Missing character", log, flags=re.M)
