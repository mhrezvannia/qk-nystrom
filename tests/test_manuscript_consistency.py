"""Manuscript ↔ results integrity checks (run after analysis/make_tables.py, make_figures.py, manuscript/build_manuscript.py)."""
import json, re, sys, os, hashlib
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript" / "manuscript.md"

pytestmark = pytest.mark.skipif(not MS.exists(), reason="manuscript not built")

def test_no_unverified_or_placeholder_references():
    txt = MS.read_text()
    assert "Siva" not in txt and "Anonymous (2025)" not in txt and "[record to be verified" not in txt

def test_no_notebook_is_a_source():
    for f in list((ROOT / "analysis").glob("*.py")) + [ROOT / "manuscript" / "build_manuscript.py"]:
        assert ".ipynb" not in f.read_text(), f

def test_every_figure_has_metadata_with_existing_sources():
    figs = re.findall(r"\]\(\.\./figures/(fig[^)]+)\.png\)", MS.read_text())
    assert figs, "no figures referenced"
    for name in figs:
        meta = ROOT / "figures" / f"{name}_metadata.json"; assert meta.exists(), name
        m = json.loads(meta.read_text())
        for s in m["sources"]:
            assert (ROOT / s).exists(), s
        # hash recorded at generation must match the current source file (no stale results)
        for fname, h in m["source_sha256"].items():
            for s in m["sources"]:
                if Path(s).name == fname:
                    assert hashlib.sha256((ROOT / s).read_bytes()).hexdigest()[:16] == h, f"{name}: stale source {fname}"

def test_headline_numbers_are_derived_from_tables():
    """The builder must not contain hard-coded accuracy/percentage literals in prose (numbers come from f-string fields)."""
    src = (ROOT / "manuscript" / "build_manuscript.py").read_text()
    prose = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    # allowed literals: equation constants, section numbers, seeds, grids; forbid patterns like '95.83%' or '96.33 ± 1.12'
    bad = re.findall(r"(?<![{\w.])\d{2}\.\d{2} ± \d\.\d{2}", prose) + re.findall(r"(?<![{\w.])\d{2}\.\d{2}%", prose)
    assert not bad, bad

def test_tables_regenerate_identically(tmp_path):
    """Regenerating tables from the current results reproduces the committed CSV tables byte-for-byte (CSV only; md/tex carry dates)."""
    import subprocess, shutil, filecmp
    before = {p.name: p.read_bytes() for p in (ROOT / "tables").glob("*.csv")}
    subprocess.run([sys.executable, str(ROOT / "analysis" / "make_tables.py")], check=True, capture_output=True)
    after = {p.name: p.read_bytes() for p in (ROOT / "tables").glob("*.csv")}
    assert before.keys() == after.keys()
    diffs = [k for k in before if before[k] != after[k]]
    assert not diffs, diffs
