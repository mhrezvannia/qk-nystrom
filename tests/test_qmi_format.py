"""Journal-format checks (Quantum Machine Intelligence submission guidelines) on the generated manuscript and figure files."""
import re, sys, zipfile, shutil, subprocess, glob
from pathlib import Path
import pytest, yaml
ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript" / "manuscript.md"
pytestmark = pytest.mark.skipif(not MS.exists(), reason="manuscript not built")
sys.path.insert(0, str(ROOT / "manuscript"))


def _text():
    return MS.read_text(encoding="utf-8")


def test_abstract_and_keywords_within_limits():
    t = _text(); a = t.split("## Abstract")[1].split("**Keywords:**")[0]
    n = len(a.split()); assert 150 <= n <= 250, f"abstract has {n} words (QMI: 150–250)"
    kw = re.search(r"\*\*Keywords:\*\*(.*)", t).group(1).strip().rstrip(".").split(";")
    assert 4 <= len(kw) <= 6, kw
    assert not re.search(r"\(?[A-Z][\w\-]* (?:et al\. |and [A-Z][\w\-]* )?\d{4}\)", a), "no citations in the abstract"
    assert "Nyström" in a and "\\ll" in a   # LaTeX math in the abstract, Unicode text intact
    assert (ROOT / "manuscript" / "abstract_plain.txt").exists()


def test_title_page_has_author_metadata_and_no_placeholders():
    a = yaml.safe_load((ROOT / "manuscript" / "authors.yaml").read_text(encoding="utf-8")); t = _text()
    head = t.split("## Abstract")[0]
    for x in a["authors"]: assert x["name"] in head
    for x in a["affiliations"]: assert x["affiliation"] in head
    assert a["corresponding"]["email"] in head
    assert "FILL IN" not in t and "[withheld" not in t


def test_heading_depth_and_declarations():
    t = _text()
    assert not re.search(r"^#{4,} ", t, flags=re.M), "more than three heading levels"
    decl = t.split("## Statements and Declarations")[1].split("## References")[0]
    for k in ("**Funding**", "**Competing interests**", "**Ethics approval", "**Data availability**", "**Code availability**", "**Author contributions**", "**Supplementary information**"):
        assert k in decl, k
    assert "### 3.8 Use of artificial-intelligence tools" in t   # LLM use documented in Methods (QMI policy)


def test_floats_numbered_consecutively_cited_before_shown_and_captions_formatted():
    t = _text(); body = t.split("## References")[0]
    for kind, cap_re, men_re in (("Fig", r"^!\[\*\*Fig\. (\d+)\*\* ", r"\bFigs?\. (\d+)\b"), ("Table", r"^\*\*Table (\d+)\*\* ", r"\bTables? (\d+)\b")):
        caps = [int(x) for x in re.findall(cap_re, body, flags=re.M)]
        assert caps == list(range(1, len(caps) + 1)), f"{kind} captions not consecutive: {caps}"
        first = {}
        for m in re.finditer(men_re, body): first.setdefault(int(m.group(1)), m.start())
        order = [k for k, _ in sorted(first.items(), key=lambda kv: kv[1])]
        assert order == list(range(1, len(caps) + 1)), f"{kind} first mentions out of order: {order}"
        for n in caps:   # the caption block appears after the first in-text mention
            cap_pos = re.search(cap_re.replace(r"(\d+)", str(n)), body, flags=re.M).start()
            assert first[n] < cap_pos, f"{kind} {n} shown before it is cited"
    assert not re.search(r"\*\*Fig\. \d+\.\*\*|\*\*Table \d+\.\*\*", body), "no punctuation after the figure/table number"
    assert not re.search(r"\bFigure \d", body), "use 'Fig. N' in running text"


def test_citations_are_name_year_and_resolve_to_reference_list():
    import refs
    t = _text(); body, reflist = t.split("## References")
    assert not re.search(r"\[[^\]]*\b(19|20)\d{2}[a-z]?\]", body.split("## Statements")[0]), "bracket citations left"
    rendered = [l for l in reflist.strip().split("\n") if l.strip() and not l.startswith(":::")]
    assert len(rendered) == len(refs.REFS)
    assert rendered == [refs.render(r) for r in refs.reference_list()]          # alphabetical Springer Basic
    labels = set(refs.label(r) for r in refs.REFS)
    cited = set()
    for m in re.finditer(r"\(((?:[A-Z][\w\-]*(?: et al\.| and [A-Z][\w\-]*)? \d{4}(?:; )?)+)\)", body):
        for it in m.group(1).split("; "): cited.add(it.strip())
    for m in re.finditer(r"([A-Z][\w\-]*(?: et al\.| and [A-Z][\w\-]*)?) \((\d{4})\)", body):
        cited.add(f"{m.group(1)} {m.group(2)}")
    unknown = sorted(c for c in cited if c not in labels)
    assert not unknown, unknown
    assert labels <= cited, f"uncited references: {sorted(labels - cited)}"


def test_supplement_references_and_online_resources():
    t = _text()
    assert "Supplement" not in t.replace("Supplementary information", ""), "use 'Online Resource' wording"
    assert "Online Resource 1" in t and "Online Resource 2" in t
    sup = (ROOT / "manuscript" / "supplement.md").read_text(encoding="utf-8")
    for lab in ("**Table S1**", "**Table S3**", "**Table S4**", "**Table S5**", "**Table S6**", "**Table S7**", "**Table S8**", "**Fig. S1**", "**Fig. S2**", "## S1 ", "## S4 "):
        assert lab in sup, lab
    assert (ROOT / "manuscript" / "ESM_2.xlsx").exists()


def test_figure_files_meet_artwork_specification():
    from PIL import Image
    tifs = sorted((ROOT / "figures").glob("*.tif")); assert len(tifs) >= 12
    for f in tifs:
        im = Image.open(f); dpi = [float(x) for x in im.info.get("dpi", (0, 0))]
        assert abs(dpi[0] - 600) < 1, f"{f.name}: {dpi} dpi (need 600)"
        w_mm, h_mm = im.size[0] / dpi[0] * 25.4, im.size[1] / dpi[1] * 25.4
        assert w_mm <= 174.5 and h_mm <= 234, f"{f.name}: {w_mm:.1f} x {h_mm:.1f} mm exceeds 174 x 234 mm"
    if shutil.which("pdffonts"):
        for f in (ROOT / "figures").glob("*.pdf"):
            out = subprocess.run(["pdffonts", str(f)], capture_output=True, text=True).stdout
            assert "Type 3" not in out, f"{f.name} embeds Type 3 (bitmap) fonts"
            assert "LiberationSans" in out or "DejaVuSans" in out, f"{f.name}: sans-serif lettering expected"


def test_manuscript_docx_structure():
    d = ROOT / "manuscript" / "manuscript.docx"; assert d.exists()
    with zipfile.ZipFile(d) as z:
        names = z.namelist(); doc = z.read("word/document.xml").decode("utf-8"); styles = z.read("word/styles.xml").decode("utf-8")
        assert any(n.startswith("word/footer") for n in names) and " PAGE " in z.read([n for n in names if n.startswith("word/footer")][0]).decode("utf-8")
    assert "Times New Roman" in styles
    assert doc.count("<m:oMathPara") >= 8, "display equations must be Word equation objects"
    assert doc.count("<m:oMath>") >= 300, "inline formulas must be Word equation objects"
    t = _text()
    assert doc.count("<w:tbl>") == len(re.findall(r"^\*\*Table \d+\*\* ", t, flags=re.M))
    assert doc.count("<pic:pic>") == len(re.findall(r"^!\[\*\*Fig\. \d+\*\* ", t, flags=re.M))
    assert "<w:tblLayout w:type=\"fixed\"" in doc and "<w:tcW " in doc   # explicit column widths (scripts/fix_docx_tables.py)


def test_all_formulas_are_latex_math():
    """No Unicode mathematical notation remains outside $…$ in the prose (tables and reference list excluded)."""
    t = _text().split("## References")[0]
    body = "\n".join(l for l in t.split("\n") if not l.startswith("|"))
    nomath = re.sub(r"\$\$.*?\$\$", "", body, flags=re.S); nomath = re.sub(r"\$[^$\n]+\$", "", nomath)
    leftover = sorted({ch for ch in nomath if ord(ch) > 127 and ch not in "öíčäéü–—…§✓∎’“”"})
    assert not leftover, f"Unicode math outside math spans: {leftover}"
    spans = re.findall(r"\$([^$\n]+)\$", body)
    assert len(spans) > 300
    for s in spans:   # balanced braces, no stray Unicode inside LaTeX math
        assert s.count("{") == s.count("}"), s
        assert not [ch for ch in s if ord(ch) > 127], s
