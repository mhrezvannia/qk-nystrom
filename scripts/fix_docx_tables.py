"""Post-process a pandoc-generated .docx so every table has explicit, content-proportional column widths that sum to the text
width (Word/LibreOffice otherwise lay wide tables out with equal columns and wrap cell text).
Usage: python scripts/fix_docx_tables.py in.docx [out.docx]   (in place if out omitted)
Presentation only: cell contents are untouched."""
import sys, zipfile, shutil, tempfile, re
from pathlib import Path
from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
TEXT_WIDTH_TWIPS = 11906 - 2 * 1021   # A4 minus 1.8-cm side margins (matches manuscript/reference_qmi.docx)
CELL_MARGIN = 2 * 57                  # left + right cell margins of the 'Table' style


def q(tag): return f"{{{W}}}{tag}"


def cell_len(tc):
    txt = "".join(t.text or "" for t in tc.iter(q("t")))
    # longest word matters for non-breaking numbers; use max(longest token, 0.75 * total) as the effective length
    toks = txt.split()
    return 0.0 if not toks else max(max(len(x) for x in toks), 0.85 * len(txt))


def set_cell_font(tbl, half_points):
    """Explicit run size + zero paragraph spacing in every table cell (paragraph styles would otherwise override the table style)."""
    for tc in tbl.iter(q("tc")):
        for p in tc.findall(q("p")):
            ppr = p.find(q("pPr"))
            if ppr is None:
                ppr = etree.Element(q("pPr")); p.insert(0, ppr)
            for old in ppr.findall(q("spacing")): ppr.remove(old)
            sp = etree.SubElement(ppr, q("spacing")); sp.set(q("before"), "0"); sp.set(q("after"), "0"); sp.set(q("line"), "240"); sp.set(q("lineRule"), "auto")
            for r in p.findall(q("r")):
                rpr = r.find(q("rPr"))
                if rpr is None:
                    rpr = etree.Element(q("rPr")); r.insert(0, rpr)
                for old in rpr.findall(q("sz")) + rpr.findall(q("szCs")): rpr.remove(old)
                sz = etree.SubElement(rpr, q("sz")); sz.set(q("val"), str(half_points))
                szc = etree.SubElement(rpr, q("szCs")); szc.set(q("val"), str(half_points))


def fix_table(tbl):
    rows = tbl.findall(q("tr"))
    if not rows: return
    ncol = max(len(r.findall(q("tc"))) for r in rows)
    need = [0.0] * ncol
    for r in rows:
        for j, tc in enumerate(r.findall(q("tc"))):
            need[j] = max(need[j], cell_len(tc))
    # 8-pt text; fall back to 7 pt for tables that would not fit the text width
    for half_points, char_twips in ((16, 88), (14, 77)):   # ≈0.55 em per glyph (digits/upper case dominate table cells)
        widths = [max(3.0, n) * char_twips + CELL_MARGIN for n in need]
        if sum(widths) <= TEXT_WIDTH_TWIPS: break
    set_cell_font(tbl, half_points)
    total = sum(widths)
    scale = TEXT_WIDTH_TWIPS / total if total > TEXT_WIDTH_TWIPS else 1.0   # shrink proportionally only if still too wide
    widths = [int(w * scale) for w in widths]
    tblpr = tbl.find(q("tblPr"))
    if tblpr is None:
        tblpr = etree.SubElement(tbl, q("tblPr")); tbl.insert(0, tblpr)
    for old in tblpr.findall(q("tblW")) + tblpr.findall(q("tblLayout")): tblpr.remove(old)
    tw = etree.SubElement(tblpr, q("tblW")); tw.set(q("w"), str(sum(widths))); tw.set(q("type"), "dxa")
    lay = etree.SubElement(tblpr, q("tblLayout")); lay.set(q("type"), "fixed")
    grid = tbl.find(q("tblGrid"))
    if grid is None:
        grid = etree.Element(q("tblGrid")); tbl.insert(list(tbl).index(tblpr) + 1, grid)
    for gc in list(grid): grid.remove(gc)
    for w in widths:
        gc = etree.SubElement(grid, q("gridCol")); gc.set(q("w"), str(w))
    for r in rows:
        for j, tc in enumerate(r.findall(q("tc"))):
            tcpr = tc.find(q("tcPr"))
            if tcpr is None:
                tcpr = etree.Element(q("tcPr")); tc.insert(0, tcpr)
            for old in tcpr.findall(q("tcW")): tcpr.remove(old)
            tcw = etree.Element(q("tcW")); tcw.set(q("w"), str(widths[j])); tcw.set(q("type"), "dxa"); tcpr.insert(0, tcw)


def fix(src: Path, dst: Path):
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(src) as z: z.extractall(tmp)
    p = tmp / "word" / "document.xml"
    tree = etree.parse(str(p)); root = tree.getroot()
    tables = root.findall(f".//{q('tbl')}")
    for tbl in tables: fix_table(tbl)
    tree.write(str(p), xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp_out = tmp / "out.docx"
    with zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tmp / "[Content_Types].xml", "[Content_Types].xml")
        for f in sorted(tmp.rglob("*")):
            if f.is_file() and f.name not in ("[Content_Types].xml", "out.docx"):
                z.write(f, str(f.relative_to(tmp)))
    shutil.move(str(tmp_out), str(dst)); shutil.rmtree(tmp)
    return len(tables)


if __name__ == "__main__":
    src = Path(sys.argv[1]); dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src
    print("tables fixed:", fix(src, dst))
