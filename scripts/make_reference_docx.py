"""Generate manuscript/reference_qmi.docx — the pandoc reference document that gives the Word manuscript its journal styling
(Times New Roman, 11-pt body, black bold headings, 9-pt captions, 8-pt tables with horizontal rules, A4 with 1.8-cm side margins so a
174-mm figure fits, automatic page numbers in the footer). Deterministic: derived from pandoc's own default reference.docx.
Usage: python scripts/make_reference_docx.py
"""
import re, subprocess, shutil, tempfile, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "reference_qmi.docx"
FONT = "Times New Roman"
RFONTS = f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}" />'


def style_block(s, style_id):
    m = re.search(rf'<w:style [^>]*w:styleId="{style_id}".*?</w:style>', s, flags=re.S)
    assert m, style_id
    return m


def set_rpr(block, **kw):
    """Replace/insert simple run properties in a style block (sz, b, i, color removal, fonts)."""
    rpr = re.search(r"<w:rPr>.*?</w:rPr>", block, flags=re.S)
    body = rpr.group(0)[7:-8] if rpr else ""
    body = re.sub(r"<w:rFonts [^>]*/>", "", body); body = re.sub(r"<w:color [^>]*/>", "", body)
    body = re.sub(r"<w:sz [^>]*/>", "", body); body = re.sub(r"<w:szCs [^>]*/>", "", body)
    if kw.get("font", True): body = RFONTS + body
    if "sz" in kw: body += f'<w:sz w:val="{kw["sz"]}" /><w:szCs w:val="{kw["sz"]}" />'
    if kw.get("bold") is True and "<w:b />" not in body and "<w:b/>" not in body: body += "<w:b /><w:bCs />"
    if kw.get("bold") is False: body = body.replace("<w:b />", "").replace("<w:bCs />", "")
    if kw.get("italic") is False: body = body.replace("<w:i />", "").replace("<w:iCs />", "")
    if kw.get("italic") is True and "<w:i />" not in body: body += "<w:i /><w:iCs />"
    new = f"<w:rPr>{body}</w:rPr>"
    return block.replace(rpr.group(0), new) if rpr else block.replace("</w:style>", new + "</w:style>")


def set_ppr(block, inner_add="", jc=None, spacing=None, ind=None):
    ppr = re.search(r"<w:pPr>.*?</w:pPr>", block, flags=re.S)
    body = ppr.group(0)[7:-8] if ppr else ""
    if jc: body = re.sub(r"<w:jc [^>]*/>", "", body) + f'<w:jc w:val="{jc}" />'
    if spacing: body = re.sub(r"<w:spacing [^>]*/>", "", body) + f"<w:spacing {spacing} />"
    if ind: body = re.sub(r"<w:ind [^>]*/>", "", body) + f"<w:ind {ind} />"
    body += inner_add
    new = f"<w:pPr>{body}</w:pPr>"
    return block.replace(ppr.group(0), new) if ppr else block.replace("<w:rPr>", new + "<w:rPr>", 1) if "<w:rPr>" in block else block.replace("</w:style>", new + "</w:style>")


def main():
    tmp = Path(tempfile.mkdtemp()); ref = tmp / "ref.docx"
    subprocess.run(["pandoc", "-o", str(ref), "--print-default-data-file", "reference.docx"], check=True)
    x = tmp / "x"; x.mkdir()
    with zipfile.ZipFile(ref) as z: z.extractall(x)
    s = (x / "word" / "styles.xml").read_text(encoding="utf-8")
    # document defaults: Times New Roman 11 pt, 6 pt after paragraphs, single spacing
    s = re.sub(r'<w:rFonts w:asciiTheme="minorHAnsi"[^>]*/>', RFONTS, s, count=1)
    s = s.replace('<w:sz w:val="24" />\n        <w:szCs w:val="24" />', '<w:sz w:val="22" />\n        <w:szCs w:val="22" />', 1)
    s = s.replace('<w:pPrDefault>\n      <w:pPr>\n        <w:spacing w:after="200" />', '<w:pPrDefault>\n      <w:pPr>\n        <w:spacing w:after="120" w:line="264" w:lineRule="auto" />', 1)
    edits = {
        "Title": dict(rpr=dict(sz=32, bold=True), ppr=dict(jc="center", spacing='w:before="240" w:after="240"')),
        "Author": dict(rpr=dict(sz=22, bold=False), ppr=dict(jc="left")),
        "Heading1": dict(rpr=dict(sz=28, bold=True), ppr=dict(spacing='w:before="360" w:after="120"')),
        "Heading2": dict(rpr=dict(sz=24, bold=True), ppr=dict(spacing='w:before="240" w:after="120"')),
        "Heading3": dict(rpr=dict(sz=22, bold=True, italic=True), ppr=dict(spacing='w:before="200" w:after="80"')),
        "BodyText": dict(rpr=dict(sz=22), ppr=dict(jc="both")),
        "FirstParagraph": dict(rpr=dict(sz=22), ppr=dict(jc="both")),
        "Compact": dict(rpr=dict()),   # no explicit size: inherits 11 pt in lists, 8 pt inside tables (table style)
        "ImageCaption": dict(rpr=dict(sz=18, italic=False), ppr=dict(jc="left", spacing='w:before="60" w:after="240"')),
        "TableCaption": dict(rpr=dict(sz=18, italic=False), ppr=dict(jc="left")),
        "Caption": dict(rpr=dict(sz=18, italic=False)),
        "Bibliography": dict(rpr=dict(sz=20), ppr=dict(ind='w:left="454" w:hanging="454"', spacing='w:after="80"', jc="left")),
        "Abstract": dict(rpr=dict(sz=20)),
        "BlockText": dict(rpr=dict(sz=20)),
        "FootnoteText": dict(rpr=dict(sz=18)),
    }
    for sid, e in edits.items():
        m = style_block(s, sid); blk = m.group(0)
        if "rpr" in e: blk = set_rpr(blk, **e["rpr"])
        if "ppr" in e: blk = set_ppr(blk, **e["ppr"])
        s = s[:m.start()] + blk + s[m.end():]
    # headings: remove theme colours everywhere (black text)
    s = re.sub(r'<w:color w:val="[0-9A-Fa-f]{6}" w:themeColor="[^"]*"( w:themeShade="[^"]*")? />', "", s)
    # table style: 8-pt text, horizontal rules only, compact cell margins
    m = style_block(s, "Table"); blk = m.group(0)
    blk = blk.replace('<w:tblPr>\n      <w:tblInd w:w="0" w:type="dxa" />',
                      '<w:tblPr>\n      <w:tblInd w:w="0" w:type="dxa" />\n      <w:tblBorders><w:top w:val="single" w:sz="8" w:space="0" w:color="000000" /><w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000" /></w:tblBorders>', 1)
    blk = blk.replace('<w:left w:w="108" w:type="dxa" />', '<w:left w:w="57" w:type="dxa" />').replace('<w:right w:w="108" w:type="dxa" />', '<w:right w:w="57" w:type="dxa" />')
    blk = blk.replace("<w:tblPr>", f'<w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto" /></w:pPr><w:rPr>{RFONTS}<w:sz w:val="16" /><w:szCs w:val="16" /></w:rPr><w:tblPr>', 1)
    s = s[:m.start()] + blk + s[m.end():]
    (x / "word" / "styles.xml").write_text(s, encoding="utf-8")
    # page set-up (A4, 1.8 cm sides, 2.5 cm top/bottom) and footer with automatic page numbers
    d = (x / "word" / "document.xml").read_text(encoding="utf-8")
    sect = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFooterQmi" /><w:pgSz w:w="11906" w:h="16838" />'
            '<w:pgMar w:top="1418" w:right="1021" w:bottom="1418" w:left="1021" w:header="709" w:footer="709" w:gutter="0" />'
            '<w:cols w:space="708" /><w:docGrid w:linePitch="360" /></w:sectPr>')
    if "<w:sectPr />" in d:
        d = d.replace("<w:sectPr />", sect)
    else:
        assert "<w:sectPr" not in d; d = d.replace("</w:body>", sect + "</w:body>")
    (x / "word" / "document.xml").write_text(d, encoding="utf-8")
    (x / "word" / "footer1.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<w:p><w:pPr><w:jc w:val="center" /></w:pPr>'
        f'<w:r><w:rPr>{RFONTS}<w:sz w:val="20" /></w:rPr><w:fldChar w:fldCharType="begin" /></w:r>'
        f'<w:r><w:rPr>{RFONTS}<w:sz w:val="20" /></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
        f'<w:r><w:rPr>{RFONTS}<w:sz w:val="20" /></w:rPr><w:fldChar w:fldCharType="separate" /></w:r>'
        f'<w:r><w:rPr>{RFONTS}<w:sz w:val="20" /></w:rPr><w:t>1</w:t></w:r>'
        f'<w:r><w:rPr>{RFONTS}<w:sz w:val="20" /></w:rPr><w:fldChar w:fldCharType="end" /></w:r></w:p></w:ftr>', encoding="utf-8")
    rels = (x / "word" / "_rels" / "document.xml.rels").read_text(encoding="utf-8")
    rels = rels.replace("</Relationships>", '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Id="rIdFooterQmi" Target="footer1.xml" /></Relationships>')
    (x / "word" / "_rels" / "document.xml.rels").write_text(rels, encoding="utf-8")
    ct = (x / "[Content_Types].xml").read_text(encoding="utf-8")
    ct = ct.replace("</Types>", '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml" /></Types>')
    (x / "[Content_Types].xml").write_text(ct, encoding="utf-8")
    OUT.unlink(missing_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        # [Content_Types].xml first, as Word expects
        z.write(x / "[Content_Types].xml", "[Content_Types].xml")
        for p in sorted(x.rglob("*")):
            if p.is_file() and p.name != "[Content_Types].xml":
                z.write(p, str(p.relative_to(x)))
    shutil.rmtree(tmp)
    print("written", OUT)


if __name__ == "__main__":
    main()
