"""Float handling for the generated manuscript (presentation only; no content is altered).

renumber_floats: renumber figures and tables so that the first in-text mention of each is in consecutive order (journal rule:
"cite in consecutive numerical order"). Captions and all mentions are remapped consistently; "Fig. S1"/"Table S3" are untouched.
place_floats:    move each figure block and each table block (caption paragraph + table) out of the trailing "## Tables" /
"## Figures" sections and insert it directly after the paragraph that first mentions it (Word/journal layout). Blocks whose
label is never mentioned raise an error (every figure and table must be cited).
"""
import re

FIG_MENTION = re.compile(r"\bFigs?\. (\d+)\b")            # Fig. 3 / Figs. 3
TAB_MENTION = re.compile(r"\bTables? (\d+)\b")            # Table 3 / Tables 3
FIG_BLOCK = re.compile(r"^!\[\*\*Fig\. (\d+)\*\* .*?\]\([^)]*\)(?:\{[^}]*\})?$", re.M)
TAB_CAPTION = re.compile(r"^\*\*Table (\d+)\*\* .*$", re.M)


def _split_sections(text):
    """Return (body, tables_section, figures_section) using the trailing '## Tables' and '## Figures' headings."""
    it = text.find("\n## Tables\n"); ig = text.find("\n## Figures\n")
    assert it > 0 and ig > it, "expected trailing '## Tables' and '## Figures' sections"
    return text[:it], text[it:ig], text[ig:]


def _mention_order(body, pattern, exclude_caption_prefix):
    first = {}
    for m in pattern.finditer(body):
        first.setdefault(int(m.group(1)), m.start())
    return [k for k, _ in sorted(first.items(), key=lambda kv: kv[1])]


def renumber_floats(text):
    body, tabs, figs = _split_sections(text)
    for kind, pat, tag in (("fig", FIG_MENTION, "Fig"), ("tab", TAB_MENTION, "Table")):
        order = _mention_order(body, pat, tag)
        mapping = {old: new for new, old in enumerate(order, 1)}
        if kind == "fig":
            blocks = [int(x) for x in FIG_BLOCK.findall(figs)]
        else:
            blocks = [int(x) for x in TAB_CAPTION.findall(tabs)]
        missing = [b for b in blocks if b not in mapping]
        if missing:
            raise SystemExit(f"{tag}(s) never cited in the text: {missing}")
        if mapping == {k: k for k in mapping}:
            continue
        # remap mentions (body) and captions (tables/figures sections); each substitution reads the ORIGINAL number, so no chains
        if kind == "fig":
            body = re.sub(r"\b(Figs?)\. (\d+)\b", lambda m: f"{m.group(1)}. {mapping.get(int(m.group(2)), int(m.group(2)))}", body)
            figs = re.sub(r"^!\[\*\*Fig\. (\d+)\*\* ", lambda m: f"![**Fig. {mapping[int(m.group(1))]}** ", figs, flags=re.M)
            figs = re.sub(r"\b(Figs?)\. (\d+)\b", lambda m: f"{m.group(1)}. {mapping.get(int(m.group(2)), int(m.group(2)))}", figs)
            tabs = re.sub(r"\b(Figs?)\. (\d+)\b", lambda m: f"{m.group(1)}. {mapping.get(int(m.group(2)), int(m.group(2)))}", tabs)
        else:
            body = re.sub(r"\b(Tables?) (\d+)\b", lambda m: f"{m.group(1)} {mapping.get(int(m.group(2)), int(m.group(2)))}", body)
            tabs = re.sub(r"^\*\*Table (\d+)\*\* ", lambda m: f"**Table {mapping[int(m.group(1))]}** ", tabs, flags=re.M)
            tabs = re.sub(r"(?<!\*\*)\b(Tables?) (\d+)\b", lambda m: f"{m.group(1)} {mapping.get(int(m.group(2)), int(m.group(2)))}", tabs)
            figs = re.sub(r"\b(Tables?) (\d+)\b", lambda m: f"{m.group(1)} {mapping.get(int(m.group(2)), int(m.group(2)))}", figs)
    return body + tabs + figs


def _table_blocks(tabs):
    """Split the '## Tables' section into {number: block} where block = caption paragraph + table (+ trailing blank)."""
    lines = tabs.split("\n"); blocks = {}; rest = []
    i = 0
    while i < len(lines):
        m = TAB_CAPTION.match(lines[i])
        if m:
            j = i + 1
            while j < len(lines) and lines[j].strip() == "": j += 1          # blank lines
            while j < len(lines) and lines[j].startswith("|"): j += 1       # table rows
            blocks[int(m.group(1))] = "\n".join(lines[i:j]).rstrip("\n"); i = j
        else:
            rest.append(lines[i]); i += 1
    return blocks, "\n".join(rest)


def _fig_blocks(figs):
    blocks = {int(m.group(1)): m.group(0) for m in FIG_BLOCK.finditer(figs)}
    rest = FIG_BLOCK.sub("", figs)
    return blocks, rest


def _insert_after_first_mention(body, num, block, pattern):
    """Insert block after the end of the paragraph containing the first mention of `num`."""
    for m in pattern.finditer(body):
        if int(m.group(1)) == num:
            end = body.find("\n\n", m.end())
            if end < 0: end = len(body)
            return body[:end] + "\n\n" + block + body[end:]
    raise SystemExit(f"float {num} has no in-text mention")


def place_floats(text):
    body, tabs, figs = _split_sections(text)
    tblocks, tabs_rest = _table_blocks(tabs)
    fblocks, figs_rest = _fig_blocks(figs)
    # insert in reverse order of first mention so earlier insertions do not shift later anchors incorrectly
    order_t = _mention_order(body, TAB_MENTION, "Table"); order_f = _mention_order(body, FIG_MENTION, "Fig")
    for n in reversed(order_t):
        if n in tblocks: body = _insert_after_first_mention(body, n, tblocks.pop(n), TAB_MENTION)
    for n in reversed(order_f):
        if n in fblocks: body = _insert_after_first_mention(body, n, fblocks.pop(n), FIG_MENTION)
    if tblocks or fblocks:
        raise SystemExit(f"uncited floats: tables {sorted(tblocks)}, figures {sorted(fblocks)}")
    # keep whatever prose remains in the old sections (e.g. the Online Resource paragraph), drop the empty headings
    tail = "\n".join(l for l in (tabs_rest + "\n" + figs_rest).split("\n") if l.strip() not in ("## Tables", "## Figures"))
    tail = re.sub(r"\n{3,}", "\n\n", tail).strip("\n")
    if tail:  # remaining prose (the Online Resource paragraph) becomes the 'Supplementary information' statement before the references
        assert "\n## References" in body
        body = body.replace("\n## References", "\n**Supplementary information** " + tail + "\n\n## References", 1)
    return body.rstrip("\n") + "\n"
