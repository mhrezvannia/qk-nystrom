"""Reference handling for the manuscript: Springer Basic (name–year) rendering, in-text citation conversion, BibTeX export.

Source of truth: manuscript/references.yaml (verified records). The builder writes citations as "[Surname Year; Surname & Surname Year]"
or narrative "Surname et al. [Year]"; `convert_citations` turns them into Springer name–year form "(Surname et al. Year)" /
"Surname et al. (Year)" and checks that every citation resolves to exactly one record.
"""
import re, unicodedata
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
REFS = yaml.safe_load((HERE / "references.yaml").read_text(encoding="utf-8"))
BY_KEY = {r["key"]: r for r in REFS}


def _ascii(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def surname(author: str) -> str:
    return author.split(",")[0].strip()


def initials(author: str) -> str:
    return author.split(",")[1].strip() if "," in author else ""


def label(r, narrative=False) -> str:
    """Name–year label: 'Schuld 2021', 'Schuld and Killoran 2019', 'Havlíček et al. 2019' (narrative: 'Havlíček et al. (2019)')."""
    a = r["authors"]
    names = surname(a[0]) if len(a) == 1 else (f"{surname(a[0])} and {surname(a[1])}" if len(a) == 2 else f"{surname(a[0])} et al.")
    return f"{names} ({r['year']})" if narrative else f"{names} {r['year']}"


def render(r) -> str:
    """Springer Basic reference-list entry."""
    auth = ", ".join(f"{surname(a)} {initials(a)}".strip() for a in r["authors"])
    s = f"{auth} ({r['year']}) {r['title']}."
    k = r["kind"]
    if k == "article":
        s += f" {r['journal']}"
        if r.get("volume") is not None:
            s += f" {r['volume']}" + (f"({r['issue']})" if r.get("issue") is not None else "") + (f":{r['pages']}" if r.get("pages") else "")
        s += "."
    elif k == "proceedings":
        s += f" In: {r['booktitle']}." if r.get("booktitle") else ""
        if r.get("publisher"): s += f" {r['publisher']}"
        if r.get("pages"): s += f", pp {r['pages']}"
        s += "."
    elif k == "preprint":
        if r.get("arxiv"): s += f" arXiv preprint arXiv:{r['arxiv']}."
    if r.get("note"): s += f" {r['note']}."
    if r.get("doi"): s += f" https://doi.org/{r['doi']}"
    if k != "preprint" and r.get("arxiv"): s += f" (arXiv:{r['arxiv']})"
    return s.replace("..", ".")


def reference_list() -> list:
    """Alphabetical by first author surname, then year, then label (Springer name–year convention)."""
    return sorted(REFS, key=lambda r: (_ascii(surname(r["authors"][0])), r["year"], _ascii(label(r))))


def bibtex(r) -> str:
    typ = {"article": "article", "proceedings": "inproceedings", "preprint": "misc"}[r["kind"]]
    def dotted(ini):  # "SM" -> "S. M.", "H-Y" -> "H.-Y.", "LCL" -> "L. C. L."
        parts = ini.split("-")
        return "-".join(" ".join(c + "." for c in p) for p in parts)
    f = {"author": " and ".join(f"{surname(a)}, {dotted(initials(a))}" for a in r["authors"]),
         "title": "{" + r["title"] + "}", "year": r["year"]}
    if r["kind"] == "article":
        f["journal"] = r["journal"]
        if r.get("volume") is not None: f["volume"] = r["volume"]
        if r.get("issue") is not None: f["number"] = r["issue"]
        if r.get("pages"): f["pages"] = str(r["pages"]).replace("–", "--")
    elif r["kind"] == "proceedings":
        f["booktitle"] = r["booktitle"]
        if r.get("publisher"): f["publisher"] = r["publisher"]
        if r.get("pages"): f["pages"] = str(r["pages"]).replace("–", "--")
    else:
        if r.get("arxiv"): f["eprint"] = r["arxiv"]; f["archivePrefix"] = "arXiv"; f["howpublished"] = f"arXiv:{r['arxiv']}"
    if r.get("doi"): f["doi"] = r["doi"]
    if r.get("note"): f["note"] = r["note"]
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items())
    return f"@{typ}{{{r['key']},\n{body}\n}}"


# ---------------------------------------------------------------- in-text citation conversion
_ITEM = re.compile(r"^(?P<names>.+?)\s(?P<year>\d{4})[a-z]?$")
_NAME = r"[A-Z][\w\-]*"                                   # \w covers accented letters in Python 3 re
_NARR = re.compile(rf"(?P<names>{_NAME}(?:(?:, | & | and ){_NAME})*(?: et al\.)?) \[(?P<year>\d{{4}})\]")
_BRACKET = re.compile(r"\[(?P<body>[^\[\]]*?\b\d{4}[a-z]?(?:; [^\[\]]*?\b\d{4}[a-z]?)*)\]")


def resolve(names: str, year: str):
    first = re.split(r",| & | and | et al", names.strip())[0].strip()
    cands = [r for r in REFS if _ascii(surname(r["authors"][0])) == _ascii(first)]
    exact = [r for r in cands if str(r["year"]) == year]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:  # e.g. Schuld 2021 vs Schuld et al. 2021: disambiguate by the number of names written
        if "et al" in names:
            exact = [r for r in exact if len(r["authors"]) >= 3]
        else:
            n = len([x for x in re.split(r", | & | and ", names.strip()) if x])
            exact = [r for r in exact if len(r["authors"]) == n]
        if len(exact) == 1: return exact[0]
    if len(cands) == 1:   # same first author, different year written in the text -> use the verified record's year
        return cands[0]
    raise KeyError(f"citation '{names} {year}' does not resolve to exactly one reference (candidates: {[c['key'] for c in cands]})")


def convert_citations(text: str):
    """Convert bracket/narrative citations to Springer name–year; return (text, set of cited keys)."""
    cited = set()

    def _narr(m):
        r = resolve(m.group("names"), m.group("year")); cited.add(r["key"]); return label(r, narrative=True)
    text = _NARR.sub(_narr, text)

    def _brk(m):
        items = [i.strip() for i in m.group("body").split(";")]
        out = []
        for it in items:
            mm = _ITEM.match(it)
            if not mm:
                raise KeyError(f"malformed citation item '{it}'")
            r = resolve(mm.group("names"), mm.group("year")); cited.add(r["key"]); out.append(label(r))
        return "(" + "; ".join(out) + ")"
    text = _BRACKET.sub(_brk, text)
    return text, cited
