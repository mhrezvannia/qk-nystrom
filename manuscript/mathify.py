"""Convert the Unicode mathematical notation of the generated manuscript into LaTeX inline math ($…$).

Applied to the assembled Markdown after citations/floats. Skips YAML header, headings, table rows, code spans, image URLs and
existing $…$ / $$…$$ math. Prose is tokenised; maximal runs of "math tokens" (identifiers, Greek letters, numbers, operators,
brackets, sub/superscripts, decorated symbols) that carry at least one mathematical signal are wrapped in $…$ and translated
symbol by symbol (Greek → \\alpha …, ‖ → \\|, ⟨⟩ → \\langle\\rangle, x′ → x', K̃ → \\tilde K, sub/superscripts → _{}/^{}, words → \\mathrm{}).
Numbers alone, ordinary words and single English articles are left as text. Nothing is renumbered or reworded.
"""
import re

GREEK = {"α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon", "θ": r"\theta", "λ": r"\lambda", "μ": r"\mu",
         "π": r"\pi", "σ": r"\sigma", "τ": r"\tau", "φ": r"\varphi", "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega", "Δ": r"\Delta", "Σ": r"\sum",
         "Φ": r"\Phi", "Ψ": r"\Psi", "Λ": r"\Lambda", "Γ": r"\Gamma"}
OPS = {"=": "=", "≤": r"\le", "≥": r"\ge", "≈": r"\approx", "≪": r"\ll", "≫": r"\gg", "≠": r"\ne", "≡": r"\equiv", "→": r"\to", "∈": r"\in",
       "∝": r"\propto", "×": r"\times", "·": r"\cdot", "−": "-", "-": "-", "+": "+", "±": r"\pm", "<": "<", ">": ">", "/": "/", "~": r"\sim",
       "∼": r"\sim", "⊗": r"\otimes", "∏": r"\prod", "√": r"\sqrt", "∞": r"\infty", "†": r"^\dagger", "≳": r"\gtrsim", "≲": r"\lesssim", "…": r"\ldots"}
SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁺": "+", "⁻": "-", "ⁿ": "n", "ᵀ": r"\mathsf{T}"}
SUB = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "ᵢ": "i", "ⱼ": "j", "ₖ": "k", "ₙ": "n", "ₘ": "m", "₌": "="}
FUNCS = {"cos": r"\cos", "sin": r"\sin", "log": r"\log", "ln": r"\ln", "exp": r"\exp", "min": r"\min", "max": r"\max", "cond": r"\operatorname{cond}",
         "erank": r"\operatorname{erank}", "tr": r"\operatorname{tr}", "diag": r"\operatorname{diag}", "Binomial": r"\operatorname{Binomial}",
         "Var": r"\operatorname{Var}", "RZ": r"\mathrm{RZ}", "RY": r"\mathrm{RY}"}
SUBWORDS = {"min": r"\min", "max": r"\max"}         # X_min -> X_{\min}
WORD_SUBS = {"test", "train", "noisy", "clean", "sub", "eff", "fit", "te", "tr", "val"}   # X_test -> X_{\mathrm{test}}
DECOR = {"Ŵ": r"\hat W", "Ĉ": r"\hat C", "K̂": r"\hat K", "K̃": r"\tilde K", "W̃": r"\tilde W", "Ĥ": r"\hat H"}

MATH_SIGNAL = set(GREEK) | set(SUP) | set(SUB) | set("‖⟨⟩√∏Σ∈≤≥≈≪≫≠≡→∝×·−±⊗†∞^_′") | {"\u0303", "\u0302"}
# a token is one of: decorated symbol, identifier (letters, optionally with Unicode/ASCII sub/superscripts and primes, braces for _{..}),
# number (with thousands commas, decimals, percent, exponent), bracket, operator, whitespace, or a word (anything else)
_TOKEN = re.compile(r"""
    (?P<space>\s+) |
    (?P<num>\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?%?) |
    (?P<hyphword>[A-Za-z]-[A-Za-z][^\s]*) |
    (?P<ident>[A-Za-z][A-Za-z]*(?:[̂̃])?(?:_(?:\{[^}]*\}|[A-Za-z0-9αβγδεθλμπστφχψω]+)|[₀-₉ᵢⱼₖₙₘ₌]+)?(?:\^(?:\{[^}]*\}|[A-Za-z0-9+\-]+)|[⁰¹²³⁴-⁹⁺⁻ⁿᵀ]+)?′*(?![^\W\d_])) |
    (?P<greek>[αβγδεθλμπσταφχψωΔΣΦΨΛΓ](?:_(?:\{[^}]*\}|[A-Za-z0-9αβγδεθλμπστφχψω]+)|[₀-₉ᵢⱼₖₙₘ]+)?(?:\^(?:\{[^}]*\}|[A-Za-z0-9+\-]+)|[⁰¹²³⁴-⁹⁺⁻ⁿᵀ]+)?′*) |
    (?P<subsup>(?:_(?:\{[^}]*\}|[A-Za-z0-9αβγδεθλμπστφχψω]+)|[₀-₉ᵢⱼₖₙₘ₌]+)(?:\^(?:\{[^}]*\}|[A-Za-z0-9+\-]+)|[⁰¹²³⁴-⁹⁺⁻ⁿᵀ]+)?|\^(?:\{[^}]*\}|[A-Za-z0-9+\-]+)|[⁰¹²³⁴-⁹⁺⁻ⁿᵀ]+) |
    (?P<bra>[()\[\]{}|‖⟨⟩]) |
    (?P<op>[=≤≥≈≪≫≠≡→∈∝×·−+±<>/~∼⊗∏√∞†≳≲…](?:_(?:\{[^}]*\}|[A-Za-z0-9]+)|[₀-₉ᵢⱼₖₙₘ₌]+)?(?:\^(?:\{[^}]*\}|[A-Za-z0-9+\-]+)|[⁰¹²³⁴-⁹⁺⁻ⁿᵀ]+)?) |
    (?P<punct>[,;:.]) |
    (?P<word>[^\s]+)
""", re.X)
LONE_PLAIN = {"a", "A", "I", "i", "e"}   # single letters that are English words when standing alone


def _lone_is_var(toks, j, run):
    """'a', 'I' … are variables only next to an operator/bracket: a = π, [0, a], (a, C, θ), = I."""
    prev = j - 1
    while prev >= 0 and toks[prev][0] == "space": prev -= 1
    nxt = j + 1
    while nxt < len(toks) and toks[nxt][0] == "space": nxt += 1
    p_ok = prev >= 0 and (toks[prev][0] == "op" or (toks[prev][0] == "bra" and toks[prev][1] in "([{") or (toks[prev][0] == "punct" and toks[prev][1] == "," and run))
    n_ok = nxt < len(toks) and (toks[nxt][0] == "op" or (toks[nxt][0] == "bra" and toks[nxt][1] in ")]}") or (toks[nxt][0] == "punct" and toks[nxt][1] == "," and run))
    return p_ok or n_ok


def _sub(s):
    if s.startswith("{"): return "_{" + _translate_inner(s[1:-1]) + "}"
    if s in SUBWORDS: return "_{" + SUBWORDS[s] + "}"
    if s in WORD_SUBS or (len(s) > 1 and s.isalpha() and s.islower() and s not in ("ij", "pq", "ii", "jj")): return "_{\\mathrm{" + s + "}}"
    if s in GREEK: return "_" + GREEK[s] + " "
    if any(c in GREEK for c in s): return "_{" + _translate_inner(s) + "}"
    return "_" + s if len(s) == 1 else "_{" + s + "}"


def _sup(s):
    if s.startswith("{"): return "^{" + _translate_inner(s[1:-1]) + "}"
    return "^" + s if len(s) == 1 and s not in "+-" else "^{" + s + "}"


def _translate_inner(s):
    """Translate an already-LaTeX-ish inner string (inside braces): Unicode symbols only."""
    out = []
    for ch in s:
        if ch in GREEK: out.append(GREEK[ch] + " ")
        elif ch in OPS and ch not in "-+/*<>=": out.append(OPS[ch] + " ")
        elif ch == "′": out.append("'")
        elif ch in SUP: out.append("^{" + SUP[ch] + "}")
        elif ch in SUB: out.append("_{" + SUB[ch] + "}")
        else: out.append(ch)
    return "".join(out)


def _scripts(s):
    """Translate a standalone sub/superscript attachment such as '_F²', '₂', '^{-1}', 'ⱼ₌₁ⁿ'."""
    m = re.match(r"(?:_(\{[^}]*\}|[A-Za-z0-9αβγδεθλμπστφχψω]+)|([₀-₉ᵢⱼₖₙₘ₌]+))?(?:\^(\{[^}]*\}|[A-Za-z0-9+\-]+)|([⁰¹²³⁴-⁹⁺⁻ⁿᵀ]+))?$", s)
    sub, usub, sup, usup = m.groups(); out = ""
    if sub: out += _sub(sub)
    if usub: out += "_{" + "".join(SUB[c] for c in usub) + "}" if len(usub) > 1 else "_" + SUB[usub]
    if sup: out += _sup(sup)
    if usup:
        val = "".join(SUP[c] for c in usup); out += "^{" + val + "}" if (len(usup) > 1 or usup in "⁺⁻ᵀ") else "^" + val
    return out


def _ident(tok):
    m = re.match(r"([A-Za-z]+)([\u0302\u0303]?)(?:_(\{[^}]*\}|[A-Za-z0-9αβγδεθλμπστφχψω]+)|([₀-₉ᵢⱼₖₙₘ₌]+))?(?:\^(\{[^}]*\}|[A-Za-z0-9+\-]+)|([⁰¹²³⁴-⁹⁺⁻ⁿᵀ]+))?(′*)$", tok)
    base, dec, sub, usub, sup, usup, primes = m.groups()
    if base + dec in DECOR: core = DECOR[base + dec] + " "
    elif dec: core = ("\\hat " if dec == "\u0302" else "\\tilde ") + base + " "
    elif base in FUNCS: core = FUNCS[base] + " "
    elif len(base) > 1: core = "\\mathrm{" + base + "}"
    else: core = base
    s = core
    if primes: s += "'" * len(primes)
    if sub: s += _sub(sub)
    if usub: s += "_{" + "".join(SUB[c] for c in usub) + "}" if len(usub) > 1 else "_" + SUB[usub]
    if sup: s += _sup(sup)
    if usup:
        val = "".join(SUP[c] for c in usup)
        s += "^{" + val + "}" if (len(usup) > 1 or usup in "⁺⁻ᵀ") else "^" + val
    return s


def _greek(tok):
    m = re.match(r"(.)(?:_(\{[^}]*\}|[A-Za-z0-9αβγδεθλμπστφχψω]+)|([₀-₉ᵢⱼₖₙₘ]+))?(?:\^(\{[^}]*\}|[A-Za-z0-9+\-]+)|([⁰¹²³⁴-⁹⁺⁻ⁿᵀ]+))?(′*)$", tok)
    g, sub, usub, sup, usup, primes = m.groups()
    s = GREEK[g] + " "
    if primes: s += "'" * len(primes)
    if sub: s += _sub(sub)
    if usub: s += "_{" + "".join(SUB[c] for c in usub) + "}" if len(usub) > 1 else "_" + SUB[usub]
    if sup: s += _sup(sup)
    if usup:
        val = "".join(SUP[c] for c in usup); s += "^{" + val + "}" if (len(usup) > 1 or usup in "⁺⁻ᵀ") else "^" + val
    return s


def _num(tok):
    pct = tok.endswith("%"); tok = tok.rstrip("%")
    m = re.match(r"(\d+(?:,\d{3})*(?:\.\d+)?)e(-?\d+)$", tok)
    if m: s = f"{m.group(1)}\\times 10^{{{int(m.group(2))}}}"
    else: s = tok.replace(",", "{,}")
    return s + (r"\%" if pct else "")


def translate_run(tokens):
    """tokens: list of (kind, text). Return LaTeX for a math run."""
    out = []
    for kind, tx in tokens:
        if kind == "space": out.append(" ")
        elif kind == "num": out.append(_num(tx))
        elif kind == "ident": out.append(_ident(tx))
        elif kind == "greek": out.append(_greek(tx))
        elif kind == "subsup": out.append(_scripts(tx))
        elif kind == "supnum":
            val = "".join(SUP[c] for c in tx); out.append("^{" + val + "}")
        elif kind == "bra":
            out.append({"‖": r"\|", "⟨": r"\langle ", "⟩": r"\rangle ", "{": r"\{", "}": r"\}"}.get(tx, tx))
        elif kind == "op":
            base, scripts = tx[0], tx[1:]
            mac = r"\sqrt" if base == "√" else OPS[base]
            out.append(mac + (" " if mac.startswith("\\") else "") + (_scripts(scripts) + " " if scripts else ""))
        elif kind == "punct": out.append(tx)
        else: out.append(tx)
    s = "".join(out)
    s = re.sub(r"\\sqrt ?([A-Za-z0-9]|\\[a-zA-Z]+ ?)", r"\\sqrt{\1}", s)          # √N -> \sqrt{N}
    s = re.sub(r"\^\\dagger ?", r"^{\\dagger}", s)
    s = re.sub(r" +", " ", s)
    s = re.sub(r"(\\[a-zA-Z]+) (?=[_^)\],;'|}/])", r"\1", s)                    # no space before scripts/closing brackets
    s = re.sub(r" +([_^])", r"\1", s)
    s = re.sub(r"\{ ", "{", s).strip()
    return s


def _is_signal(kind, tx):
    if kind in ("greek", "supnum", "subsup"): return True
    if kind == "op" and tx[0] not in ("/", "-", "+"): return True
    if kind == "bra" and tx in "‖⟨⟩": return True
    if kind == "ident" and (any(c in MATH_SIGNAL for c in tx) or "_" in tx or "^" in tx or "′" in tx or tx in FUNCS): return True
    if kind == "bra" and tx in "[]{}": return True
    return False


def _depth(tokens):
    d = 0
    for k, x in tokens:
        if k == "bra" and x in "([{": d += 1
        elif k == "bra" and x in ")]}": d -= 1
    return d


def _balanced(tokens):
    depth = {"(": 0, "[": 0, "{": 0}
    close = {")": "(", "]": "[", "}": "{"}
    for k, t in tokens:
        if k == "bra" and t in depth: depth[t] += 1
        elif k == "bra" and t in close:
            depth[close[t]] -= 1
            if depth[close[t]] < 0: return False
    return all(v == 0 for v in depth.values())


def mathify_text(text):
    """Convert one prose segment (no existing math / code inside)."""
    text = text.replace("Ŵ", "W\u0302").replace("Ĉ", "C\u0302").replace("Ĥ", "H\u0302")
    toks = []
    for m in _TOKEN.finditer(text):
        k, tx = m.lastgroup, m.group(0)
        if k == "hyphword": k = "word"
        if k == "ident" and re.fullmatch(r"[A-Za-z]{2,}", tx) and tx not in FUNCS:
            k = "word"                       # an ordinary English word (math identifiers are single letters or known symbols)
        toks.append((k, tx))
    stack = []
    for idx, (k, tx) in enumerate(toks):
        if k == "bra" and tx in "([{": stack.append((tx, idx))
        elif k == "bra" and tx in ")]}":
            if stack:
                otx, oi = stack.pop()
                if any(toks[q][0] == "word" for q in range(oi + 1, idx)) or (idx == oi + 1):
                    toks[oi] = ("word", otx); toks[idx] = ("word", tx)
            else:
                toks[idx] = ("word", tx)
    for otx, oi in stack: toks[oi] = ("word", otx)
    out = []; i = 0; n = len(toks)
    while i < n:
        k, t = toks[i]
        if k in ("word", "punct", "space"):
            out.append(t); i += 1; continue
        # start of a candidate run: collect math-ish tokens (allow spaces and commas inside)
        j = i; run = []
        while j < n:
            kj, tj = toks[j]
            if kj in ("num", "ident", "greek", "supnum", "subsup", "bra", "op"):
                if kj == "ident" and tj in LONE_PLAIN and not _lone_is_var(toks, j, run):
                    break
                run.append((kj, tj)); j += 1
            elif kj == "space" and run and j + 1 < n and toks[j + 1][0] in ("num", "ident", "greek", "supnum", "subsup", "bra", "op") and not (toks[j + 1][0] == "ident" and toks[j + 1][1] in LONE_PLAIN and not _lone_is_var(toks, j + 1, run)) \
                    and not (run[-1][0] in ("num", "ident", "greek", "subsup", "supnum") and (toks[j + 1][0] == "num" or toks[j + 1] == ("bra", "("))) \
                    and not (run[-1][0] == "bra" and run[-1][1] in ")]}" and (toks[j + 1][0] == "num" or toks[j + 1] == ("bra", "("))):
                run.append((kj, tj)); j += 1
            elif kj == "punct" and tj == "," and run and j + 2 < n and toks[j + 1][0] == "space" and toks[j + 2][0] in ("num", "ident", "greek", "op") and _depth(run) > 0:
                run.append((kj, tj)); j += 1           # comma inside parentheses: K(x, x′)
            else:
                break
        if not run:
            out.append(t); i += 1; continue
        # a lone binary operator between words ("mean ± SD", "task×seed") is set as a symbol
        if len(run) == 1 and run[0][0] == "op" and run[0][1][0] in "×±→·≈≤≥≪≫≠∞":
            out.append("$" + translate_run(run) + "$"); i = j; continue
        # trim: leading/trailing spaces and dangling operators; unbalanced brackets at the edges become plain text
        pre, post = "", ""
        while run and run[0][0] in ("space",): pre += run.pop(0)[1]
        while run and run[0][0] == "op" and run[0][1] in "=/": pre += run.pop(0)[1]
        while run and run[0][0] in ("space",): pre += run.pop(0)[1]
        while run and run[-1][0] in ("space",): post = run.pop()[1] + post
        while run and run[-1][0] == "op" and run[-1][1] in "=≈≤≥<>→·−+/±∈": post = run.pop()[1] + post   # ("40×" keeps its ×)
        while run and run[-1][0] == "space": post = run.pop()[1] + post
        while run and not _balanced(run):
            if run[0][0] == "bra" and run[0][1] in "([{": pre += run.pop(0)[1]
            elif run[-1][0] == "bra" and run[-1][1] in ")]}": post = run.pop()[1] + post
            elif run[-1][0] == "bra" and run[-1][1] in "([{": post = run.pop()[1] + post
            elif run[0][0] == "bra" and run[0][1] in ")]}": pre += run.pop(0)[1]
            elif run[-1][0] in ("punct", "space"): post = run.pop()[1] + post
            elif run[0][0] == "space": pre += run.pop(0)[1]
            else: break
        while run and run[-1][0] == "punct": post = run.pop()[1] + post
        # strip enclosing prose parentheses "(a = π)" -> "(" $a = \pi$ ")"
        if len(run) >= 3 and run[0] == ("bra", "(") and run[-1] == ("bra", ")") and _balanced(run[1:-1]):
            pre += "("; post = ")" + post; run = run[1:-1]
        while run and run[0][0] == "space": pre += run.pop(0)[1]
        while run and run[-1][0] == "space": post = run.pop()[1] + post
        signal = any(_is_signal(k2, t2) for k2, t2 in run) or any(run[q][0] == "ident" and run[q + 1] == ("bra", "(") for q in range(len(run) - 1))
        # a leading binary operator adjacent to plain words (e.g. "tasks × 10 seeds") is emitted alone
        if run and run[0][0] == "op" and run[0][1] in "×·→≈" and not any(k2 in ("ident", "greek") for k2, t2 in run):
            out.append(pre + "$" + OPS[run[0][1]] + "$"); rest = run[1:]
            out.append("".join(t2 for k2, t2 in rest) + post); i = j; continue
        lone_ident = len(run) == 1 and run[0][0] == "ident" and re.fullmatch(r"[A-Za-z]", run[0][1]) and run[0][1] not in LONE_PLAIN
        if run and (signal or lone_ident):
            out.append(pre + "$" + translate_run(run) + "$" + post)
        else:
            out.append(pre + "".join(t2 for k2, t2 in run) + post)
        i = j
    return "".join(out)


_SKIP = re.compile(r"(\$\$.*?\$\$|\$[^$\n]+\$|`[^`]*`|\]\([^)]*\)|\*\*Fig\. \d+\*\*|\*\*Table \d+\*\*|https?://\S+|\bFig\. S?\d+|\bTable S?\d+|\bSect\. S?\d+(?:\.\d+)?|\bRQ\d|\([a-e]\)|\((?:i|ii|iii|iv|v|vi)\)|\bi\.e\.|\be\.g\.|\bvs\.|(?:[A-Z]\.){2,}|\bx86_64\b|\bESM_\d\.\w+|\b[Ss]cenario [A-C]\b|\bF-MNIST\b|\bPCA-8\b|\bL-BFGS-B\b|\bMNIST \d/\d)", re.S)


def mathify_paragraph(p):
    parts = []; pos = 0
    for m in _SKIP.finditer(p):
        parts.append(mathify_text(p[pos:m.start()])); parts.append(m.group(0)); pos = m.end()
    parts.append(mathify_text(p[pos:]))
    return "".join(parts)


def mathify(md):
    head, sep, body = md.partition("\n---\n")          # YAML header
    if not sep: head, sep, body = "", "", md
    body_main, refsep, refs = body.partition("\n## References")
    out = []
    for line in body_main.split("\n"):
        if line.startswith("|") or line.startswith("#") or line.startswith(":::") or line.strip() == "" or line.startswith("$$") or line.startswith("<!--"):
            out.append(line); continue
        if line.startswith("!["):   # image: convert the alt text only
            m = re.match(r"^!\[(.*)\]\((.*)\)(\{.*\})?$", line)
            out.append("![" + mathify_paragraph(m.group(1)) + "](" + m.group(2) + ")" + (m.group(3) or "")); continue
        out.append(mathify_paragraph(line))
    return head + sep + "\n".join(out) + refsep + refs
