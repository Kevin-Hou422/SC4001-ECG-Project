"""Convert docs/report/REPORT_DRAFT.md into a self-contained LaTeX project.

Output: docs/report/latex/{main.tex, references.bib, figures/*.pdf}. Compile with
XeLaTeX and biber: locally ``latexmk -xelatex main.tex`` in that folder, or upload
the folder to Overleaf and set the compiler to XeLaTeX. Arial is used when it is
installed, otherwise TeX Gyre Heros (a close Helvetica-style substitute).

The draft uses a small Markdown subset: '#'/'##' headings, paragraphs, '- '
lists, pipe tables preceded by a '**Table N.** caption' paragraph, figures as
'![caption](a.png b.png){width=inches}', citations as [Key, Key] with keys from
REFERENCES_VERIFIED.md, '> ' blocks for sections another member still has to
write (printed as a highlighted box), a '\\newpage' line, and HTML comments,
which are dropped except for the <!-- cover --> block and the <!-- references -->
marker. 'Table N' and 'Fig. N' in the text become \\ref links to the captions
numbered N in the draft, so LaTeX renumbers them when tables or figures are added.

Once the team edits main.tex directly, that file is the master copy and this
script should not be rerun over it.
Run from the repository root: python tools/build_report_tex.py
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/report"
SOURCE = REPORT / "REPORT_DRAFT.md"
OUT = REPORT / "latex"
# Draft keys that the shared references.bib spells differently.
BIB_ALIASES = {"MITDB": "moody2005mitbih", "deChazal2004": "dechazal2004automatic",
               "Nonaka2020": "nonaka2020augmentation", "Raghu2022": "raghu2022augmentation"}
SUPERSCRIPTS = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
LINE_WIDTH_PT = 482  # A4 with 2 cm margins
CHAR_PT = 4.6        # average character width of \small Arial, for table layout

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{fontspec}
\IfFontExistsTF{Arial}{\setmainfont{Arial}}{\setmainfont{TeX Gyre Heros}}
\IfFontExistsTF{Consolas}{\setmonofont[Scale=0.9]{Consolas}}{}
\usepackage{microtype}
\usepackage{graphicx,booktabs,tabularx,array}
\usepackage[font=small,labelfont=bf,skip=4pt]{caption}
\usepackage{enumitem}
\setlist{nosep,leftmargin=1.4em}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection}{0.6em}{}
\titlespacing*{\section}{0pt}{9pt}{3pt}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.6em}{}
\titlespacing*{\subsection}{0pt}{6pt}{2pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlength{\tabcolsep}{3.5pt}
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{6pt plus 2pt minus 2pt}
\setlength{\intextsep}{6pt plus 2pt minus 2pt}
\usepackage{csquotes}
\usepackage[backend=biber,style=ieee,sorting=none,url=false,isbn=false]{biblatex}
\addbibresource{references.bib}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
% A section another member still has to write; delete the box when it is done.
\newcommand{\placeholder}[1]{\par\noindent\fcolorbox{orange}{orange!8}{%
  \parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{\small #1}}\par}
"""


def bib_keys():
    text = (REPORT / "references.bib").read_text(encoding="utf-8")
    return set(re.findall(r"^@\w+\{([^,\s]+),", text, flags=re.M))


KEYS = bib_keys()
LABELS = set()  # caption numbers present in the draft, e.g. {"tab:2", "fig:C1"}


def escape(text):
    out = []
    for ch in text:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        else:
            out.append(ch)
    return "".join(out)


def inline(text):
    """Markdown inline markup and citations -> LaTeX."""
    held = []

    def hold(latex):
        held.append(latex)
        return f"\x00{len(held) - 1}\x00"

    def cite(match):
        keys = [k.strip() for k in match.group(1).split(",")]
        mapped = [BIB_ALIASES.get(k, k) for k in keys]
        if all(k in KEYS for k in mapped):
            return hold(r"~\cite{" + ",".join(mapped) + "}")
        return match.group(0)

    def ref(match):
        label = ("tab:" if match.group(1) == "Table" else "fig:") + match.group(2)
        if label not in LABELS:
            return match.group(0)
        return hold(("Table" if label.startswith("tab") else "Fig.") + r"~\ref{" + label + "}")

    text = re.sub(r"`([^`]+)`", lambda m: hold(r"\texttt{" + escape(m.group(1)) + "}"), text)
    text = re.sub(r"\s*\[([A-Za-z]\w*(?:,\s*[A-Za-z]\w*)*)\]", cite, text)
    text = re.sub(r"\b(Table|Fig\.) ([A-Z]?\d+)\b", ref, text)
    text = re.sub(r"(\d)([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
                  lambda m: m.group(1) + hold("$^{" + m.group(2).translate(SUPERSCRIPTS) + "}$"), text)
    text = text.replace(r"\*", hold(r"\textsuperscript{*}"))
    # Keep an interval such as [-0.135, +0.042] on one line.
    text = re.sub(r"\[([−+-]?[\d.]+), ([−+-]?[\d.]+)\]",
                  lambda m: "[" + m.group(1) + "," + hold("~") + m.group(2) + "]", text)
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", text)
    text = re.sub(r'"([^"]+)"', "“\\1”", text)
    text = text.replace(" ± ", r"\,±\,")
    return re.sub("\x00(\\d+)\x00", lambda m: held[int(m.group(1))], text)


def strip_number(caption, word):
    return re.sub(rf"^\*\*{word} [A-Z]?\d+\.\*\*\s*", "", caption.strip())


def caption_label(caption, word, prefix):
    """\\label for a '**Table 2.**'-style caption, matching the 'Table 2' text links."""
    number = re.match(rf"^\*\*{word} ([A-Z]?\d+)\.\*\*", caption.strip())
    return r"\label{" + prefix + number.group(1) + "}" if number else ""


def table(rows, caption):
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    ncol = len(cells[0])
    widths = [max(len(r[i]) for r in cells) for i in range(ncol)]
    estimate = sum(widths) * CHAR_PT + ncol * 7
    if estimate > LINE_WIDTH_PT:
        spec = "".join(r">{\raggedright\arraybackslash}X" if w > 16 else "l" for w in widths)
        begin, end = r"\begin{tabularx}{\linewidth}{" + spec + "}", r"\end{tabularx}"
    else:
        begin, end = r"\begin{tabular}{" + "l" * ncol + "}", r"\end{tabular}"
    head = " & ".join(r"\textbf{" + inline(c) + "}" for c in cells[0]) + r" \\"
    body = [" & ".join(inline(c) for c in r) + r" \\" for r in cells[1:]]
    return "\n".join([r"\begin{table}[!htb]", r"\centering\small",
                      r"\caption{" + inline(strip_number(caption, "Table")) + "}"
                      + caption_label(caption, "Table", "tab:"),
                      begin, r"\toprule", head, r"\midrule", *body, r"\bottomrule", end,
                      r"\end{table}"])


def figure(line):
    match = re.match(r"!\[(.*)\]\((.*?)\)(?:\{width=([\d.]+)\})?\s*$", line)
    caption, paths, width = match.group(1), match.group(2).split(), match.group(3) or "6.3"
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    graphics = []
    for p in paths:
        src = REPORT / p
        pdf = src.with_suffix(".pdf")
        src = pdf if pdf.is_file() else src
        shutil.copy2(src, OUT / "figures" / src.name)
        graphics.append(rf"\includegraphics[width={width}in]{{figures/{src.name}}}")
    return "\n".join([r"\begin{figure}[!htb]", r"\centering", r"\hspace{0.15in}".join(graphics),
                      r"\caption{" + inline(strip_number(caption, "Figure")) + "}"
                      + caption_label(caption, "Figure", "fig:"), r"\end{figure}"])


def cover(lines):
    title, subtitle, course, group, *members, date = lines
    return "\n".join([
        r"\begin{titlepage}", r"\centering", r"\vspace*{3.5cm}",
        r"{\LARGE\bfseries " + inline(title) + r"\par}", r"\vspace{0.4cm}",
        r"{\Large " + inline(subtitle) + r"\par}", r"\vspace{2cm}",
        r"{\large " + inline(course) + r"\par}", r"\vspace{0.6cm}",
        r"{\large\bfseries " + inline(group) + r"\par}", r"\vspace{1.2cm}",
        *[inline(m) + r"\par\vspace{0.15cm}" for m in members],
        r"\vfill", inline(date), r"\end{titlepage}", r"\setcounter{page}{1}"])


def convert(markdown):
    cover_block = re.search(r"<!-- cover -->\n(.*?)\n<!-- /cover -->", markdown, re.S)
    cover_lines = cover_block.group(1)
    markdown = markdown.replace(cover_block.group(0), "")
    markdown = markdown.replace("<!-- references -->", "@@REFERENCES@@")
    markdown = re.sub(r"<!--.*?-->", "", markdown, flags=re.S)
    LABELS.update("tab:" + n for n in re.findall(r"\*\*Table ([A-Z]?\d+)\.\*\*", markdown))
    LABELS.update("fig:" + n for n in re.findall(r"\*\*Figure ([A-Z]?\d+)\.\*\*", markdown))
    blocks = re.split(r"\n\s*\n", markdown)
    out, appendix, pending_caption = [cover(cover_lines.strip().splitlines())], False, None
    for block in blocks:
        lines = [l for l in block.strip("\n").splitlines() if l.strip()]
        if not lines:
            continue
        first = lines[0]
        if pending_caption is not None and not first.startswith("|"):
            out.append(inline(pending_caption))
            pending_caption = None
        if first.startswith("# "):
            title = first[2:].strip()
            number = re.match(r"(\d+)\s+(.*)", title)
            if title == "Appendix":
                appendix = True
                out += [r"\clearpage", r"\appendix",
                        r"\renewcommand{\thefigure}{\thesection\arabic{figure}}",
                        r"\renewcommand{\thetable}{\thesection\arabic{table}}",
                        r"\counterwithin*{figure}{section}", r"\counterwithin*{table}{section}"]
            elif number:
                out.append(r"\section{" + inline(number.group(2)) + "}")
            else:
                out.append(r"\section*{" + inline(title) + "}")
        elif first.startswith("## "):
            title = re.sub(r"^(\d+\.\d+|[A-Z])\s+", "", first[3:].strip())
            out.append((r"\section{" if appendix else r"\subsection{") + inline(title) + "}")
        elif first.strip() == "@@REFERENCES@@":
            out.append(r"\printbibliography[heading=none]")
        elif first.strip() == r"\newpage":
            out.append(r"\clearpage")
        elif first.startswith("!["):
            out += [figure(l) for l in lines]
        elif first.startswith("|"):
            out.append(table(lines, pending_caption or ""))
            pending_caption = None
        elif first.startswith("- "):
            items, current = [], []
            for line in lines:
                if line.startswith("- "):
                    if current:
                        items.append(" ".join(current))
                    current = [line[2:].strip()]
                else:
                    current.append(line.strip())
            items.append(" ".join(current))
            # Monospaced spaces cannot stretch, so lists with commands are not justified.
            ragged = r"\raggedright" if any("`" in i for i in items) else ""
            out.append("\n".join([r"{" + ragged + r"\begin{itemize}",
                                  *[r"\item " + inline(i) for i in items], r"\end{itemize}\par}"]))
        elif first.startswith("**Table"):
            pending_caption = " ".join(l.strip() for l in lines)
        elif first.startswith("> "):
            out.append(r"\placeholder{" + inline(" ".join(l[2:].strip() for l in lines)) + "}")
        else:
            out.append(inline(" ".join(l.strip() for l in lines)))
    return PREAMBLE + "\\begin{document}\n\n" + "\n\n".join(out) + "\n\n\\end{document}\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main.tex").write_text(convert(SOURCE.read_text(encoding="utf-8")), encoding="utf-8")
    shutil.copy2(REPORT / "references.bib", OUT / "references.bib")
    print(f"Wrote {OUT / 'main.tex'}; compile there with: latexmk -xelatex main.tex")


if __name__ == "__main__":
    main()
