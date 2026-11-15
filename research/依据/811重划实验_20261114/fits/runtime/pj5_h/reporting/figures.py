"""Small helper that turns tidy CSV tables into \\input-able pgfplots figures.

matplotlib crashes the interpreter in this environment (broken BLAS above ~4000 elements),
so every figure in this project is emitted as a pgfplots/tikzpicture snippet plus the CSV it
reads.  That keeps the figures vector, editable and reproducible from the same run directory.
"""
from __future__ import annotations

from pathlib import Path


def _axis_options(width, height, xlabel, ylabel, title, logx=False, logy=False,
                  legend_pos="north east", ymin=None, ymax=None):
    opts = [f"width={width}", f"height={height}", f"xlabel={{{xlabel}}}",
            f"ylabel={{{ylabel}}}", f"title={{{title}}}",
            "legend style={font=\\scriptsize,at={(" + legend_pos.split()[0].replace("north", "0.98").replace("south", "0.02")
            + "," + legend_pos.split()[1].replace("east", "0.98").replace("west", "0.02") + ")},anchor=" + legend_pos + "}",
            "grid=major", "grid style={gray!20}"]
    if logx:
        opts.append("xmode=log")
    if logy:
        opts.append("ymode=log")
    if ymin is not None:
        opts.append(f"ymin={ymin}")
    if ymax is not None:
        opts.append(f"ymax={ymax}")
    return ",\n  ".join(opts)


def pgfplots_multiline(csv_name: str, xcol: str, ycols: list[tuple[str, str]],
                       tex_path: Path, xlabel: str = "$x$", ylabel: str = "$y$",
                       title: str = "", width: str = "0.48\\linewidth", height: str = "5.2cm",
                       logx: bool = False, logy: bool = False, mark: str = "none") -> None:
    """One axis, several curves taken from columns of ``csv_name`` (same directory)."""
    lines = [
        "% Auto-generated -- do not edit by hand.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[" + _axis_options(width, height, xlabel, ylabel, title, logx, logy) + "]",
    ]
    for col, label in ycols:
        lines.append(f"\\addplot+[mark={mark},thick] table[x={xcol},y={col},col sep=comma] {{{csv_name}}};")
        lines.append(f"\\addlegendentry{{{label}}}")
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    Path(tex_path).write_text("\n".join(lines), encoding="utf-8")


def pgfplots_panels(panels: list[dict], tex_path: Path, ncols: int = 2) -> None:
    """Several side-by-side axes.  Each panel dict holds the arguments of pgfplots_multiline."""
    out = ["% Auto-generated -- do not edit by hand."]
    for i, p in enumerate(panels):
        if i % ncols == 0:
            out.append("\\begin{tikzpicture}[baseline]")
        else:
            out.append("\\begin{tikzpicture}[baseline]")
        out.append("\\begin{axis}[" + _axis_options(
            p.get("width", "0.47\\linewidth"), p.get("height", "5.0cm"),
            p.get("xlabel", "$x$"), p.get("ylabel", "$y$"), p.get("title", ""),
            p.get("logx", False), p.get("logy", False)) + "]")
        for col, label in p["ycols"]:
            out.append(f"\\addplot+[mark={p.get('mark','none')},thick] "
                       f"table[x={p['xcol']},y={col},col sep=comma] {{{p['csv']}}};")
            out.append(f"\\addlegendentry{{{label}}}")
        out += ["\\end{axis}", "\\end{tikzpicture}"]
        if i % ncols == ncols - 1 or i == len(panels) - 1:
            if p.get("caption"):
                out.append("\\par\\smallskip\\footnotesize " + p["caption"])
            out.append("\\par\\medskip")
    Path(tex_path).write_text("\n".join(out), encoding="utf-8")
