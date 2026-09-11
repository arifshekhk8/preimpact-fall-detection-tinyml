#!/usr/bin/env python3
"""Produce a Word version of the IEEE paper from the LaTeX source.

    paper/ieee/PreImpact_Fall_Detection_IEEE.docx

LaTeX remains the source of truth; this is a convenience copy for anyone who needs
to read or comment in Word. Pandoc does the text, tables and numeric IEEE citations.
Everything that pandoc cannot carry across is reapplied here:

  * the TikZ method figure, which is pre-rendered to PNG (Word cannot draw TikZ);
  * the five-author block, stripped before conversion because pandoc mangles
    IEEEtran's author macros;
  * two-column body layout, with the title, authors and abstract spanning both
    columns as IEEE requires;
  * Times New Roman throughout at IEEE's sizes.

Run:  python scripts/build_paper_docx.py
"""

from __future__ import annotations

import copy
import re
import shutil
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "ieee"
OUT = TEX / "PreImpact_Fall_Detection_IEEE.docx"

TITLE = ("Pre-Impact Fall Detection on a Low-Cost Microcontroller: "
         "Cross-Dataset Generalisation, Domain Adaptation and INT8 Deployment")

AUTHORS = [
    ("Md. Arif Shekh", "23103022"),
    ("Md. Meherab Hossain Talukder", "23103032"),
    ("Tasfia Islam Prapty", "23103286"),
    ("G. M. Imtiaz Dinar", "23103080"),
    ("Tasnia Chowdhury Toshita", "23103038"),
]
DEPT = "Department of Computer Science and Engineering"
UNIV = "IUBAT\u2014International University of Business Agriculture and Technology"
CITY = "Dhaka, Bangladesh"
CORRESPONDING = "Corresponding author: arifshekh.k8@gmail.com"


# ------------------------------------------------------------------- helpers
def style_run(run, size=10, bold=False, italic=False, font="Times New Roman"):
    run.font.name = font
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs"):
        rFonts.set(qn(a), font)
    return run


def set_columns(section, n: int, space_twips: int = 360) -> None:
    """Word stores column count on the section's sectPr.

    A document that has never been multi-column carries no <w:cols> element at all,
    so it is created when absent rather than assumed to exist.
    """
    sectPr = section._sectPr
    found = sectPr.xpath("./w:cols")
    if found:
        cols = found[0]
    else:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(n))
    cols.set(qn("w:space"), str(space_twips))
    cols.set(qn("w:equalWidth"), "1")



def set_section_continuous(sectPr) -> None:
    """Make a section start on the same page as the previous one.

    The break type lives in the sectPr of the section being STARTED, not the one
    being ended. Omitting it defaults to nextPage, which is what was forcing the
    two-column body onto its own page. OOXML also fixes the child order, so w:type
    must land after footnotePr/endnotePr but before pgSz.
    """
    existing = sectPr.find(qn("w:type"))
    if existing is not None:
        existing.set(qn("w:val"), "continuous")
        return
    typ = OxmlElement("w:type")
    typ.set(qn("w:val"), "continuous")
    anchor = None
    for tag in ("w:endnotePr", "w:footnotePr"):
        found = sectPr.find(qn(tag))
        if found is not None:
            anchor = found
            break
    if anchor is not None:
        anchor.addnext(typ)
    else:
        sectPr.insert(0, typ)


# ------------------------------------------------------------- figure render
def render_method_figure() -> Path:
    """TikZ -> PDF -> PNG, because Word cannot draw TikZ."""
    png = TEX / "fig_method.png"
    standalone = TEX / "_figstandalone.tex"
    standalone.write_text(
        "\\documentclass[border=4pt]{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds}\n"
        "\\begin{document}\n\\input{fig_method}\n\\end{document}\n"
    )
    subprocess.run(["tectonic", "-X", "compile", standalone.name],
                   cwd=TEX, capture_output=True, timeout=600)
    pdf = TEX / "_figstandalone.pdf"
    if pdf.exists():
        import pymupdf
        page = pymupdf.open(pdf)[0]
        page.get_pixmap(dpi=400).save(png)
    for f in (standalone, pdf):
        f.unlink(missing_ok=True)
    return png


# -------------------------------------------------------------- tex -> docx
def run_pandoc() -> Path:
    src = (TEX / "main.tex").read_text()

    src = src.replace(r"\input{fig_method}",
                      r"\includegraphics[width=\textwidth]{fig_method.png}")
    src = re.sub(r"\\usepackage\{tikz\}\n\\usetikzlibrary\{[^}]*\}\n", "", src)
    src = re.sub(r"%% Lets a five-author block.*?\\makeatother\n\n", "", src, flags=re.S)
    # pandoc mangles IEEEtran's author macros; the block is rebuilt in Word below
    src = re.sub(r"\\author\{.*?\n\}\n\n(?=\\maketitle)", "", src, flags=re.S)

    tmp = TEX / "_pandoc.tex"
    tmp.write_text(src)

    csl = TEX / "ieee.csl"
    cmd = ["pandoc", tmp.name, "-o", "_raw.docx", "--citeproc",
           "--bibliography=refs.bib"]
    if csl.exists():
        cmd.append(f"--csl={csl.name}")
    proc = subprocess.run(cmd, cwd=TEX, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    tmp.unlink(missing_ok=True)
    return TEX / "_raw.docx"



def polish(doc) -> None:
    """Final IEEE touches pandoc cannot know about."""
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

    # 1. Hyperlink runs live inside <w:hyperlink> and are not reachable via
    #    paragraph.runs, which is why they stayed blue after the earlier pass.
    for para in doc.paragraphs:
        for link in para._element.findall(qn("w:hyperlink")):
            for r in link.findall(qn("w:r")):
                rPr = r.find(qn("w:rPr"))
                if rPr is None:
                    rPr = OxmlElement("w:rPr")
                    r.insert(0, rPr)
                for tag in ("w:color", "w:u"):
                    for old in rPr.findall(qn(tag)):
                        rPr.remove(old)
                col = OxmlElement("w:color")
                col.set(qn("w:val"), "000000")
                rPr.append(col)

    # 2. Pandoc numbers cross-references decimally (5.2); the body now uses IEEE's
    #    Roman section / letter subsection form, so rewrite them to match.
    #
    #    ONLY the hyperlink fields pandoc generates for \ref are touched. An earlier
    #    version rewrote any \d+\.\d+ found in ordinary runs and corrupted every
    #    decimal number in the paper -- 0.950 became "X-e", 2.0 s became "II-@ s".
    #    Cross-references are always hyperlinks here, so that is the safe anchor.
    def to_ieee(m):
        sec, sub = int(m.group(1)), int(m.group(2))
        if sec > len(roman) or not (1 <= sub <= 26):
            return m.group(0)
        return f"{roman[sec - 1]}-{chr(ord('A') + sub - 1)}"

    for para in doc.paragraphs:
        for link in para._element.findall(qn("w:hyperlink")):
            for t in link.iter(qn("w:t")):
                if t.text and re.fullmatch(r"\s*\d+\.\d+\s*", t.text):
                    t.text = re.sub(r"(\d+)\.(\d+)", to_ieee, t.text)

    # 3. Index Terms label, which \begin{IEEEkeywords} carries in LaTeX.
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip().startswith("fall detection, pre-impact"):
            if para.runs:
                para.runs[0].text = "Index Terms\u2014" + para.runs[0].text
                style_run(para.runs[0], 9, bold=True, italic=True)
                for r in para.runs[1:]:
                    style_run(r, 9, bold=False, italic=True)
            break

    # 4. Figure captions: pandoc emits them with the "Image Caption" style but
    #    without the "Fig. N." prefix IEEE expects.
    fig_n = 0
    for para in doc.paragraphs:
        if (para.style.name or "") != "Image Caption" or not para.text.strip():
            continue
        fig_n += 1
        if para.runs:
            para.runs[0].text = f"Fig. {fig_n}. " + para.runs[0].text
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.space_after = Pt(6)
        for r in para.runs:
            style_run(r, 8, bold=False, italic=False)
            r.font.color.rgb = RGBColor(0, 0, 0)


# ------------------------------------------------------------ IEEE dressing
def apply_ieee_layout(raw: Path) -> None:
    doc = Document(raw)
    body = doc.element.body

    for s in doc.sections:
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        s.left_margin = s.right_margin = Inches(0.625)
        s.top_margin = Inches(0.75)
        s.bottom_margin = Inches(1.0)

    # Pandoc emits the title as the first paragraph; drop it, we rebuild the head.
    if doc.paragraphs and doc.paragraphs[0].text.strip().startswith("Pre-Impact"):
        doc.paragraphs[0]._element.getparent().remove(doc.paragraphs[0]._element)

    # --- single-column head, inserted before everything else ---
    head = []

    def head_para(text, size, bold=False, italic=False, after=4,
                  align=WD_ALIGN_PARAGRAPH.CENTER):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(after)
        p.paragraph_format.space_before = Pt(0)
        if text:
            style_run(p.add_run(text), size, bold, italic)
        head.append(p._element)
        return p

    head_para(TITLE, 20, bold=False, after=10)
    marks = "\u002a\u2020\u2021\u00a7\u00b6"
    head_para(", ".join(f"{n}{marks[i]}" for i, (n, _) in enumerate(AUTHORS)),
              11, after=2)
    head_para(DEPT, 10, italic=True, after=1)
    head_para(UNIV, 10, italic=True, after=1)
    head_para(CITY, 10, after=2)
    head_para("   ".join(f"{marks[i]}{sid}" for i, (_, sid) in enumerate(AUTHORS)),
              9, after=1)
    head_para(CORRESPONDING, 9, after=10)

    for el in reversed(head):
        body.insert(0, el)

    # --- split: head stays one column, body becomes two ---
    # OOXML fixes the child order inside <w:sectPr>: w:type must precede w:cols.
    # Getting that wrong makes Word treat a continuous break as a page break, which
    # left the whole first page blank.
    marker = None
    for p in doc.paragraphs:
        if p.text.strip().lower().startswith("abstract"):
            marker = p
            break
    if marker is not None:
        src = doc.sections[-1]._sectPr
        brk = OxmlElement("w:p")
        pPr = OxmlElement("w:pPr")
        sectPr = OxmlElement("w:sectPr")

        typ = OxmlElement("w:type")
        typ.set(qn("w:val"), "continuous")
        sectPr.append(typ)
        for tag in ("w:pgSz", "w:pgMar"):
            found = src.xpath(f"./{tag}")
            if found:
                sectPr.append(copy.deepcopy(found[0]))
        cols = OxmlElement("w:cols")
        cols.set(qn("w:num"), "1")
        cols.set(qn("w:space"), "360")
        sectPr.append(cols)

        pPr.append(sectPr)
        brk.append(pPr)
        marker._element.addprevious(brk)

    set_columns(doc.sections[-1], 2)
    set_section_continuous(doc.sections[-1]._sectPr)

    # --- typography ---
    # Pandoc styles headings blue and links blue; IEEE wants everything black, with
    # sections numbered in Roman and subsections in letters.
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    sec_i, sub_i = 0, 0

    for p in doc.paragraphs:
        if p._element in head:
            continue
        name = (p.style.name or "").lower()
        txt = p.text.strip()

        if name == "heading 1":
            sec_i += 1
            sub_i = 0
            label = f"{roman[min(sec_i, len(roman)) - 1]}. "
            if p.runs:
                p.runs[0].text = label + p.runs[0].text
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            size, bold, italic = 10, False, False
            p.paragraph_format.space_before = Pt(8)
        elif name == "heading 2":
            sub_i += 1
            label = f"{chr(ord('A') + sub_i - 1)}. "
            if p.runs:
                p.runs[0].text = label + p.runs[0].text
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            size, bold, italic = 10, False, True
            p.paragraph_format.space_before = Pt(6)
        elif txt.lower().startswith("abstract"):
            size, bold, italic = 9, True, False
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        else:
            size, bold, italic = 10, False, False
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.0
        for r in p.runs:
            style_run(r, size, bold or r.bold, italic or r.italic)
            r.font.color.rgb = RGBColor(0, 0, 0)

    # small caps heading look is not worth chasing; black text is the thing that
    # actually matters for print

    for t in doc.tables:
        t.autofit = True
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    for r in p.runs:
                        style_run(r, 8, r.bold, r.italic)
                        r.font.color.rgb = RGBColor(0, 0, 0)

    for shape in doc.inline_shapes:
        if shape.width > Inches(3.4):
            ratio = shape.height / shape.width
            shape.width = Inches(3.35)
            shape.height = int(Inches(3.35) * ratio)

    polish(doc)

    doc.save(OUT)
    raw.unlink(missing_ok=True)


def export_pdf(path: Path) -> Path | None:
    exe = next((c for c in ("/Applications/LibreOffice.app/Contents/MacOS/soffice",
                            "soffice", "libreoffice")
                if Path(c).exists() or shutil.which(c)), None)
    if exe is None:
        return None
    subprocess.run([exe, "--headless", "--convert-to", "pdf",
                    "--outdir", str(path.parent), str(path)],
                   capture_output=True, timeout=600)
    pdf = path.with_suffix(".pdf")
    return pdf if pdf.exists() else None


def main() -> int:
    png = render_method_figure()
    print(f"method figure: {png.name} ({'ok' if png.exists() else 'MISSING'})")
    raw = run_pandoc()
    apply_ieee_layout(raw)

    d = Document(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  paragraphs {len(d.paragraphs)}  tables {len(d.tables)}  "
          f"images {len(d.inline_shapes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
