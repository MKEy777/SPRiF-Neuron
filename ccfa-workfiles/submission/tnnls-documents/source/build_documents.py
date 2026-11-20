from pathlib import Path
import json
import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[4]
MANUSCRIPT = ROOT / "TNNLS-Manuscript"
WORK = Path(__file__).resolve().parents[1]
FACTS = WORK / "source" / "title_page_details.json"


def document_base(title, size=11.5, spacing=1.05):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.8)
    sec.bottom_margin = Inches(0.8)
    sec.left_margin = Inches(0.9)
    sec.right_margin = Inches(0.9)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(size)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    normal.paragraph_format.line_spacing = spacing
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.widow_control = True
    for name in ("Title", "Heading 1", "Heading 2", "List Paragraph"):
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.color.rgb = RGBColor(0, 0, 0)
        for attr in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
            style.element.rPr.rFonts.attrib.pop(qn("w:" + attr), None)
    for border in doc.styles.element.xpath(".//w:pBdr"):
        border.getparent().remove(border)
    doc.core_properties.title = title
    doc.core_properties.author = "Bin Hu"
    doc.core_properties.subject = "Submission to IEEE Transactions on Neural Networks and Learning Systems"
    doc.core_properties.keywords = "SPRiF, spiking neural networks, TNNLS"
    return doc


def add_markdown_runs(p, text):
    for i, chunk in enumerate(re.split(r"\*\*(.*?)\*\*", text)):
        p.add_run(chunk).bold = bool(i % 2)


def build_cover(output_path=None):
    source = (MANUSCRIPT / "Cover_Letter_TNNLS.md").read_text(encoding="utf-8")
    doc = document_base("Cover Letter for SPRiF Neurons", size=12, spacing=1.05)
    blocks = source.strip().split("\n\n")
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines()]
        p = doc.add_paragraph()
        if re.match(r"^[1-3]\. ", lines[0]):
            p.paragraph_format.left_indent = Inches(0.18)
            p.paragraph_format.first_line_indent = Inches(-0.18)
            p.paragraph_format.keep_together = True
        for i, line in enumerate(lines):
            if i:
                p.add_run().add_break()
            add_markdown_runs(p, line)
    path = Path(output_path) if output_path else MANUSCRIPT / "Cover_Letter_TNNLS.docx"
    doc.save(path)
    expected = re.sub(r"\*\*(.*?)\*\*", r"\1", source)
    expected = " ".join(expected.split())
    actual = " ".join(" ".join(p.text for p in Document(path).paragraphs).split())
    assert actual == expected, "Cover letter conversion changed the text"
    return path


def heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12)
    return p


def build_title_page(output_path=None):
    tex = (MANUSCRIPT / "SPRiF_TNNLS.tex").read_text(encoding="utf-8")
    title = re.search(r"\\title\{([^}]+)\}", tex).group(1)
    authors = re.findall(r"([A-Za-z ]+)\\textsuperscript\{([123]),([^}]+)\}", tex)
    # Authors without contribution/correspondence markers use a single affiliation.
    raw_author_block = tex.split(r"\author{", 1)[1].split(r"\thanks", 1)[0]
    authors = re.findall(r"([A-Za-z ]+)\\textsuperscript\{([^}]+)\}", raw_author_block)
    affiliations = re.findall(r"\\thanks\{\\textsuperscript\{([123])\}([^\n]+)\}", tex)
    doc = document_base(title, size=11.5, spacing=1.05)
    p = doc.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(17)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(17)
    r.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(15)
    for i, (name, marks) in enumerate(authors):
        name = name.strip()
        if name.startswith("and "):
            name = name[4:]
        if i == 4:
            p.add_run(",").add_break()
        elif i:
            p.add_run(", " if i < len(authors) - 1 else " and ")
        p.add_run(name)
        mark = p.add_run(marks.replace(r"\textdagger", "\u2020"))
        mark.font.superscript = True

    for number, text in affiliations:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(7)
        r = p.add_run(number)
        r.font.superscript = True
        p.add_run(" " + text)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(1)
    p.add_run("\u2020 ").font.superscript = True
    p.add_run("Jingxin Liu and Yiming Zhang contributed equally to this work.")

    heading(doc, "* Corresponding authors")
    for name, email in (("Lixian Zhu", "zhulx@bit.edu.cn"), ("Bin Hu", "bh@bit.edu.cn")):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.add_run(name).bold = True
        p.add_run("\n" + email)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.add_run(affiliations[0][1])

    heading(doc, "Acknowledgments")
    details = json.loads(FACTS.read_text(encoding="utf-8")) if FACTS.exists() else {}
    doc.add_paragraph(details.get("acknowledgments", ""))
    path = Path(output_path) if output_path else MANUSCRIPT / "Title_Page_TNNLS.docx"
    doc.save(path)
    check = "\n".join(p.text for p in Document(path).paragraphs)
    assert "zhulx@bit.edu.cn" in check and "bh@bit.edu.cn" in check
    assert len(authors) == 7 and len(affiliations) == 3
    assert title in check
    return path


if __name__ == "__main__":
    print(build_title_page())
    print(build_cover())
