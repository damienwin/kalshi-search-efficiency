"""Render Markdown docs into one PDF for review.

    python scripts/make_pdf.py --out docs/reports/progress_1_full.pdf \
        docs/reports/progress_1.md docs/DATA_CARD.md docs/SPRING_ARTIFACT.md

Supports the subset this repo's docs use: headings, paragraphs, bullet and
numbered lists, tables, fenced code, blockquotes, rules, and inline
bold/italic/code/links.
"""

import argparse
import os
import re
import sys
from datetime import date
from html import escape

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, KeepTogether, ListFlowable,
                                ListItem, PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)

INK = colors.HexColor("#1B2733")
ACCENT = colors.HexColor("#116B71")
MUTED = colors.HexColor("#5B6B77")
RULE = colors.HexColor("#D3DDE2")
BAND = colors.HexColor("#EEF4F5")

S = {
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.7, leading=14, textColor=INK,
                           alignment=TA_LEFT, spaceAfter=7),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=INK,
                         spaceBefore=6, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=ACCENT,
                         spaceBefore=14, spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=INK,
                         spaceBefore=10, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.6, leading=11.6, textColor=INK),
    "cellhead": ParagraphStyle("cellhead", fontName="Helvetica-Bold", fontSize=8.6, leading=11.6,
                               textColor=colors.white),
    "quote": ParagraphStyle("quote", fontName="Helvetica-Oblique", fontSize=9.4, leading=13.5,
                            textColor=MUTED, leftIndent=12, spaceAfter=7),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.2, leading=10.5, textColor=INK),
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=11, leading=15, textColor=MUTED),
}


# Helvetica's WinAnsi encoding has no Greek or superscripts: map them, don't drop them.
GLYPH = {"Δ": "delta ", "λ": "lambda ", "τ": "tau ", "α": "alpha ", "σ": "sigma ", "μ": "mu "}
SUPER = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
         "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+"}


def sanitize(text: str) -> str:
    for k, v in GLYPH.items():
        text = text.replace(k, v)
    out, run = [], []
    for ch in text:
        if ch in SUPER:
            run.append(SUPER[ch])
            continue
        if run:
            out.append(f"<super>{''.join(run)}</super>")
            run = []
        out.append(ch)
    if run:
        out.append(f"<super>{''.join(run)}</super>")
    return "".join(out)


def inline(tokens) -> str:
    """Markdown inline tokens -> reportlab mini-HTML."""
    out, link = [], None
    for t in tokens:
        if t.type == "text":
            out.append(sanitize(escape(t.content)))
        elif t.type == "code_inline":
            out.append(f'<font face="Courier" size="8.6">{escape(t.content)}</font>')
        elif t.type == "strong_open":
            out.append("<b>")
        elif t.type == "strong_close":
            out.append("</b>")
        elif t.type in ("em_open", "em_close"):
            out.append("<i>" if t.type == "em_open" else "</i>")
        elif t.type == "link_open":
            link = dict(t.attrs).get("href", "")
            out.append(f'<link href="{escape(link)}" color="#116B71">')
        elif t.type == "link_close":
            out.append("</link>")
        elif t.type == "softbreak":
            out.append(" ")
        elif t.type == "html_inline":
            out.append(escape(t.content))
    return "".join(out)


def table_flowable(rows, width):
    head, body = rows[0], rows[1:]
    ncols = len(head)
    data = [[Paragraph(c, S["cellhead"]) for c in head]]
    data += [[Paragraph(c, S["cell"]) for c in r] + [Paragraph("", S["cell"])] * (ncols - len(r)) for r in body]
    first = max(1.0, min(2.2, max((len(r[0]) for r in rows), default=10) / 22))
    rest = (ncols - 1) or 1
    widths = [width * first / (first + rest)] + [width * 1 / (first + rest)] * rest
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
    ]))
    return t


def match_close(tokens, i: int) -> int:
    """Index of the token closing the block opened at i."""
    open_type, close_type = tokens[i].type, tokens[i].type.replace("_open", "_close")
    depth = 0
    for j in range(i, len(tokens)):
        if tokens[j].type == open_type:
            depth += 1
        elif tokens[j].type == close_type:
            depth -= 1
            if depth == 0:
                return j
    return len(tokens) - 1


def render_tokens(tokens, width: float, quote: bool = False) -> list:
    flow, i = [], 0
    list_stack = ["quote"] if quote else []
    while i < len(tokens):
        t = tokens[i]
        if t.type == "heading_open":
            text = inline(tokens[i + 1].children or [])
            flow.append(Paragraph(text, S.get(t.tag, S["h3"])))
            if t.tag == "h2":
                flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceAfter=6))
            i += 3
            continue
        if t.type == "paragraph_open":
            text = inline(tokens[i + 1].children or [])
            style = S["quote"] if list_stack and list_stack[-1] == "quote" else S["body"]
            flow.append(Paragraph(text, style))
            i += 3
            continue
        if t.type in ("bullet_list_open", "ordered_list_open"):
            end = match_close(tokens, i)
            items, j = [], i + 1
            while j < end:
                if tokens[j].type == "list_item_open":
                    item_end = match_close(tokens, j)
                    inner = render_tokens(tokens[j + 1:item_end], width - 14)
                    items.append(ListItem(inner or [Spacer(1, 1)], leftIndent=14))
                    j = item_end + 1
                    continue
                j += 1
            flow.append(ListFlowable(items, bulletType="1" if t.type == "ordered_list_open" else "bullet",
                                     bulletFontSize=7, leftIndent=14, spaceAfter=4))
            i = end + 1
            continue
        if t.type == "table_open":
            rows, j = [], i
            while j < len(tokens) and tokens[j].type != "table_close":
                if tokens[j].type in ("th_open", "td_open"):
                    rows[-1].append(inline(tokens[j + 1].children or []))
                elif tokens[j].type == "tr_open":
                    rows.append([])
                j += 1
            flow.append(table_flowable(rows, width))
            flow.append(Spacer(1, 8))
            i = j + 1
            continue
        if t.type in ("fence", "code_block"):
            code = t.content.rstrip("\n")
            box = Table([[Preformatted(code, S["code"])]], colWidths=[width], hAlign="LEFT")
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BAND),
                                     ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 7),
                                     ("TOPPADDING", (0, 0), (-1, -1), 6),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            flow += [box, Spacer(1, 8)]
            i += 1
            continue
        if t.type == "hr":
            flow.append(HRFlowable(width="100%", thickness=0.5, color=RULE, spaceBefore=6, spaceAfter=8))
        elif t.type == "blockquote_open":
            end = match_close(tokens, i)
            flow += render_tokens(tokens[i + 1:end], width, quote=True)
            i = end + 1
            continue
        i += 1
    return flow


def render(md_text: str, width: float) -> list:
    return render_tokens(MarkdownIt("commonmark").enable("table").parse(md_text), width)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Kalshi AutoML — Progress Report 1")
    ap.add_argument("--subtitle", default="Damien Nguyen · CSC 4999 · Advisor: Prof. Keith Mills")
    args = ap.parse_args()

    W, H = LETTER
    margin = 0.85 * inch
    width = W - 2 * margin
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.line(margin, 0.62 * inch, W - margin, 0.62 * inch)
        canvas.setFont("Helvetica", 7.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(margin, 0.45 * inch, f"{args.title} · {date.today().isoformat()}")
        canvas.drawRightString(W - margin, 0.45 * inch, f"{canvas.getPageNumber()}")
        canvas.restoreState()

    doc = BaseDocTemplate(out, pagesize=LETTER, leftMargin=margin, rightMargin=margin,
                          topMargin=margin, bottomMargin=0.85 * inch, title=args.title,
                          author="Damien Nguyen")
    doc.addPageTemplates([PageTemplate(id="body", frames=[Frame(margin, 0.85 * inch, width,
                                                                H - margin - 0.85 * inch, id="f")],
                                       onPage=decorate)])

    story = [Spacer(1, 1.6 * inch), Paragraph(args.title, S["title"]), Spacer(1, 6),
             Paragraph(args.subtitle, S["subtitle"]), Spacer(1, 10),
             HRFlowable(width="100%", thickness=1, color=ACCENT), Spacer(1, 14)]
    contents = [("Progress report 1", args.files[0])] + [(None, f) for f in args.files[1:]]
    story.append(Paragraph("Contents", S["h3"]))
    names = []
    for f in args.files:
        first = next((l for l in open(f) if l.startswith("# ")), os.path.basename(f))
        names.append(first.lstrip("# ").strip())
    story.append(ListFlowable([ListItem(Paragraph(n, S["body"]), leftIndent=14) for n in names],
                              bulletType="1", bulletFontSize=7, leftIndent=14))
    story.append(PageBreak())

    for n, f in enumerate(args.files):
        story += render(open(f).read(), width)
        if n < len(args.files) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
