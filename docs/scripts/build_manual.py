"""Build the standalone PDF from docs/manual.md and its local screenshots.

Requires reportlab and Pillow. Run from any directory; see docs/development.md.
"""
from pathlib import Path
import argparse
import re
from html import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image, PageBreak, Spacer, KeepTogether, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[2]
BLUE = colors.HexColor('#176583')
INK = colors.HexColor('#182B3A')


def inline(text):
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',
                  lambda m: '<a color="#176583" href="' + (
                      m[2] if m[2].startswith(('#', 'https://', 'http://')) else
                      'https://github.com/ckolivas/PlanetaryTools/blob/master/docs/' + m[2]
                  ) + '">' + m[1] + '</a>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    return text


class Manual(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, 'anchor'):
            self.canv.bookmarkPage(flowable.anchor)
            self.canv.addOutlineEntry(flowable.getPlainText(), flowable.anchor, 0)
            self.notify('TOCEntry', (0, flowable.getPlainText(), self.page, flowable.anchor))


def build(output, fonts):
    for name, suffix in [('PT', ''), ('PT-Bold', '-Bold'), ('PT-Italic', '-Oblique'),
                         ('PT-BoldItalic', '-BoldOblique')]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / ('DejaVuSans' + suffix + '.ttf'))))
    pdfmetrics.registerFontFamily('PT', normal='PT', bold='PT-Bold', italic='PT-Italic',
                                 boldItalic='PT-BoldItalic')
    styles = getSampleStyleSheet()
    for name in ('Normal', 'BodyText', 'Title', 'Heading1', 'Heading2'):
        styles[name].fontName = 'PT'
        styles[name].textColor = INK
    body = ParagraphStyle('ManualBody', parent=styles['BodyText'], fontSize=9.5,
                          leading=13, spaceAfter=7, allowOrphans=0, allowWidows=0)
    title = ParagraphStyle('ManualTitle', parent=styles['Title'], fontName='PT-Bold',
                           fontSize=27, leading=34, spaceAfter=20, alignment=0)
    h1 = ParagraphStyle('Section', parent=styles['Heading1'], fontName='PT-Bold',
                        fontSize=21, leading=27, spaceAfter=15, textColor=BLUE)
    h2 = ParagraphStyle('Subsection', parent=styles['Heading2'], fontName='PT-Bold',
                        fontSize=11, leading=15, spaceBefore=10, spaceAfter=5, textColor=BLUE, keepWithNext=True)
    small = ParagraphStyle('Caption', parent=body, fontSize=8, leading=11,
                           textColor=colors.HexColor('#536773'))
    items = ParagraphStyle('Step', parent=body, leftIndent=17, firstLineIndent=-17)
    md = (ROOT / 'docs/manual.md').read_text()
    revision = next(line for line in md.splitlines() if line.startswith('Revised **'))
    story = [Spacer(1, 35), Paragraph('Planetary Tools', title),
             Paragraph('Illustrated user manual', h1), Paragraph(inline(revision), body)]

    def figure(path, width, maxheight):
        image = Image(str(path))
        scale = min(width / image.imageWidth, maxheight / image.imageHeight, 1)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        image.hAlign = 'CENTER'
        return image

    story += [Spacer(1, 12), figure(ROOT / 'docs/images/workspace_dock.png', 499, 340),
              Spacer(1, 18), Paragraph('Use the linked contents or PDF bookmarks to jump to a tool. '
              'Screenshots and instructions are embedded for offline reading. The GitHub manual '
              'and this PDF use the same source text.', body), PageBreak()]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle('ContentsEntry', fontName='PT', fontSize=10,
                                     leading=14, spaceBefore=0, spaceAfter=0, textColor=INK)]
    # Bookmark without listing the contents page in itself.
    story += [Paragraph('<a name="contents"/>Contents', h1), toc]
    anchor = None
    block = []
    active = False

    def flush():
        if block:
            story.append(Paragraph(inline(' '.join(block)), body))
            block.clear()

    for line in md.splitlines():
        if line.startswith('<a id='):
            flush()
            anchor = re.search(r'id="([^"]+)"', line)[1]
            active = True
        elif not active:
            continue
        elif line.startswith('## '):
            flush()
            story.append(PageBreak())
            heading = Paragraph(inline(line[3:]), h1)
            heading.anchor = anchor
            story.append(heading)
        elif line.startswith('### '):
            flush()
            if line[4:] == 'Undocking, resizing and reattaching tool panels':
                story.append(PageBreak())
            story.append(Paragraph(inline(line[4:]), h2))
        elif line.startswith('<img '):
            flush()
            src = re.search(r'src="([^"]+)"', line)[1]
            alt = re.search(r'alt="([^"]+)"', line)[1]
            width = int(re.search(r'width="(\d+)"', line)[1])
            image = figure(ROOT / 'docs' / src, 499 if width >= 600 else 300, 270 if width >= 600 else 340)
            figure_block = KeepTogether([image, Spacer(1, 6), Paragraph(escape(alt), small)])
            figure_block.wide = width >= 600
            figure_block.source = src
            story.append(figure_block)
        elif line.startswith('[Back to contents]'):
            flush()
        elif re.match(r'^\d+\. ', line):
            flush();story.append(Paragraph(inline(line), items))
        elif line.startswith('- '):
            flush();story.append(Paragraph('• ' + inline(line[2:]), items))
        elif not line.strip():
            flush()
        else:
            block.append(line)
    flush()
    footer = re.search(r'Revised \*\*(.*?)\*\*.*?build \*\*(.*?)\*\*', revision)

    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont('PT-Bold', 8);canvas.setFillColor(BLUE)
        canvas.drawString(48, A4[1]-30, 'PLANETARY TOOLS / USER MANUAL')
        canvas.setFont('PT', 7)
        canvas.drawString(48, 28, f'Build {footer[2]} | Revised {footer[1]}')
        canvas.drawRightString(A4[0]-48, 28, str(doc.page))
        if doc.page > 2:
            canvas.drawRightString(A4[0]-48, A4[1]-30, 'Contents')
            canvas.linkRect('', 'contents', (A4[0]-100, A4[1]-34, A4[0]-48, A4[1]-23), thickness=0)
        canvas.restoreState()

    # Use space alongside narrow tool screenshots instead of leaving a large
    # blank area. Wide workspace/list images retain the full page width.
    laid_out = []
    section = []

    def arrange(parts):
        figures = [i for i, part in enumerate(parts) if hasattr(part, 'wide')]
        section_title = next((part.getPlainText() for part in parts
                              if isinstance(part, Paragraph) and hasattr(part, 'anchor')), '')
        if section_title in ('Opening and saving images', 'Derotate/Align',
                             'Animate and motion interpolation', 'Batch Processing'):
            i = figures[0]
            image = parts[i]._content[0]
            factor = min(499 / image.imageWidth, 470 / image.imageHeight, 1)
            image.drawWidth = image.imageWidth * factor
            image.drawHeight = image.imageHeight * factor
            continuation = Paragraph(inline(section_title) + ' / How to use', h1)
            return parts[:i+1] + [PageBreak(), continuation] + parts[i+1:]
        if section_title == 'A practical processing sequence':
            image = parts[figures[0]]._content[0]
            factor = min(499 / image.imageWidth, 190 / image.imageHeight, 1)
            image.drawWidth = image.imageWidth * factor
            image.drawHeight = image.imageHeight * factor
        if section_title == 'View, help and updates':
            left = []
            for position, i in enumerate(figures):
                image = parts[i]._content[0]
                factor = min(215 / image.imageWidth, 160 / image.imageHeight, 1)
                image.drawWidth = image.imageWidth * factor
                image.drawHeight = image.imageHeight * factor
                left += parts[i]._content + [Spacer(1, 10)]
                if position + 1 < len(figures):
                    left += parts[i+1:figures[position+1]]
            table = Table([[left, parts[figures[-1]+1:]]], colWidths=[232, 267])
            table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
            return parts[:figures[0]] + [table]
        if len(figures) == 1 and not parts[figures[0]].wide:
            i = figures[0]
            image_block = parts[i]
            image = image_block._content[0]
            factor = min(215 / image.imageWidth, 555 / image.imageHeight, 1)
            image.drawWidth = image.imageWidth * factor
            image.drawHeight = image.imageHeight * factor
            headings = [n for n in range(i+1, len(parts))
                        if isinstance(parts[n], Paragraph) and parts[n].style is h2]
            end = headings[1] if len(headings) > 1 else len(parts)
            table = Table([[image_block._content, parts[i+1:end]]],
                          colWidths=[232, 267], splitInRow=1)
            table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                      ('LEFTPADDING', (0, 0), (-1, -1), 0),
                                      ('RIGHTPADDING', (0, 0), (0, 0), 17),
                                      ('RIGHTPADDING', (1, 0), (1, 0), 0)]))
            return parts[:i] + [table] + parts[end:]
        return parts

    for part in story:
        if isinstance(part, PageBreak):
            laid_out.extend(arrange(section))
            laid_out.append(part)
            section = []
        else:
            section.append(part)
    laid_out.extend(arrange(section))
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = Manual(str(output), pagesize=A4, leftMargin=48, rightMargin=48,
                 topMargin=53, bottomMargin=51, title='Planetary Tools - Illustrated User Manual',
                 author='Planetary Tools documentation')
    doc.multiBuild(laid_out, onFirstPage=page, onLaterPages=page)
    print(f'Wrote {output} ({doc.page} pages)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'Planetary-Tools-Manual.pdf')
    parser.add_argument('--font-dir', type=Path, default=Path('/usr/share/fonts/truetype/dejavu'),
                        help='Directory containing the four DejaVu Sans TTF styles')
    args = parser.parse_args()
    build(args.output.resolve(), args.font_dir)
