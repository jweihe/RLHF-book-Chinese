"""Render real PDF pages for README. Requires pymupdf and Pillow.

Example: python scripts/make_pdf_previews.py build/pdf/book.pdf --font /path/to/CJK.otf
"""
import argparse
import hashlib
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def render(page, width):
    pix = page.get_pixmap(matrix=fitz.Matrix(width / page.rect.width, width / page.rect.width), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--font", type=Path, required=True, help="CJK font used only for preview labels")
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(args.pdf)
    formula = next(page - 1 for level, title, page in doc.get_toc()
                   if level == 1 and "直接对齐算法" in title)
    code = next(i for i, page in enumerate(doc) if "log_target=True" in page.get_text())
    flow = next(page - 1 for level, title, page in doc.get_toc()
                if level == 1 and "训练概览" in title) + 1
    pages = {"cover": 0, "formulas": formula, "code": code, "workflow": flow}
    for name, index in pages.items():
        render(doc[index], 1200).save(args.output / f"pdf-{name}.png", optimize=True)

    canvas = Image.new("RGB", (1800, 1040), "#EDF2F7")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(str(args.font), 43)
    label = ImageFont.truetype(str(args.font), 28)
    small = ImageFont.truetype(str(args.font), 24)
    draw.text((68, 36), "RLHF 中文手册  /  从概念到推导，再到实现", font=title, fill="#16324F")
    draw.text((70, 99), "19 章系统阅读 · 中文排版 · 公式与代码 · 持续勘误", font=small, fill="#52667C")
    for x, name, caption in [(70, "cover", "01  中文手册"),
                              (650, "formulas", "02  数学推导"),
                              (1230, "code", "03  代码示例")]:
        page_image = render(doc[pages[name]], 500)
        y = 174
        shadow = Image.new("RGBA", canvas.size)
        ImageDraw.Draw(shadow).rectangle((x+4, y+8, x+504, y+page_image.height+8), fill=(28, 52, 74, 50))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow.filter(ImageFilter.GaussianBlur(12)))
        canvas.paste(page_image, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((x, 914), caption, font=label, fill="#16324F")
    draw.text((70, 983), "由仓库新版 PDF 实际页面渲染；点击 README 中的原图可放大查看。", font=small, fill="#52667C")
    canvas.convert("RGB").save(args.output / "reading-preview.png", optimize=True)
    manifest = {"pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
                "page_count": len(doc), "pages_one_based": {k: v+1 for k,v in pages.items()}}
    (args.output / "previews.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
