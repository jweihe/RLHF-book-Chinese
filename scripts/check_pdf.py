"""Structural PDF smoke check. Requires pymupdf; visual inspection is still required."""
import argparse
import re
import fitz

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("pdf")
args = parser.parse_args()
doc = fitz.open(args.pdf)
errors = []
chapters = [entry for entry in doc.get_toc() if entry[0] == 1 and re.match(r"第.+章", entry[1])]
if len(chapters) != 19:
    errors.append(f"Expected 19 chapter bookmarks, found {len(chapters)}")
images = 0
for index, page in enumerate(doc):
    text = page.get_text()
    if any(token in text for token in ("{#eq:", "{#fig:", "{#tbl:", "�", "□", "??")):
        errors.append(f"Page {index+1}: unresolved markup or replacement glyph")
    for x0,y0,x1,y1,content,*_ in page.get_text("blocks"):
        if x0 < 24 or x1 > page.rect.width-24 or y0 < 20 or y1 > page.rect.height-20:
            errors.append(f"Page {index+1}: text outside safe page bounds")
    for img in page.get_image_info():
        images += 1
        box = fitz.Rect(img["bbox"])
        if box.width < 40 or box.height < 15:
            errors.append(f"Page {index+1}: suspiciously small illustration")
if images < 15:
    errors.append(f"Too few rendered illustrations: {images}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Validated PDF: {len(doc)} pages, 19 chapters, {images} illustrations; no detected clipping or raw labels.")
