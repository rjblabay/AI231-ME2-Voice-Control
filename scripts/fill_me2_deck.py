"""Fill only the existing fields in the supplied ME2 PDF template."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "MEX2_Slides" / "me2-deck.pdf"
OUTPUT = ROOT / "MEX2_Slides" / "me2-deck-filled.pdf"
BLACK, TEXT, WHITE = HexColor("#000000"), HexColor("#15243A"), HexColor("#F5F8FF")


def mask(c, x, y, width, height, color=BLACK):
    c.setFillColor(color); c.setStrokeColor(color); c.rect(x, y, width, height, fill=1, stroke=0)


def label(c, x, y, value, max_width, size=9.5, color=WHITE, bold=False):
    font = "Helvetica-Bold" if bold else "Helvetica"
    while size > 5.5 and stringWidth(value, font, size) > max_width: size -= .2
    c.setFont(font, size); c.setFillColor(color); c.drawString(x, y, value)


def fill_page_one(c):
    # Keep the original diagram and cards; replace text inside only.
    mask(c, 35, 418, 372, 22, HexColor("#F2F5FB")); label(c, 43, 425, "Mic · 16 kHz → 40-bin log-Mel × 300 frames", 356, color=TEXT, bold=True)
    mask(c, 35, 387, 372, 21, HexColor("#E5F0E6")); label(c, 47, 394, "CRNN encoder · 3 CNN blocks + 2-layer BiGRU", 348, color=HexColor("#244827"), bold=True)
    mask(c, 74, 377, 265, 10); label(c, 118, 378, "streaming inference", 150, size=7.8, color=HexColor("#397524"))
    for x, width, value in ((36,176,"Intent classifier"),(227,177,"Phrase / slot classifier")):
        mask(c, x, 354, width, 20, HexColor("#FFF2D8")); label(c, x + 25, 361, value, width - 35, color=HexColor("#674010"), bold=True)
    mask(c, 36, 323, 176, 20, HexColor("#EEE9FC")); label(c, 66, 330, "Local action handler", 137, color=HexColor("#3A2B78"), bold=True)

    # Dataset and training values (existing table value column only).
    dataset = ["AI231 Hugging Face + custom Astra", "— / 16,868 command + wake clips", "speaker metadata not recorded", "22 canonical intents · 99 output classes"]
    training = ["UP HPC GPU / CUDA", "weighted CE (command) + focal loss (wake)", "Adam + cosine learning-rate schedule", "66 command / 80 wake epochs"]
    for y, value in zip((402,374,347,320), dataset):
        mask(c, 555, y, 250, 18, HexColor("#F3F5F9")); label(c, 560, y + 5, value, 239, color=TEXT)
    for y, value in zip((231,201,173,144), training):
        mask(c, 554, y, 250, 18, HexColor("#F3F5F9")); label(c, 559, y + 5, value, 239, color=TEXT)

    mask(c, 226, 270, 182, 18, HexColor("#F3F5F9")); label(c, 231, 275, "1.35 M + 0.65 M / 5.2 + 2.5 MB", 170, color=TEXT, size=8.4)
    for y, value, size in ((174,"98.44% / 90.16%",9.5),(145,"not measured",9.5),(117,"not measured",9.5),(90,"ONNX Runtime · Raspberry Pi 4",8.5)):
        mask(c, 227, y, 181, 18, HexColor("#F3F5F9")); label(c, 231, y + 5, value, 170, color=TEXT, size=size)


def fill_page_two(c):
    values = ["Pending: publish project repository", "AI231 Hugging Face dataset + local Astra data", "UP HPC GPU / CUDA · 66 command, 80 wake epochs", "Saved locally: models/macbook/*.onnx"]
    for y, value in zip((495,449,403,357), values):
        mask(c, 224, y, 566, 18); label(c, 232, y + 5, value, 550, color=WHITE, size=9.5)
    for y, status in zip((273,242,211,180,149,118), ("PENDING","PENDING","LOCAL ONLY","PENDING","COMPLETE","NOT INCLUDED")):
        mask(c, 695, y, 82, 18, HexColor("#F3F5F9")); label(c, 703, y + 5, status, 66, color=TEXT, bold=True, size=8.4)


def main():
    reader = PdfReader(str(SOURCE)); writer = PdfWriter()
    for number, page in enumerate(reader.pages):
        stream = BytesIO(); canvas = Canvas(stream, pagesize=(float(page.mediabox.width), float(page.mediabox.height)))
        (fill_page_one if number == 0 else fill_page_two)(canvas); canvas.save(); stream.seek(0)
        page.merge_page(PdfReader(stream).pages[0]); writer.add_page(page)
    with OUTPUT.open("wb") as result: writer.write(result)
    print(OUTPUT)


if __name__ == "__main__": main()
