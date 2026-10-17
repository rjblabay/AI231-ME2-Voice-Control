"""Fill only verified fields in the editable ME2 deck without changing layout."""
from pathlib import Path
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "MEX2_Slides" / "me2-deck-editable.pptx"
OUTPUT = ROOT / "MEX2_Slides" / "me2-deck-filled-known.pptx"


def set_existing_text(shape, value: str) -> None:
    """Change the first existing run so its original font/format is retained."""
    paragraph = shape.text_frame.paragraphs[0]
    if paragraph.runs:
        paragraph.runs[0].text = value
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        shape.text_frame.text = value


def apply(slide, values: dict[int, str]) -> None:
    for index, value in values.items():
        set_existing_text(slide.shapes[index], value)


def main() -> None:
    deck = Presentation(SOURCE)
    page_one, page_two = deck.slides

    # Slide 1: exact placeholder objects only. Empty strings remove their
    # matching angle-bracket glyphs while preserving the template spacing.
    apply(page_one, {
        # Model diagram
        106: "40", 107: "", 110: "300", 111: "",
        112: "CRNN encoder ·", 113: "", 114: "3 CNN blocks", 115: "",
        116: "·", 117: "", 118: "2-layer BiGRU", 119: "", 120: "",
        121: "Streaming inference", 122: "Intent classifier",
        123: "Phrase / slot head", 124: "Action handler",
        # Model footprint. Both deployed ONNX models are shown in the one row.
        12: "", 13: "1.35 / 0.65", 14: "", 16: "", 17: "5.2 / 2.5", 18: "",
        # Held-out accuracy: wake-word / command model.
        24: "", 25: "98.44", 26: "", 28: "", 29: "90.16", 30: "",
        # Dataset source and total clips. The speaker count remains unchanged
        # because it is not recorded in the saved artifacts.
        55: "", 56: "AI231 HF + Astra", 57: "",
        59: "", 60: "—", 61: "", 63: "", 64: "16,868", 65: "",
        # GPU training configuration.
        83: "", 84: "UP HPC GPU / CUDA", 85: "",
        87: "", 88: "weighted CE / focal loss", 89: "",
        91: "", 92: "Adam + cosine LR", 93: "",
        95: "", 96: "66 cmd", 97: "", 99: "", 100: "80 wake", 101: "",
    })

    # Slide 2: submission fields/statuses that are actually known today.
    apply(page_two, {
        37: "Pending publication",
        39: "AI231 HF dataset + local Astra data",
        41: "UP HPC GPU / CUDA · 66 command / 80 wake epochs",
        43: "Saved locally: models/macbook/*.onnx",
        8: "", 9: "pending", 10: "",
        13: "", 14: "pending", 15: "",
        18: "", 19: "local only", 20: "",
        23: "", 24: "pending", 25: "",
        28: "", 29: "partial", 30: "",
        33: "", 34: "not included", 35: "",
    })
    deck.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
