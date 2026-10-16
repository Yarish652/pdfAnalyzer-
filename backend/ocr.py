"""OCR for PDF pages that have no text layer (scanned documents, image slides).

Only pages without any extractable text are sent here, so text-based PDFs are
never affected. Recognized lines are grouped into blocks in reading order and
returned in the same "lines joined by newlines" form that PyMuPDF text blocks
use, so the rest of the pipeline treats OCR pages like any other page.

The engine is RapidOCR (PaddleOCR PP-OCRv6 models on ONNX Runtime): CPU only,
no system install, about 97% word recall on rendered book pages.
"""

import logging
import re
import threading

import numpy as np
import pymupdf


logger = logging.getLogger(__name__)

# Pages are rendered so their longer side is this many pixels. Recall was flat
# between 1200 and 2000 px; larger renders only cost time.
RENDER_LONG_SIDE_PX = 1600

# Lines recognized with lower confidence are usually noise from figures.
MIN_LINE_SCORE = 0.5

# A line joins the block above it when the vertical gap is at most this many
# line heights and the two overlap horizontally.
MAX_LINE_GAP = 0.9

# Mean line confidence below which recognition is unreliable. Printed scans
# and slides measured 0.94-0.996; handwritten notes 0.80-0.85.
LOW_CONFIDENCE = 0.9

# Minimum body lines in each half before a page is treated as two-column.
MIN_COLUMN_LINES = 4

WORD = re.compile(r"[A-Za-z]{2,}")

_engine = None
_engine_lock = threading.Lock()
_engine_unavailable = False


def _get_engine():
    """Load the OCR engine on first use; text-only workloads never load it."""
    global _engine, _engine_unavailable
    if _engine is None and not _engine_unavailable:
        # Concurrent uploads must not each load their own copy of the models.
        with _engine_lock:
            if _engine is None and not _engine_unavailable:
                try:
                    from rapidocr import RapidOCR

                    _engine = RapidOCR(params={"Global.log_level": "error"})
                except Exception:
                    _engine_unavailable = True
                    logger.warning("OCR engine unavailable; scanned pages will be skipped", exc_info=True)
    return _engine


def is_available() -> bool:
    return _get_engine() is not None


def render_page(page) -> np.ndarray:
    zoom = RENDER_LONG_SIDE_PX / max(page.rect.width, page.rect.height)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)


def ocr_page(page) -> tuple[list[str], float | None]:
    """Return the page's text blocks (lines joined by newlines) and the mean
    recognition confidence of its lines, or None when nothing was found."""
    engine = _get_engine()
    if engine is None:
        return [], None

    image = render_page(page)
    # One recognition at a time: uploads run in worker threads and the engine
    # is not documented as thread safe.
    with _engine_lock:
        output = engine(image)

    if output.boxes is None or not output.txts:
        return [], None

    scores = [float(score) for score in output.scores]
    confidence = sum(scores) / len(scores)

    lines = []
    for box, text, score in zip(output.boxes, output.txts, output.scores):
        text = text.strip()
        if not text or float(score) < MIN_LINE_SCORE:
            continue
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        lines.append({"text": text, "x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys)})

    return group_into_blocks(lines, page_width=image.shape[1]), confidence


def group_into_blocks(lines: list[dict], page_width: float) -> list[str]:
    """Group recognized lines into paragraph-like blocks in reading order.

    Single-column pages: fragments on the same row are joined left to right
    first, so "Project title ... Feb '26 - Jun '26" and table rows such as
    "2027 | B.Tech | Jamia Millia Islamia | 7/10" stay together.
    Two-column pages: each column is read top to bottom, left column first.
    In both cases, rows stacked directly on top of each other form a block.
    """
    if not lines:
        return []

    if is_two_column(lines, page_width):
        middle = page_width / 2
        left = [line for line in lines if (line["x0"] + line["x1"]) / 2 < middle]
        right = [line for line in lines if (line["x0"] + line["x1"]) / 2 >= middle]
        return stack_into_blocks(left) + stack_into_blocks(right)

    return stack_into_blocks(merge_rows(lines))


def is_two_column(lines: list[dict], page_width: float) -> bool:
    """Whether the page is laid out as two text columns.

    Body lines in a two-column layout fill most of their column (30-50% of the
    page width), and there are many in each half. Short fragments such as
    table cells or dates on a resume do not count.
    """
    column_lines = [
        line for line in lines
        if 0.3 * page_width <= line["x1"] - line["x0"] <= 0.5 * page_width
    ]
    left = sum(1 for line in column_lines if line["x1"] <= 0.55 * page_width)
    right = sum(1 for line in column_lines if line["x0"] >= 0.45 * page_width)
    needed = max(MIN_COLUMN_LINES, 0.25 * len(lines))
    return left >= needed and right >= needed


def merge_rows(lines: list[dict]) -> list[dict]:
    """Join fragments whose vertical centers align into one left-to-right line."""
    rows = []
    for line in sorted(lines, key=lambda item: (item["y0"] + item["y1"]) / 2):
        center = (line["y0"] + line["y1"]) / 2
        height = line["y1"] - line["y0"]
        row = rows[-1] if rows else None
        if row and abs(center - row["center"]) <= 0.5 * min(height, row["height"]):
            row["parts"].append(line)
        else:
            rows.append({"center": center, "height": height, "parts": [line]})

    merged = []
    for row in rows:
        parts = sorted(row["parts"], key=lambda item: item["x0"])
        merged.append({
            "text": " ".join(part["text"] for part in parts),
            "x0": min(part["x0"] for part in parts),
            "x1": max(part["x1"] for part in parts),
            "y0": min(part["y0"] for part in parts),
            "y1": max(part["y1"] for part in parts),
        })
    return merged


def is_noisy(text: str) -> bool:
    """Mostly symbols and single letters: diagram labels or garbled math."""
    tokens = text.split()
    if not tokens:
        return True
    return sum(not WORD.search(token) for token in tokens) / len(tokens) > 0.5


def stack_into_blocks(lines: list[dict]) -> list[str]:
    """Group lines into blocks when one sits directly below the other.

    Noisy rows (diagram labels) are never stacked onto clean text or vice
    versa, so the chunker's per-block noise filter can still drop them.
    """
    blocks = []
    for line in sorted(lines, key=lambda item: (item["y0"], item["x0"])):
        height = line["y1"] - line["y0"]
        noisy = is_noisy(line["text"])
        target = None
        for block in reversed(blocks):
            gap = line["y0"] - block["y1"]
            overlaps = line["x0"] < block["x1"] and line["x1"] > block["x0"]
            if overlaps and -0.5 * height <= gap <= MAX_LINE_GAP * height:
                if block["noisy"] == noisy:
                    target = block
                break
        if target is None:
            blocks.append({**line, "noisy": noisy, "lines": [line["text"]]})
        else:
            target["lines"].append(line["text"])
            target["x0"] = min(target["x0"], line["x0"])
            target["x1"] = max(target["x1"], line["x1"])
            target["y1"] = max(target["y1"], line["y1"])

    return ["\n".join(block["lines"]) for block in sorted(blocks, key=lambda block: (block["y0"], block["x0"]))]
