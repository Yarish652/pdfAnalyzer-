import re
import unicodedata

import pymupdf

try:
    from backend import ocr as ocr_engine
except ModuleNotFoundError:
    import ocr as ocr_engine


# A word broken across lines with a hyphen: "suc-\ncessful".
LINE_BREAK_HYPHEN = re.compile(r"(\w+)-\n(\w+)")
WORD = re.compile(r"\w+")
DOT_LEADER = re.compile(r"(?:[ \t]*\.){4,}[ \t]*")


def _open(file):
    """Open a path or a binary file-like object such as an upload stream."""
    if hasattr(file, "read"):
        file.seek(0)
        return pymupdf.open(stream=file.read(), filetype="pdf")
    return pymupdf.open(file)


def _join_hyphenated(text: str, vocabulary: set[str]) -> str:
    """Drop a line-break hyphen only when the joined word occurs elsewhere.

    LaTeX hyphenation ("recogni-\\ntion") should be undone, but genuine
    compounds that happen to wrap ("grapheme-\\nto-phoneme") must keep the
    hyphen. The document's own vocabulary decides between the two.
    """

    def replace(match):
        joined = match.group(1) + match.group(2)
        if joined.lower() in vocabulary:
            return joined
        return f"{match.group(1)}-{match.group(2)}"

    return LINE_BREAK_HYPHEN.sub(replace, text)


def extract_document(file, ocr: bool = True, on_progress=None):
    """Extract text blocks (roughly paragraphs) and their lines per page.

    PyMuPDF keeps word spacing and inline math far better than pypdf on
    LaTeX-generated PDFs, and its text blocks follow the visual layout, so a
    block is usually a paragraph, a heading, a caption, or a list item group.

    Pages with no text layer (scans, image-only slides) are read with OCR when
    `ocr` is true and an engine is installed; such pages are marked
    "ocr": True. Pages with a text layer are never OCRed. Pages that still
    have no text are omitted, so an unreadable PDF yields []. OCR pages also
    carry "ocr_confidence", the mean recognition confidence of their lines.

    on_progress(done, total) is called after each OCRed page.
    """

    with _open(file) as pdf:
        pages = [
            [
                page_number,
                page.rect.width > page.rect.height,
                [
                    unicodedata.normalize("NFKC", block[4])
                    for block in page.get_text("blocks")
                    # block[6] is 0 for text and 1 for images.
                    if block[6] == 0 and block[4].strip()
                ],
                False,
                None,
            ]
            for page_number, page in enumerate(pdf, start=1)
        ]

        needs_ocr = [entry for entry in pages if not entry[2]]
        if ocr and needs_ocr and ocr_engine.is_available():
            for done, entry in enumerate(needs_ocr, start=1):
                blocks, confidence = ocr_engine.ocr_page(pdf[entry[0] - 1])
                entry[2] = [unicodedata.normalize("NFKC", block) for block in blocks]
                entry[3] = True
                entry[4] = confidence
                if on_progress:
                    on_progress(done, len(needs_ocr))

    vocabulary = {
        word.lower()
        for _, _, raw_blocks, _, _ in pages
        for block in raw_blocks
        for word in WORD.findall(block)
    }

    document = []
    for page_number, landscape, raw_blocks, from_ocr, confidence in pages:
        blocks = []
        lines = []
        for raw_block in raw_blocks:
            text = _join_hyphenated(raw_block, vocabulary)
            # Collapse table-of-contents dot leaders (". . . . . 23").
            text = DOT_LEADER.sub(" … ", text)
            block_lines = [line.strip() for line in text.split("\n") if line.strip()]
            if block_lines:
                blocks.append(" ".join(block_lines))
                lines.append(block_lines)

        if blocks:
            page = {
                "page": page_number,
                "landscape": landscape,
                # Paragraph-level text, used by prose and slide chunking.
                "blocks": blocks,
                # The same blocks split into visual lines. Layout-driven
                # documents such as resumes often merge a heading and its
                # body into one block, and only the line breaks reveal it.
                "lines": lines,
            }
            if from_ocr:
                page["ocr"] = True
                page["ocr_confidence"] = confidence
            document.append(page)

    return document
