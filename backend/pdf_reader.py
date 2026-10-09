import re
import unicodedata

import pymupdf


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


def extract_document(file):
    """Extract text blocks (roughly paragraphs) and their lines per page.

    PyMuPDF keeps word spacing and inline math far better than pypdf on
    LaTeX-generated PDFs, and its text blocks follow the visual layout, so a
    block is usually a paragraph, a heading, a caption, or a list item group.
    Pages without a text layer are omitted, so an image-only PDF yields [].
    """

    with _open(file) as pdf:
        pages = [
            (
                page_number,
                page.rect.width > page.rect.height,
                [
                    unicodedata.normalize("NFKC", block[4])
                    for block in page.get_text("blocks")
                    # block[6] is 0 for text and 1 for images.
                    if block[6] == 0 and block[4].strip()
                ],
            )
            for page_number, page in enumerate(pdf, start=1)
        ]

    vocabulary = {
        word.lower()
        for _, _, raw_blocks in pages
        for block in raw_blocks
        for word in WORD.findall(block)
    }

    document = []
    for page_number, landscape, raw_blocks in pages:
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
            document.append({
                "page": page_number,
                "landscape": landscape,
                # Paragraph-level text, used by prose and slide chunking.
                "blocks": blocks,
                # The same blocks split into visual lines. Layout-driven
                # documents such as resumes often merge a heading and its
                # body into one block, and only the line breaks reveal it.
                "lines": lines,
            })

    return document
