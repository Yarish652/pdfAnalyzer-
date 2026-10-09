"""Type-aware document chunking.

    PDF -> pdf_reader (pages of text blocks)
        -> document_classifier (resume / slides / prose)
        -> type-specific chunker
        -> normalized chunks -> embeddings -> vector store

Every chunker returns the same normalized chunk format, so the rest of the
pipeline does not need to know which one ran:

    {
        "text": "...",
        "metadata": {
            "page": 1,
            "section": "...",
            "subsection": "...",
            "chunk_index": 0,
            "document_type": "prose",
        }
    }

Design rules shared by all chunkers:

- Chunks never cross a detected section boundary.
- Size is a safety limit, not the primary boundary. Oversized units are split
  at sentence (or bullet) boundaries, never mid-sentence.
- Chunks carry their heading in the text, so a chunk that continues a section
  or a resume entry still says what it is about.
"""

import re
from collections import Counter
from typing import List

import nltk
from nltk.tokenize import sent_tokenize

try:
    from backend.document_classifier import PROSE, RESUME, SLIDES, block_lines, classify_document, is_resume_section
except ModuleNotFoundError:
    from document_classifier import PROSE, RESUME, SLIDES, block_lines, classify_document, is_resume_section


def ensure_nltk_resource(resource_path: str, package: str):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package, quiet=True)


ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")


# ---------------------------------------------------------
# Chunk configuration
# ---------------------------------------------------------

# Approximate maximum words per chunk for each document type. MPNet reads at
# most 384 tokens, so chunks stay comfortably below that.
MAX_CHUNK_WORDS = {
    RESUME: 120,
    SLIDES: 150,
    PROSE: 150,
}

# Sentences carried from one prose chunk into the next within a section.
OVERLAP_SENTENCES = 1

# A sentence longer than this is split by words so no chunk becomes huge.
MAX_SENTENCE_WORDS = 200

BULLET_PREFIXES = ("•", "●", "▪", "◦", "-", "–", "*", "")
BULLET_SPLIT = re.compile("\\s*(?=[\u2022\u25cf\u25aa\u25e6\uf0b7])")
WORD = re.compile(r"[A-Za-z]{2,}")


# ---------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------

def make_chunk(
    text: str,
    page: int,
    document_type: str,
    section: str | None = None,
    subsection: str | None = None,
    chunk_index: int = 0,
):
    return {
        "text": text.strip(),
        "metadata": {
            "page": page,
            "section": section or "",
            "subsection": subsection or "",
            "chunk_index": chunk_index,
            "document_type": document_type,
        },
    }


def word_count(text: str) -> int:
    """Return the approximate number of words in text."""
    return len(text.split())


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, breaking up any runaway sentence by words."""
    sentences = []
    for sentence in sent_tokenize(text):
        words = sentence.split()
        for start in range(0, len(words), MAX_SENTENCE_WORDS):
            piece = " ".join(words[start:start + MAX_SENTENCE_WORDS])
            if piece:
                sentences.append(piece)
    return sentences


def is_bullet(block: str) -> bool:
    return block.lstrip().startswith(BULLET_PREFIXES)


def noise_ratio(text: str) -> float:
    """Share of tokens that are not words: figure labels, math symbols."""
    tokens = text.split()
    if not tokens:
        return 1.0
    return sum(not WORD.search(token) for token in tokens) / len(tokens)


def strip_repeated_blocks(document):
    """Remove running headers, footers, and bare page numbers.

    A block that appears (ignoring digits) on at least a quarter of the pages
    of a multi-page document is page furniture, not content.
    """

    pages = len(document)
    if pages < 3:
        return document

    def signature(block: str) -> str:
        return re.sub(r"\d+", "#", block.strip().lower())

    def edge_indexes(blocks):
        # Headers and footers sit at the top or bottom of a page.
        return set(range(min(2, len(blocks)))) | set(range(max(0, len(blocks) - 2), len(blocks)))

    counts = Counter(
        page_signature
        for page in document
        for page_signature in {
            signature(page["blocks"][index]) for index in edge_indexes(page["blocks"])
        }
    )
    threshold = max(3, pages // 4)

    def is_furniture(blocks, index):
        block = blocks[index].strip()
        return index in edge_indexes(blocks) and (
            counts[signature(block)] >= threshold
            or re.fullmatch(r"\d{1,4}", block) is not None
        )

    return [
        {
            **page,
            "blocks": [
                block
                for index, block in enumerate(page["blocks"])
                if not is_furniture(page["blocks"], index)
            ],
        }
        for page in document
    ]


def pack_units(units: list[str], max_words: int, overlap: int = 0) -> list[list[str]]:
    """Greedily group text units (sentences or bullets) into chunk-sized lists."""
    groups = []
    current = []

    for unit in units:
        if current and word_count(" ".join(current + [unit])) > max_words:
            groups.append(current)
            current = current[-overlap:] if overlap else []
        current.append(unit)

    if current:
        groups.append(current)

    return groups


def with_heading(heading: str, body: str) -> str:
    return f"{heading}\n{body}" if heading else body


# ---------------------------------------------------------
# Resume chunking
# ---------------------------------------------------------

def is_resume_heading(block: str) -> bool:
    block = block.strip()
    if is_resume_section(block):
        return True
    letters = re.sub(r"[^A-Za-z]", "", block)
    return (
        len(block) <= 40
        and len(letters) >= 4
        and block.isupper()
        and not is_bullet(block)
    )


def resume_units(document):
    """Turn blocks into (page, kind, text) units using their visual lines.

    kind is "heading", "bullet", or "text". Inside a block, lines that wrap a
    bullet stay with that bullet and consecutive plain lines are merged, so a
    block reading "EXPERIENCE / Intern - Honda / June '26 / - Built ..." yields
    a heading, one text unit, and one bullet unit.
    """

    units = []
    for page in document:
        for lines in block_lines(page):
            current = None
            previous_line = ""
            for line in lines:
                if is_resume_heading(line):
                    units.append((page["page"], "heading", line))
                    current = None
                elif is_bullet(line):
                    current = [page["page"], "bullet", line]
                    units.append(current)
                elif current is not None and (
                    current[1] == "text" or wraps_previous_line(previous_line, line)
                ):
                    current[2] = f"{current[2]} {line}"
                else:
                    current = [page["page"], "text", line]
                    units.append(current)
                previous_line = line
    return [tuple(unit) for unit in units]


def wraps_previous_line(previous_line: str, line: str) -> bool:
    """Whether a plain line after a bullet continues it rather than starting
    the next entry: "and REST APIs." continues, while "BidBazaar - Auction
    Platform" after a bullet ending in a period is a new entry title."""
    return line[:1].islower() or not previous_line.rstrip().endswith((".", "!", "?"))


def resume_chunker(document):
    """Chunk a resume into one chunk per entry (job, project, degree, skills).

    Within a section, text that follows bullets starts a new entry, so "title,
    dates, bullets, title, dates, bullets" yields two entries while a list of
    skill lines stays together. Every chunk is prefixed with
    "SECTION > entry title" so a split entry keeps its name.
    """

    max_words = MAX_CHUNK_WORDS[RESUME]
    entries = []
    section = ""
    entry = None

    def start_entry(page, title):
        nonlocal entry
        entry = {"page": page, "section": section, "title": title, "units": [], "has_bullets": False}
        entries.append(entry)

    for page, kind, text in resume_units(document):
        if kind == "heading":
            section = text
            entry = None
        elif kind == "bullet":
            if entry is None:
                start_entry(page, "")
            entry["units"].extend(b for b in BULLET_SPLIT.split(text) if b.strip())
            entry["has_bullets"] = True
        elif entry is None or entry["has_bullets"]:
            start_entry(page, text)
        else:
            entry["units"].append(text)

    chunks = []
    for item in entries:
        title_line = " > ".join(part for part in (item["section"], item["title"]) if part)
        units = item["units"] or [""]

        for group in pack_units(units, max_words):
            body = " ".join(group).strip()
            text = with_heading(title_line, body)
            if not text.strip():
                continue
            chunks.append(make_chunk(
                text=text,
                page=item["page"],
                document_type=RESUME,
                section=item["section"],
                subsection=item["title"][:120],
                chunk_index=len(chunks),
            ))

    return chunks


# ---------------------------------------------------------
# Slide chunking
# ---------------------------------------------------------

def slides_chunker(document):
    """One chunk per slide, titled by the slide's first short block.

    Near-empty slides (section dividers, "Questions?") are carried forward as
    a prefix of the next slide instead of becoming fragment chunks.
    """

    max_words = MAX_CHUNK_WORDS[SLIDES]
    chunks = []
    carried = ""

    for page in strip_repeated_blocks(document):
        blocks = [block.strip() for block in page["blocks"] if block.strip()]
        if not blocks:
            continue

        title = blocks[0] if word_count(blocks[0]) <= 12 else ""
        text = " ".join(blocks)

        if word_count(text) < 8:
            carried = f"{carried} {text}".strip()
            continue

        heading = " > ".join(part for part in (carried, title) if part)
        body_blocks = blocks[1:] if title else blocks
        sentences = split_sentences(" ".join(body_blocks)) or [title]
        carried = ""

        for group in pack_units(sentences, max_words):
            chunks.append(make_chunk(
                text=with_heading(heading, " ".join(group)),
                page=page["page"],
                document_type=SLIDES,
                subsection=title,
                chunk_index=len(chunks),
            ))

    return chunks


# ---------------------------------------------------------
# Prose chunking (books, papers, reports, notes)
# ---------------------------------------------------------

NUMBERED_HEADING = re.compile(r"^(\d{1,3}(\.\d{1,3})*)\.?\s+[A-Z]")
MULTI_LEVEL_NUMBER = re.compile(r"^\d{1,3}(\.\d{1,3})+\s")
STRUCTURAL_HEADING = re.compile(r"^(chapter|appendix|part)\s+[\w.]+", re.IGNORECASE)

def is_prose_heading(block: str) -> bool:
    block = block.strip()
    words = word_count(block)

    if not block or words > 15 or block.endswith((".", ",", ";", ":")):
        return False

    if NUMBERED_HEADING.match(block):
        # Multi-level numbers like "10.2.1" are strong evidence. A single
        # number ("1 Introduction", "1. Regardless of ...") may be a list
        # item, so it must also be short.
        return bool(MULTI_LEVEL_NUMBER.match(block)) or words <= 8

    if STRUCTURAL_HEADING.match(block) and words <= 10:
        return True

    # Uppercase headings need a real word, so figure labels like "W W W" fail.
    has_real_word = any(len(word) >= 3 for word in WORD.findall(block))
    return block.isupper() and has_real_word and words <= 8


def prose_chunker(document):
    """Sentence-packed chunks that respect section headings.

    Math-only fragments and figure label clouds are dropped, except captions
    and equations, which still carry meaning.
    """

    max_words = MAX_CHUNK_WORDS[PROSE]
    chunks = []
    section = ""
    sentences = []
    start_page = None

    def emit(groups):
        for group in groups:
            chunks.append(make_chunk(
                text=" ".join(group),
                page=start_page,
                document_type=PROSE,
                section=section,
                chunk_index=len(chunks),
            ))

    def flush():
        nonlocal sentences, start_page
        emit(pack_units(sentences, max_words, OVERLAP_SENTENCES))
        sentences = []
        start_page = None

    for page in strip_repeated_blocks(document):
        for block in page["blocks"]:
            block = block.strip()
            if not block:
                continue

            if is_prose_heading(block):
                flush()
                section = block
                continue

            keep = (
                block.startswith(("Figure", "Table"))
                or "=" in block
                or noise_ratio(block) <= 0.5
            )
            if not keep:
                continue

            if start_page is None:
                start_page = page["page"]

            new_sentences = split_sentences(block)
            # A sentence broken by a page or column break continues in a block
            # that starts lowercase; rejoin it instead of starting a fragment.
            if (
                sentences
                and new_sentences
                and new_sentences[0][:1].islower()
                and not sentences[-1].rstrip().endswith((".", "!", "?"))
            ):
                sentences[-1] = f"{sentences[-1]} {new_sentences.pop(0)}"
            sentences.extend(new_sentences)

            # Emit full chunks as we go so each chunk's page stays accurate.
            if word_count(" ".join(sentences)) > max_words:
                *full, sentences = pack_units(sentences, max_words, OVERLAP_SENTENCES)
                emit(full)
                start_page = page["page"]

    flush()
    return chunks


# ---------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------

CHUNKERS = {
    RESUME: resume_chunker,
    SLIDES: slides_chunker,
    PROSE: prose_chunker,
}


def chunk_document(document, document_type: str | None = None):
    """Classify the document (unless a type is given) and chunk it."""
    if document_type is None:
        document_type = classify_document(document).document_type

    return CHUNKERS[document_type](document)
