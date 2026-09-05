from typing import List
import re
import nltk
from nltk.tokenize import sent_tokenize


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

# Approximate maximum number of words in a chunk.
# This is a starting point and can be tuned after testing
# retrieval quality.
MAX_CHUNK_WORDS = 80

# Number of sentences carried from the previous chunk
# into the next chunk when fallback chunking is required.
OVERLAP_SENTENCES = 1


# ---------------------------------------------------------
# Normalized chunk format
# ---------------------------------------------------------
# Every chunk returned by this module has the same format:
#
# {
#     "text": "...",
#     "metadata": {
#         "page": 1,
#         "section": "...",
#         "subsection": "...",
#         "chunk_index": 0
#     }
# }
#
# The rest of the RAG pipeline does not need to know
# whether the document was structured or unstructured.
# ---------------------------------------------------------

def make_chunk(
    text: str,
    page: int,
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
        },
    }

# ---------------------------------------------------------
# Heading detection
# ---------------------------------------------------------

def is_section_heading(block: str) -> bool:
    """
    Detect a likely top-level section.

    Uppercase text is one useful signal for documents such
    as resumes and reports.

    This is only a heuristic. We intentionally keep it
    conservative because PDF text extraction can remove
    formatting information such as font size and boldness.
    """

    block = block.strip()

    if not block:
        return False

    # Bullets are content
    if block.startswith(("•", "-", "*")):
        return False

    # Very long blocks are content
    if len(block) > 100:
        return False

    # Sentences are content
    if block.endswith((".", "!", "?")):
        return False

    # Uppercase text is a strong section signal
    return block.isupper()


def is_subsection_heading(block: str) -> bool:


    """
    Detect a likely subsection heading.

    We deliberately require stronger structural evidence
    than simply checking whether a block is short.

    Numbered headings and explicit structural prefixes are
    strong signals. Otherwise, we treat the block as content.
    """

    block = block.strip()

    if not block:
        return False

    # Bullets are content, not headings
    if block.startswith(("•", "-", "*")):
        return False

    # Long blocks are unlikely to be headings
    if len(block) > 100:
        return False

    # Sentences are generally content
    if block.endswith((".", "!", "?")):
        return False

    # Uppercase blocks are handled as top-level sections
    if block.isupper():
        return False

    # Numbered headings:
    #
    # 1 Introduction
    # 1.1 Background
    # 2.3.1 Architecture
    #
    if re.match(r"^\d+(\.\d+)*[\s:.-]+", block):
        return True

    # Common structural prefixes.
    #
    # These are document-structure signals, not
    # document-specific keywords.
    structural_prefixes = (
        "chapter ",
        "section ",
        "appendix ",
        "part ",
    )

    if block.lower().startswith(structural_prefixes):
        return True

    # Without strong evidence, do not guess.
    return False

def is_content_group_start(blocks: list[str], index: int) -> bool:
    """
    Detect whether the current block is likely the beginning
    of a new semantic content group.

    We use neighboring blocks rather than looking at the
    current block in isolation.

    Example:

        Project A
        Technologies
        • Bullet
        • Bullet
        Project B
        Technologies
        • Bullet

    "Project B" is a likely new content group because it is
    followed by a short metadata-like block and then content.

    This is intentionally heuristic. The goal is to detect
    strong boundaries without making assumptions about a
    particular document type.
    """

    block = blocks[index].strip()

    if not block:
        return False

    # We cannot detect a boundary without a following block.
    if index + 1 >= len(blocks):
        return False

    next_block = blocks[index + 1].strip()

    # Bullets themselves belong to the current group.
    if block.startswith(("•", "-", "*")):
        return False

    # A sentence is normally content.
    if block.endswith((".", "!", "?")):
        return False

    # Long text is normally content.
    if len(block) > 100:
        return False

    # If the next block is also a bullet, the current block
    # is unlikely to be a new semantic group.
    if next_block.startswith(("•", "-", "*")):
        return False

    # Numbered structures are strong evidence.
    if re.match(r"^\d+(\.\d+)*[\s:.-]+", block):
        return True

    # Explicit structural prefixes are strong evidence.
    structural_prefixes = (
        "chapter ",
        "section ",
        "appendix ",
        "part ",
    )

    if block.lower().startswith(structural_prefixes):
        return True

    return False

def looks_like_content_boundary(block: str) -> bool:
    """
    Detect whether a block may represent the beginning of a
    new semantic content group.

    This does NOT identify what the block means.

    It only looks for structural signals that suggest the
    previous group may have ended and a new one is beginning.
    """

    block = block.strip()

    if not block:
        return False

    # Bullets are normally part of the current content group.
    if block.startswith(("•", "-", "*")):
        return False

    # Long blocks are normally body content.
    if len(block) > 100:
        return False

    # Sentences are normally body content.
    if block.endswith((".", "!", "?")):
        return False

    # Numbered items are strong boundary signals.
    if re.match(r"^\d+(\.\d+)*[\s:.-]+", block):
        return True

    # Explicit structural labels.
    structural_prefixes = (
        "chapter ",
        "section ",
        "appendix ",
        "part ",
    )

    if block.lower().startswith(structural_prefixes):
        return True

    return False

def looks_like_short_title(block: str) -> bool:
    """
    Identify a short title-like block.

    This does not use document-specific keywords.
    It only looks at the shape of the text.
    """

    block = block.strip()

    if not block:
        return False

    if block.startswith(("•", "-", "*")):
        return False

    if len(block) > 100:
        return False

    if block.endswith((".", "!", "?")):
        return False

    words = block.split()

    # Very short blocks can be labels/headings.
    if 1 <= len(words) <= 12:
        return True

    return False
# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def word_count(text: str) -> int:
    """Return the approximate number of words in text."""
    return len(text.split())


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    NLTK is used here because sentence boundaries are more
    useful for RAG chunking than blindly cutting text at
    a fixed character position.
    """

    return [
        sentence.strip()
        for sentence in sent_tokenize(text)
        if sentence.strip()
    ]


# ---------------------------------------------------------
# Structured document detection
# ---------------------------------------------------------

def is_structured(document) -> bool:
    """
    Determine whether the document contains enough reliable
    structure to use structure-aware chunking.

    For the current implementation, we look for multiple
    strong section signals rather than assuming that one
    uppercase block means the entire document is structured.

    This can be improved later using PDF layout information.
    """

    section_count = 0

    for page in document:
        for block in page["blocks"]:
            if is_section_heading(block):
                section_count += 1

    # Require at least two sections before calling the
    # document structurally organized.
    return section_count >= 2


# ---------------------------------------------------------
# Structured chunking
# ---------------------------------------------------------

def structured_chunker(document):
    """
    Chunk a structured document while preserving detected
    section and subsection boundaries.

    Within a section, we also try to identify semantic
    content-group boundaries.

    Size-based splitting is only used when a logical group
    becomes too large.
    """

    chunks = []
    chunk_index = 0

    current_section = None
    current_subsection = None

    current_text = []
    current_page = None

    def flush_chunk():
        nonlocal chunk_index
        nonlocal current_text
        nonlocal current_page

        if not current_text:
            return

        text = " ".join(current_text).strip()

        chunks.append(
            make_chunk(
                text=text,
                page=current_page,
                section=current_section,
                subsection=current_subsection,
                chunk_index=chunk_index,
            )
        )

        chunk_index += 1
        current_text = []
        current_page = None

    for page in document:
        page_number = page["page"]
        blocks = page["blocks"]

        for i, raw_block in enumerate(blocks):
            block = raw_block.strip()

            if not block:
                continue

            # ---------------------------------------------
            # Top-level section boundary
            # ---------------------------------------------

            if is_section_heading(block):
                flush_chunk()

                current_section = block
                current_subsection = None

                continue

            # ---------------------------------------------
            # Explicit subsection boundary
            # ---------------------------------------------

            if is_subsection_heading(block):
                flush_chunk()

                current_subsection = block

                continue

            # ---------------------------------------------
            # Content-group boundary
            # ---------------------------------------------
            #
            # This is checked before adding the current
            # block to the existing chunk.
            #
            # We only split when there is evidence that
            # this block begins a new logical group.
            # ---------------------------------------------

            if (
                current_text
                and is_content_group_start(blocks, i)
            ):
                flush_chunk()

            # ---------------------------------------------
            # Normal content
            # ---------------------------------------------

            if current_page is None:
                current_page = page_number

            candidate = " ".join(
                current_text + [block]
            )

            # Size is a safety limit, not the primary
            # semantic boundary.
            if (
                current_text
                and word_count(candidate) > MAX_CHUNK_WORDS
            ):
                flush_chunk()
                current_page = page_number

            current_text.append(block)

    # Flush final chunk
    flush_chunk()

    return chunks
# ---------------------------------------------------------
# Fallback chunking
# ---------------------------------------------------------

def fallback_chunker(document):
    """
    Chunk an unstructured document.

    Fallback strategy:

        blocks
          ↓
        sentences
          ↓
        retrieval-sized chunks
          ↓
        sentence overlap

    No section or subsection metadata is invented.
    """

    chunks = []
    chunk_index = 0

    for page in document:
        page_number = page["page"]

        for block in page["blocks"]:
            block = block.strip()

            if not block:
                continue

            sentences = split_sentences(block)

            if not sentences:
                continue

            current_sentences = []

            for sentence in sentences:

                candidate = " ".join(
                    current_sentences + [sentence]
                )

                # If the next sentence would exceed the
                # target size, save the current chunk.
                if (
                    current_sentences
                    and word_count(candidate) > MAX_CHUNK_WORDS
                ):
                    text = " ".join(current_sentences)

                    chunks.append(
                        make_chunk(
                            text=text,
                            page=page_number,
                            section=None,
                            subsection=None,
                            chunk_index=chunk_index,
                        )
                    )

                    chunk_index += 1

                    # Keep a small amount of context overlap.
                    current_sentences = current_sentences[
                        -OVERLAP_SENTENCES:
                    ]

                current_sentences.append(sentence)

            # Save remaining sentences
            if current_sentences:
                text = " ".join(current_sentences)

                chunks.append(
                    make_chunk(
                        text=text,
                        page=page_number,
                        section=None,
                        subsection=None,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

    return chunks


# ---------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------

def chunk_document(document):
    """
    Main entry point for document chunking.

    Structured documents use their detected hierarchy.

    Unstructured documents use the fallback strategy.

    Both paths return exactly the same normalized chunk
    representation.
    """

    if is_structured(document):
        return structured_chunker(document)

    return fallback_chunker(document)

"""
CHUNKING NOTES
==============

Current architecture:

    PDF
      ↓
    pdf_reader.py
      ↓
    Structured document
      ↓
    chunker.py
      ↓
    Section detection
      ↓
    Semantic / size-controlled chunks
      ↓
    Embeddings
      ↓
    Vector database
      ↓
    Similarity search
      ↓
    Retrieve relevant chunks
      ↓
    Give context to LLM


STRUCTURE-AWARE CHUNKING
========================

The goal is NOT to create a chunker specifically for resumes.

The goal is to preserve the natural semantic structure of
different types of documents.

Examples:

Resume:
    EXPERIENCE
    PROJECTS
    TECHNICAL SKILLS

Research paper:
    INTRODUCTION
    METHODOLOGY
    RESULTS
    CONCLUSION

Documentation:
    INSTALLATION
    CONFIGURATION
    AUTHENTICATION
    API REFERENCE

Book:
    CHAPTER 1
        SECTION 1.1
        SECTION 1.2
    CHAPTER 2

Legal document:
    DEFINITIONS
    OBLIGATIONS
    TERMINATION
    LIABILITY


GENERAL CHUNKING PIPELINE
=========================

    Document
        ↓
    Detect structure
        ↓
    Identify sections
        ↓
    Preserve section boundaries
        ↓
    Split oversized sections
        ↓
    Split using paragraphs / sentences
        ↓
    Add controlled overlap
        ↓
    Create retrieval-sized chunks


IMPORTANT DESIGN PRINCIPLE
==========================

First understand the document.

Then decide where to cut it.

Structure detection and chunk sizing are separate concerns.


SECTION BOUNDARIES
==================

Chunks should NOT normally cross semantic section boundaries.

Good:

    PROJECTS
        ├── Chunk 1
        ├── Chunk 2
        └── Chunk 3

    TECHNICAL SKILLS
        ├── Chunk 4
        └── Chunk 5

Avoid:

    Chunk 1:
        end of PROJECTS
        +
        beginning of TECHNICAL SKILLS


HIERARCHICAL STRUCTURE
======================

A document may contain multiple levels:

    Section
        ↓
    Subsection
        ↓
    Paragraph
        ↓
    Chunk

Example:

    PROJECTS
        ↓
    Real-Time Pronunciation Assessment System
        ↓
    Project description
        ↓
    Retrieval chunk


METADATA
========

Chunks should eventually preserve metadata such as:

    - page number
    - section
    - subsection
    - chunk index

Example:

    {
        "text": "...",
        "page": 1,
        "section": "PROJECTS",
        "subsection": "Real-Time Pronunciation Assessment System",
        "chunk_index": 3
    }

This metadata will later help with:

    - source citations
    - debugging retrieval
    - displaying sources
    - filtering
    - reranking
    - document navigation


FALLBACK STRATEGY
=================

Not every PDF will have obvious sections.

Therefore:

    Structure detected
        ↓
    Section-aware chunking

    Structure unavailable
        ↓
    Paragraph-aware chunking

    Paragraph structure unavailable
        ↓
    Sentence-based chunking


CURRENT LIMITATION
==================

PDF extraction does not always preserve semantic structure perfectly.

For example, extracted text may contain:

    Y ear
    CGP A
    F rameworks & T echnologies
    F eb

These artifacts originate from PDF extraction/layout handling,
not from the chunking algorithm itself.

Therefore the architecture separates:

    pdf_reader.py
        ↓
    Extract and preserve document structure

    chunker.py
        ↓
    Interpret structure and create retrieval chunks


RAG GOAL
========

    Document
        ↓
    Extract structure
        ↓
    Detect semantic boundaries
        ↓
    Create meaningful chunks
        ↓
    Create embeddings
        ↓
    Similarity search
        ↓
    Retrieve relevant chunks
        ↓
    Rerank / filter
        ↓
    Give context to LLM
        ↓
    Generate grounded answer
"""
