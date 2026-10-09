"""Classify an extracted document so it can be chunked appropriately.

The classifier uses cheap layout and vocabulary signals rather than an LLM so
it is fast, deterministic, and explainable. Every decision comes with the
reasons that produced it, which are logged and stored with the chunks.

Document types:

    resume  short document with contact details and resume section headings
    slides  landscape pages with little text per page (lecture decks)
    prose   everything else: books, chapters, papers, reports, notes
"""

import re
from dataclasses import dataclass, field


RESUME = "resume"
SLIDES = "slides"
PROSE = "prose"

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"\+?\d[\d\s()-]{8,}\d")

RESUME_SECTIONS = {
    "summary", "professional summary", "profile", "objective", "about me",
    "experience", "work experience", "professional experience", "employment",
    "internships", "education", "projects", "personal projects",
    "skills", "technical skills", "certifications", "achievements",
    "awards", "coursework", "relevant coursework", "publications",
    "leadership", "extracurricular activities", "languages", "interests",
    "positions of responsibility", "volunteering",
}

# A resume is short; anything longer is treated as prose even if it contains
# an email address and an "Education" heading (for example a report).
MAX_RESUME_PAGES = 3
MIN_RESUME_SECTIONS = 2

# Slides: mostly landscape pages that carry little text each.
MIN_SLIDE_LANDSCAPE_SHARE = 0.8
MAX_SLIDE_WORDS_PER_PAGE = 150


@dataclass
class Classification:
    document_type: str
    reasons: list[str] = field(default_factory=list)


def normalize_heading(block: str) -> str:
    return " ".join(re.sub(r"[^a-z ]", " ", block.lower()).split())


def is_resume_section(block: str) -> bool:
    """Known resume heading, including compounds like "Achievements & Awards"."""
    if len(block) > 40:
        return False
    parts = [
        normalize_heading(part)
        for part in re.split(r"\s*(?:&|/|,|\band\b)\s*", block.lower())
        if normalize_heading(part)
    ]
    return bool(parts) and all(part in RESUME_SECTIONS for part in parts)


def block_lines(page) -> list[list[str]]:
    """Lines of each block, falling back to whole blocks when lines are absent."""
    return page.get("lines") or [[block] for block in page["blocks"]]


def classify_document(document) -> Classification:
    pages = len(document)
    blocks = [block for page in document for block in page["blocks"]]
    words = sum(len(block.split()) for block in blocks)

    top_text = " ".join(blocks[:15])
    has_contact = bool(EMAIL.search(top_text) or PHONE.search(top_text))
    resume_sections = {
        normalize_heading(line)
        for page in document
        for lines in block_lines(page)
        for line in lines
        if is_resume_section(line)
    }

    if (
        pages <= MAX_RESUME_PAGES
        and has_contact
        and len(resume_sections) >= MIN_RESUME_SECTIONS
    ):
        return Classification(RESUME, [
            f"{pages} page(s)",
            "contact details near the top",
            f"resume sections: {', '.join(sorted(resume_sections))}",
        ])

    landscape_share = (
        sum(page.get("landscape", False) for page in document) / pages
        if pages
        else 0
    )
    words_per_page = words / pages if pages else 0

    if (
        landscape_share >= MIN_SLIDE_LANDSCAPE_SHARE
        and words_per_page <= MAX_SLIDE_WORDS_PER_PAGE
    ):
        return Classification(SLIDES, [
            f"{landscape_share:.0%} landscape pages",
            f"{words_per_page:.0f} words per page",
        ])

    return Classification(PROSE, [
        f"{pages} page(s)",
        f"{words_per_page:.0f} words per page",
    ])
