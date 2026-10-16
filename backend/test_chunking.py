from io import BytesIO
import unittest

import pymupdf

from chunker import (
    chunk_document,
    is_prose_heading,
    MAX_CHUNK_WORDS,
    prose_chunker,
    resume_chunker,
    slides_chunker,
    strip_repeated_blocks,
)
from context_builder import expand_with_neighbors, format_context
from document_classifier import PROSE, RESUME, SLIDES, classify_document, is_resume_section
from pdf_reader import _join_hyphenated, extract_document
import ocr


def page(number, blocks, lines=None, landscape=False):
    result = {"page": number, "landscape": landscape, "blocks": blocks}
    if lines is not None:
        result["lines"] = lines
    return result


RESUME_LINES = [
    ["Jane Doe", "jane@example.com | +91-9999999999"],
    ["EDUCATION", "2027 B.Tech, Electronics Jamia Millia Islamia 7/10"],
    [
        "PROJECTS",
        "Speakzy - Language Platform",
        "React, Node.js, MongoDB",
        "• Built a full-stack platform with secure",
        "authentication and REST APIs.",
        "• Added semantic vocabulary search.",
        "BidBazaar - Auction Platform",
        "• Built real-time bidding.",
    ],
    ["Achievements & Certifications", "Amazon ML Challenge 2024", "• Ranked top 100."],
]


def resume_document():
    return [page(1, [" ".join(lines) for lines in RESUME_LINES], RESUME_LINES)]


class ClassifierTests(unittest.TestCase):
    def test_resume_detected_from_headings_merged_into_blocks(self):
        classification = classify_document(resume_document())

        self.assertEqual(classification.document_type, RESUME)

    def test_long_document_with_contact_details_is_prose(self):
        document = [
            page(number, ["EDUCATION", "contact me at someone@example.com", "PROJECTS"])
            for number in range(1, 6)
        ]

        self.assertEqual(classify_document(document).document_type, PROSE)

    def test_landscape_sparse_pages_are_slides(self):
        document = [
            page(number, [f"Slide {number}", "A few words of content."], landscape=True)
            for number in range(1, 11)
        ]

        self.assertEqual(classify_document(document).document_type, SLIDES)

    def test_compound_resume_headings(self):
        self.assertTrue(is_resume_section("Achievements & Certifications"))
        self.assertTrue(is_resume_section("Projects and Publications"))
        self.assertFalse(is_resume_section("Education and the Future of Work"))
        self.assertFalse(is_resume_section("&"))
        self.assertFalse(is_resume_section(""))


class ResumeChunkerTests(unittest.TestCase):
    def setUp(self):
        self.chunks = resume_chunker(resume_document())
        self.texts = [chunk["text"] for chunk in self.chunks]

    def test_year_prefixed_rows_are_kept(self):
        # "2027 B.Tech ..." used to be mistaken for a numbered heading and dropped.
        self.assertTrue(any("7/10" in text for text in self.texts))

    def test_one_chunk_per_entry_with_section_prefix(self):
        speakzy = [text for text in self.texts if "Speakzy" in text]
        bidbazaar = [text for text in self.texts if "BidBazaar" in text]

        self.assertEqual(len(speakzy), 1)
        self.assertEqual(len(bidbazaar), 1)
        self.assertTrue(speakzy[0].startswith("PROJECTS > Speakzy"))
        self.assertNotIn("bidding", speakzy[0])

    def test_wrapped_bullet_lines_stay_with_their_bullet(self):
        speakzy = next(text for text in self.texts if "Speakzy" in text)

        self.assertIn("secure authentication and REST APIs.", speakzy)

    def test_compound_heading_starts_a_section(self):
        amazon = next(chunk for chunk in self.chunks if "Amazon" in chunk["text"])

        self.assertEqual(amazon["metadata"]["section"], "Achievements & Certifications")

    def test_oversized_entry_is_split_but_keeps_its_title(self):
        bullets = [f"• Bullet number {index} with several extra words." for index in range(40)]
        document = [page(1, ["PROJECTS", "Big Project", *bullets])]

        chunks = resume_chunker(document)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk["text"].startswith("PROJECTS > Big Project"))
            self.assertLessEqual(len(chunk["text"].split()), MAX_CHUNK_WORDS[RESUME] + 10)


class ProseChunkerTests(unittest.TestCase):
    def test_heading_detection(self):
        self.assertTrue(is_prose_heading("10.2.1 Teacher Forcing and Networks with Output Recurrence"))
        self.assertTrue(is_prose_heading("1 Introduction"))
        self.assertTrue(is_prose_heading("Chapter 10"))
        self.assertTrue(is_prose_heading("INTRODUCTION"))
        self.assertFalse(is_prose_heading("W W W W"))
        self.assertFalse(is_prose_heading("1. Regardless of the sequence length, the learned model always has the same"))
        self.assertFalse(is_prose_heading("This is an ordinary sentence."))

    def test_running_headers_and_page_numbers_are_removed(self):
        letters = "ABCDEFGH"
        document = [
            page(number, [
                "CHAPTER 10. SEQUENCE MODELING",
                f"Opening about {letters[number - 1]}.",
                f"Example {number}",
                f"Closing about {letters[number - 1]}.",
                str(366 + number),
            ])
            for number in range(1, 9)
        ]

        stripped = strip_repeated_blocks(document)

        # "Example N" repeats ignoring digits, but mid-page it is content.
        self.assertEqual(
            [p["blocks"] for p in stripped],
            [
                [f"Opening about {letter}.", f"Example {number}", f"Closing about {letter}."]
                for number, letter in enumerate(letters, start=1)
            ],
        )

    def test_sections_set_metadata_and_bound_chunks(self):
        document = [page(1, [
            "10.1 First Topic",
            "First topic sentence.",
            "10.2 Second Topic",
            "Second topic sentence.",
        ])]

        chunks = prose_chunker(document)

        self.assertEqual(
            [(chunk["metadata"]["section"], chunk["text"]) for chunk in chunks],
            [("10.1 First Topic", "First topic sentence."), ("10.2 Second Topic", "Second topic sentence.")],
        )

    def test_sentence_broken_across_pages_is_rejoined(self):
        document = [
            page(1, ["The year may appear in the sixth"]),
            page(2, ["word or in the second word of the sentence. Next sentence."]),
        ]

        chunks = prose_chunker(document)

        self.assertIn("in the sixth word or in the second word", chunks[0]["text"])

    def test_figure_label_noise_is_dropped_but_equations_kept(self):
        document = [page(1, [
            "Real explanatory sentence about recurrence.",
            "( ) t ( ) t ( +1) t ( +1) t",
            "h(t) = tanh(a(t)), (10.9)",
        ])]

        text = " ".join(chunk["text"] for chunk in prose_chunker(document))

        self.assertNotIn("( +1) t", text)
        self.assertIn("tanh", text)

    def test_chunks_respect_size_limit(self):
        sentences = " ".join(f"Sentence number {index} has a few words." for index in range(200))
        chunks = prose_chunker([page(1, [sentences])])

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c["text"].split()) <= MAX_CHUNK_WORDS[PROSE] for c in chunks))


class SlidesChunkerTests(unittest.TestCase):
    def test_one_chunk_per_slide_with_title_and_carried_divider(self):
        document = [
            page(1, ["Part II"], landscape=True),
            page(2, ["Gradient Descent", "Gradient descent updates parameters along the negative gradient."], landscape=True),
        ]

        chunks = slides_chunker(document)

        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0]["text"].startswith("Part II > Gradient Descent\n"))
        self.assertEqual(chunks[0]["metadata"]["page"], 2)


class ChunkDocumentTests(unittest.TestCase):
    def test_metadata_records_document_type(self):
        chunks = chunk_document(resume_document())

        self.assertTrue(chunks)
        self.assertTrue(all(chunk["metadata"]["document_type"] == RESUME for chunk in chunks))

    def test_explicit_type_overrides_classification(self):
        chunks = chunk_document(resume_document(), PROSE)

        self.assertTrue(all(chunk["metadata"]["document_type"] == PROSE for chunk in chunks))


def prose_chunk(index, text, section="10.3 Bidirectional RNNs", page_number=22):
    return {"text": text, "metadata": {
        "document_id": "doc", "chunk_index": index, "section": section,
        "page": page_number, "document_type": PROSE,
    }}


class ContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.stored = {
            0: prose_chunk(0, "Intro sentence. Shared sentence one.", page_number=21),
            1: prose_chunk(1, "Shared sentence one. Hit sentence. Shared sentence two."),
            2: prose_chunk(2, "Shared sentence two. Continuation."),
            3: prose_chunk(3, "Other section text.", section="10.4 Encoder-Decoder"),
        }
        self.fetch = lambda _document_id, indexes: [self.stored[i] for i in indexes if i in self.stored]

    def test_prose_hit_is_widened_with_same_section_neighbors_without_overlap(self):
        passages = expand_with_neighbors([self.stored[1]], self.fetch)

        self.assertEqual(len(passages), 1)
        self.assertEqual(
            passages[0]["text"],
            "Intro sentence. Shared sentence one. Hit sentence. Shared sentence two. Continuation.",
        )
        self.assertEqual(passages[0]["metadata"]["page"], 21)

    def test_neighbor_from_another_section_is_not_added(self):
        passages = expand_with_neighbors([self.stored[2]], self.fetch)

        self.assertNotIn("Other section", passages[0]["text"])

    def test_hit_already_included_as_neighbor_is_not_repeated(self):
        passages = expand_with_neighbors([self.stored[1], self.stored[2]], self.fetch)

        self.assertEqual(len(passages), 1)

    def test_non_prose_chunks_pass_through(self):
        resume_entry = {"text": "PROJECTS > Speakzy", "metadata": {
            "document_id": "doc", "chunk_index": 4, "document_type": RESUME}}

        self.assertEqual(expand_with_neighbors([resume_entry], self.fetch), [resume_entry])

    def test_context_labels_passages_with_page_and_section(self):
        context = format_context([self.stored[1], {"text": "bare", "metadata": {}}])

        self.assertEqual(
            context,
            "[1] | page 22 | 10.3 Bidirectional RNNs\n"
            "Shared sentence one. Hit sentence. Shared sentence two.\n\n[2]\nbare",
        )


def box(text, x0, y0, x1, height=20):
    return {"text": text, "x0": x0, "x1": x1, "y0": y0, "y1": y0 + height}


class CollectionNameTests(unittest.TestCase):
    def test_each_embedding_model_gets_its_own_collection(self):
        import vector_store

        self.assertEqual(vector_store.collection_name("sentence-transformers/all-mpnet-base-v2"), "pdf_documents")
        self.assertEqual(
            vector_store.collection_name("sentence-transformers/all-MiniLM-L6-v2"),
            "pdf_documents__sentence_transformers_all_MiniLM_L6_v2",
        )


class EmbedDocumentsTests(unittest.TestCase):
    def test_vectors_keep_input_order_and_progress_is_reported(self):
        import embedding

        class FakeModel:
            def __init__(self):
                self.batches = []

            def encode(self, texts):
                self.batches.append(list(texts))
                return [[float(len(text))] for text in texts]

            def get_sentence_embedding_dimension(self):
                return 1

        original_model, original_batch = embedding.model, embedding.DOCUMENT_BATCH_SIZE
        embedding.model, embedding.DOCUMENT_BATCH_SIZE = FakeModel(), 2
        try:
            progress = []
            texts = ["ccc", "a", "bbbbb", "dd"]

            vectors = embedding.embed_documents(texts, on_progress=lambda done, total: progress.append((done, total)))

            self.assertEqual(vectors.reshape(-1).tolist(), [3.0, 1.0, 5.0, 2.0])
            self.assertEqual(embedding.model.batches, [["a", "dd"], ["ccc", "bbbbb"]])
            self.assertEqual(progress, [(2, 4), (4, 4)])
            self.assertEqual(embedding.embed_documents([]).shape, (0, 1))
        finally:
            embedding.model, embedding.DOCUMENT_BATCH_SIZE = original_model, original_batch


class OcrLayoutTests(unittest.TestCase):
    WIDTH = 1000

    def test_title_and_right_aligned_date_stay_on_one_line(self):
        lines = [
            box("PROJECTS", 50, 100, 200),
            box("Speakzy - Language Platform", 50, 140, 400),
            box("Nov '25 - Mar '26", 800, 141, 950),
            box("• Built a full-stack platform.", 60, 165, 900),
        ]

        blocks = ocr.group_into_blocks(lines, self.WIDTH)

        self.assertIn("Speakzy - Language Platform Nov '25 - Mar '26", "\n".join(blocks))

    def test_table_is_read_row_by_row(self):
        lines = []
        for row, cells in enumerate([
            ("Year", "Degree", "Institute", "CGPA"),
            ("2027", "B.Tech", "Jamia Millia Islamia", "7/10"),
            ("2022", "Class XII", "STS School", "85%"),
        ]):
            for column, cell in enumerate(cells):
                lines.append(box(cell, 50 + column * 230, 100 + row * 25, 50 + column * 230 + 150))

        text = "\n".join(ocr.group_into_blocks(lines, self.WIDTH))

        self.assertIn("2027 B.Tech Jamia Millia Islamia 7/10", text)
        self.assertIn("2022 Class XII STS School 85%", text)

    def test_two_column_page_reads_left_column_first(self):
        lines = [box(f"left line {index}", 50, 100 + index * 25, 470) for index in range(6)]
        lines += [box(f"right line {index}", 530, 100 + index * 25, 950) for index in range(6)]

        blocks = ocr.group_into_blocks(lines, self.WIDTH)

        self.assertTrue(ocr.is_two_column(lines, self.WIDTH))
        self.assertEqual(blocks, [
            "\n".join(f"left line {index}" for index in range(6)),
            "\n".join(f"right line {index}" for index in range(6)),
        ])

    def test_a_few_right_aligned_dates_are_not_columns(self):
        lines = [box(f"Full width content line {index}", 50, 100 + index * 25, 950) for index in range(8)]
        lines += [box("Jun '26", 850, 300, 950), box("Feb '26", 850, 400, 950)]

        self.assertFalse(ocr.is_two_column(lines, self.WIDTH))

    def test_no_lines_gives_no_blocks(self):
        self.assertEqual(ocr.group_into_blocks([], self.WIDTH), [])


class PdfReaderTests(unittest.TestCase):
    def test_line_break_hyphen_joined_only_for_known_words(self):
        vocabulary = {"successful", "grapheme", "phoneme"}

        self.assertEqual(_join_hyphenated("suc-\ncessful", vocabulary), "successful")
        self.assertEqual(_join_hyphenated("grapheme-\nto-phoneme", vocabulary), "grapheme-to-phoneme")

    def test_extracts_from_upload_stream_with_lines_and_dot_leaders(self):
        pdf = pymupdf.open()
        pdf_page = pdf.new_page()
        pdf_page.insert_text((72, 72), "EDUCATION")
        pdf_page.insert_text((72, 90), "Contents . . . . . . . . 23")
        stream = BytesIO(pdf.tobytes())

        document = extract_document(stream)

        self.assertEqual(document[0]["page"], 1)
        all_text = " ".join(document[0]["blocks"])
        self.assertIn("EDUCATION", all_text)
        self.assertIn("Contents … 23", all_text)
        self.assertTrue(all(isinstance(lines, list) for lines in document[0]["lines"]))

    def test_image_only_pdf_yields_no_pages_without_ocr(self):
        pdf = pymupdf.open()
        pdf.new_page()

        self.assertEqual(extract_document(BytesIO(pdf.tobytes()), ocr=False), [])


class OcrExtractionTests(unittest.TestCase):
    """The OCR engine is replaced by a fake so these tests need no models."""

    def setUp(self):
        self.original = (ocr.is_available, ocr.ocr_page)
        self.ocr_calls = []

        def fake_ocr_page(page):
            self.ocr_calls.append(page.number)
            return ["Scanned paragraph about recurrent networks."], 0.95

        ocr.is_available = lambda: True
        ocr.ocr_page = fake_ocr_page

        pdf = pymupdf.open()
        pdf.new_page().insert_text((72, 72), "A page with a real text layer.")
        pdf.new_page()  # no text layer, like a scanned page
        self.stream = BytesIO(pdf.tobytes())

    def tearDown(self):
        ocr.is_available, ocr.ocr_page = self.original

    def test_only_pages_without_text_are_ocred(self):
        progress = []

        document = extract_document(self.stream, on_progress=lambda done, total: progress.append((done, total)))

        self.assertEqual(self.ocr_calls, [1])
        self.assertEqual(progress, [(1, 1)])
        self.assertNotIn("ocr", document[0])
        self.assertNotIn("ocr_confidence", document[0])
        self.assertEqual(document[1]["blocks"], ["Scanned paragraph about recurrent networks."])
        self.assertTrue(document[1]["ocr"])
        self.assertEqual(document[1]["ocr_confidence"], 0.95)

    def test_ocr_can_be_turned_off(self):
        document = extract_document(self.stream, ocr=False)

        self.assertEqual(self.ocr_calls, [])
        self.assertEqual([page["page"] for page in document], [1])

    def test_unavailable_engine_skips_ocr(self):
        ocr.is_available = lambda: False

        document = extract_document(self.stream)

        self.assertEqual(self.ocr_calls, [])
        self.assertEqual(len(document), 1)

    def test_low_confidence_marks_document_quality(self):
        import api

        self.assertIsNone(api.ocr_quality([{"page": 1, "blocks": ["text"]}]))
        self.assertEqual(api.ocr_quality([{"page": 1, "ocr": True, "ocr_confidence": 0.98}]), "good")
        self.assertEqual(
            api.ocr_quality([
                {"page": 1, "ocr": True, "ocr_confidence": 0.95},
                {"page": 2, "ocr": True, "ocr_confidence": 0.78},
            ]),
            "low",
        )


if __name__ == "__main__":
    unittest.main()
