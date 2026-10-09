# PDF Assistant

A local V2 retrieval-augmented generation (RAG) application. Upload a PDF and
ask questions grounded in its contents. Retrieval combines MPNet semantic
search with BM25 keyword search, reciprocal rank fusion, and local
cross-encoder reranking.

## Ingestion pipeline

When a PDF is uploaded, the backend:

1. Extracts text blocks and their lines per page with PyMuPDF
   (`pdf_reader.py`). Image-only (scanned) PDFs have no text layer and are
   rejected with status `failed`.
2. Classifies the document as `resume`, `slides`, or `prose`
   (`document_classifier.py`) from layout signals: page count, contact
   details, resume section headings, page orientation, and words per page.
3. Chunks it with the matching strategy (`chunker.py`):
   - **resume**: one chunk per entry (job, project, degree, skill group),
     prefixed with `SECTION > entry title`.
   - **slides**: one chunk per slide, titled by the slide heading.
   - **prose** (books, papers, reports, notes): sentence-packed chunks of up
     to about 150 words that never cross a numbered section heading, with
     running headers, page numbers, and figure-label noise removed.
4. Embeds the chunks with MPNet and stores them in Chroma with their page,
   section, and document type.

## Retrieval pipeline

For each question, the backend:

1. Rewrites the query when conversation history requires it.
2. Retrieves the top 10 chunks with MPNet vector search.
3. Retrieves the top 10 chunks with BM25 keyword search.
4. Fuses both rankings with reciprocal rank fusion.
5. Reranks the fused candidates with a local cross-encoder and keeps the top 3.
6. For prose documents, widens each of those chunks with its neighbors from
   the same section (`context_builder.py`), then sends the passages to the
   model labelled with their page and section.

The vector store is persisted locally under `backend/chroma_db/`. It is runtime
data and is intentionally excluded from Git. Upload a document locally before
using the chat application. Documents stored before the type-aware chunker was
introduced keep their old chunks; upload them again to benefit from it.

Calls to OpenRouter are retried with 2s, 4s, and 8s backoff on per-minute rate
limits and connection errors. A daily free-tier quota error is returned
immediately, since retrying cannot succeed until the quota resets.

## Backend setup

Create and activate a Python virtual environment, then install the backend packages:

```bash
pip install fastapi "uvicorn[standard]" python-multipart pypdf pymupdf chromadb sentence-transformers openai python-dotenv nltk tenacity
python -m nltk.downloader punkt punkt_tab
```

PDF extraction uses PyMuPDF, which is licensed under the AGPL. Check that this
fits how you distribute the project.

Create `backend/.env`:

```env
OPEN_ROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# Optional: defaults to nvidia/nemotron-3.5-lightning:free
OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free
# Optional upload limits: defaults to 20 MiB and 300 pages
MAX_UPLOAD_SIZE_BYTES=20971520
MAX_PDF_PAGES=300
```

`MAX_UPLOAD_SIZE_BYTES` controls the maximum uploaded PDF size. `MAX_PDF_PAGES`
controls the maximum number of pages accepted during PDF preflight.

Start the API from the `backend/` directory:

```bash
cd backend
uvicorn api:app --reload
```

## Frontend setup

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite development server proxies `/upload` and `/ask` to the backend at `http://127.0.0.1:8000`.

## Tests

Run the unit tests from `backend/`:

```bash
cd backend
python -m unittest test_chunking test_generation
```

## Quality benchmark

`backend/tests/benchmark_long_document.py` ingests a PDF into a throwaway
in-memory collection (it never touches `chroma_db/`) and reports ingestion
diagnostics plus retrieval quality for MPNet-only, BM25-only, and the V2
pipeline: Hit@1/3/10, MRR, page hit rate, and ContextHit (whether the evidence
reaches the model after neighbor expansion). With `--generate` it also asks the
configured model every question and grades the answers.

Gold labels are evidence phrases rather than chunk indexes, so they stay valid
when extraction or chunking changes. Two datasets are included: chapter 10 of
the Deep Learning book (`long_document_dataset.py`) and a resume
(`resume_dataset.py`). Run from the repository root:

```bash
python -m backend.tests.benchmark_long_document --pdf "path/to/Sequence Modeling.pdf"
python -m backend.tests.benchmark_long_document --pdf path/to/resume.pdf --dataset resume
python -m backend.tests.benchmark_long_document --pdf "..." --generate --report report.json
```

`--generate` makes one model call per question (26 for the book dataset). The
OpenRouter free tier allows 50 requests per day, and the benchmark stops
calling the provider after three consecutive errors.

### Results

Measured on the book chapter (23 questions) and the resume (12 questions).
Each book question is worth about 0.04, so treat single-question differences
as noise. Answers were graded with the benchmark's keyword check, which agreed
with manual grading.

| Pipeline | Book chunks | Book Hit@1 | Book Hit@3 | Book Hit@10 | Book MRR | Resume Hit@3 | Book answers correct |
|---|---|---|---|---|---|---|---|
| pypdf + original chunker | 400 | 0.48 | 0.78 | 0.87 | 0.63 | 0.83 | 14/23 |
| PyMuPDF + original chunker | 314 | 0.78 | 0.96 | 0.96 | 0.85 | 1.00 | not measured |
| PyMuPDF + type-aware chunker | 144 | 0.87 | 0.96 | 1.00 | 0.91 | 1.00 | 21/23 |

The original chunker produced 132 book chunks under 15 words and 208 that
started mid-sentence, and dropped the resume's education rows. Neighbor
expansion raised the context sent to the model from about 390 to about 670
words per book question while keeping the evidence in it for 96% of
questions; its effect on answer quality has not been measured yet.

## Retrieval evaluation (legacy)

The chunk-index labels in `evaluation_dataset.py` describe a resume ingested
with the original chunker, so they no longer match documents chunked by the
current pipeline. Prefer the quality benchmark above.

The local evaluation suite compares the previous MPNet-only pipeline (V1)
with the V2 hybrid and reranked pipeline. It uses the stored resume benchmark
questions and reports Recall@3, Recall@10, MRR, NDCG@3, NDCG@10, and
Precision@3. It does not call the answer generator.

Run it from the repository root after the benchmark document has been ingested:

```bash
python -m backend.tests.evaluation_suite
```

The evaluation implementation is in
`backend/tests/evaluation_suite.py`, and the gold chunk identities are in
`backend/tests/evaluation_dataset.py`.

## Workflow

1. Open the frontend URL shown by Vite.
2. Upload a PDF. The API returns `202 Accepted` with a `document_id` while processing continues in the background.
3. Poll `GET /upload/{document_id}/status` until its status is `ready`, then ask questions about the document. Include the `document_id` in each `/ask` request.
4. Expand an answer's **Sources** section to inspect the retrieved chunks.

Uploading another PDF starts a new in-memory conversation.
