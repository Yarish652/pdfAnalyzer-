# PDF Assistant

A local V2 retrieval-augmented generation (RAG) application. Upload a PDF and
ask questions grounded in its contents. Retrieval combines MPNet semantic
search with BM25 keyword search, reciprocal rank fusion, and local
cross-encoder reranking.

## Retrieval pipeline

For each question, the backend:

1. Rewrites the query when conversation history requires it.
2. Retrieves the top 10 chunks with MPNet vector search.
3. Retrieves the top 10 chunks with BM25 keyword search.
4. Fuses both rankings with reciprocal rank fusion.
5. Reranks the fused candidates with a local cross-encoder and uses the top 3
	chunks for answer generation.

The vector store is persisted locally under `backend/chroma_db/`. It is runtime
data and is intentionally excluded from Git. Upload a document locally before
using the chat application.

## Backend setup

Create and activate a Python virtual environment, then install the backend packages:

```bash
pip install fastapi "uvicorn[standard]" python-multipart pypdf chromadb sentence-transformers openai python-dotenv nltk tenacity
python -m nltk.downloader punkt punkt_tab
```

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

Start the API from the repository root:

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

## Retrieval evaluation

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
