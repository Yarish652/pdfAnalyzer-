# PDF Assistant

A local V1 RAG application: upload a PDF, then ask questions grounded in its contents.

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

Start the API from the backend directory:

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

## Workflow

1. Open the frontend URL shown by Vite.
2. Upload a PDF. The API returns `202 Accepted` with a `document_id` while processing continues in the background.
3. Poll `GET /upload/{document_id}/status` until its status is `ready`, then ask questions about the document. Include the `document_id` in each `/ask` request.
4. Expand an answer's **Sources** section to inspect the retrieved chunks.

Uploading another PDF starts a new in-memory conversation.
