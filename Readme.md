# PDF Assistant

A local V1 RAG application: upload a PDF, then ask questions grounded in its contents.

## Backend setup

Create and activate a Python virtual environment, then install the backend packages:

```bash
pip install fastapi "uvicorn[standard]" python-multipart pypdf chromadb sentence-transformers openai python-dotenv nltk
python -m nltk.downloader punkt punkt_tab
```

Create `backend/.env`:

```env
OPEN_ROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

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
2. Upload and process a PDF.
3. Ask questions about the document.
4. Expand an answer's **Sources** section to inspect the retrieved chunks.

Uploading another PDF starts a new in-memory conversation.
