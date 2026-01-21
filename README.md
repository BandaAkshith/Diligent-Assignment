# Personal AI Assistant (Self-Hosted Ollama + Pinecone)

A minimal end-to-end assistant with:
- **Self-hosted LLaMA-family LLM** via Ollama
- **Vector database** via Pinecone
- **Chat UI** (static HTML)

## Project Structure
- [backend/app.py](backend/app.py)
- [backend/requirements.txt](backend/requirements.txt)
- [frontend/index.html](frontend/index.html)
- [.env.example](.env.example)

## Setup

1. Copy environment variables:
   - Duplicate [.env.example](.env.example) to `.env` and fill in values.
   - Set `OLLAMA_MODEL` to a local Ollama model (e.g., `llama3.1`).
2. Install Python dependencies:
   - `pip install -r backend/requirements.txt`
3. Start backend:
   - `uvicorn backend.app:app --reload`
4. Open UI:
   - Open [frontend/index.html](frontend/index.html) in your browser.

## Ingesting Data
Use any REST client, curl, or the UI ingest box:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"Your knowledge snippet."}'
```

## Notes
- The assistant answers strictly from Pinecone context; otherwise it says it doesn't know.
- Pinecone can be used with API key + index name (or direct host).
- Ensure Ollama is running locally.
