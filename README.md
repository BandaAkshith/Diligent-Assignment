# Personal AI Assistant (Self-Hosted LLM + Pinecone)

A minimal end-to-end assistant with:
- **Self-hosted LLM** via `llama.cpp`
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
2. Install Python dependencies:
   - `pip install -r backend/requirements.txt`
3. Start backend:
   - `uvicorn backend.app:app --reload`
4. Open UI:
   - Open [frontend/index.html](frontend/index.html) in your browser.

## Ingesting Data
Use any REST client or curl:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"Your knowledge snippet."}'
```

## Notes
- The assistant will answer from Pinecone context only. If nothing is found, it will say it doesn't know.
- Ensure `LLM_MODEL_PATH` points to a local GGUF model.
