import importlib
import uuid
from typing import Any, Dict, List, Optional, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

import config

app = FastAPI(title="Personal AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder: Any = SentenceTransformer(config.EMBEDDING_MODEL)

pinecone_module = None
try:
    pinecone_module = importlib.import_module("pinecone")
except Exception:
    pinecone_module = None

ollama_module = None
try:
    ollama_module = importlib.import_module("ollama")
except Exception:
    ollama_module = None

pc = (
    pinecone_module.Pinecone(api_key=config.PINECONE_API_KEY)
    if pinecone_module and config.PINECONE_API_KEY
    else None
)
index = None

if pc and pinecone_module:
    if config.PINECONE_HOST:
        index = pc.Index(host=config.PINECONE_HOST)
    else:
        existing = [idx["name"] for idx in pc.list_indexes()]
        if config.PINECONE_INDEX not in existing:
            pc.create_index(
                name=config.PINECONE_INDEX,
                dimension=config.EMBEDDING_DIM,
                metric="cosine",
                spec=pinecone_module.ServerlessSpec(
                    cloud=config.PINECONE_CLOUD,
                    region=config.PINECONE_REGION,
                ),
            )
        index = pc.Index(config.PINECONE_INDEX)

llm_ready = bool(ollama_module)


class IngestRequest(BaseModel):
    text: str
    id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


def embed_text(text: str) -> List[float]:
    vector = embedder.encode([text], normalize_embeddings=True)[0]
    return cast(List[float], vector.tolist())


def build_prompt(user_message: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(contexts)
    system = (
        "You are a helpful enterprise assistant. Use the provided context to answer. "
        "If the answer is not in the context, say you don't know."
    )
    return (
        f"System: {system}\n\n"
        f"Context:\n{context_block}\n\n"
        f"User: {user_message}\n"
        "Assistant:"
    )


def generate_answer(prompt: str) -> str:
    if not ollama_module:
        return "Ollama is not available. Please install and run Ollama."
    if not config.OLLAMA_MODEL:
        return "OLLAMA_MODEL is not configured. Please set it in your environment."

    response = ollama_module.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful enterprise assistant."},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": config.LLM_TEMPERATURE,
            "num_ctx": config.LLM_CTX,
            "num_predict": config.LLM_MAX_TOKENS,
        },
    )
    return response["message"]["content"].strip()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "pinecone": bool(index),
        "llm": llm_ready,
    }


@app.post("/ingest")
def ingest(payload: IngestRequest):
    if not index:
        return {"error": "Pinecone is not configured. Set PINECONE_API_KEY."}

    vector = embed_text(payload.text)
    vector_id = payload.id or str(uuid.uuid4())
    metadata = payload.metadata or {}
    metadata["text"] = payload.text

    index.upsert([(vector_id, vector, metadata)])
    return {"id": vector_id}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if not index:
        return ChatResponse(answer="Pinecone is not configured.", sources=[])

    query_vector = embed_text(payload.message)
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    contexts: List[str] = []
    sources: List[Dict[str, Any]] = []
    for match in results.get("matches", []):
        metadata = cast(Dict[str, Any], match.get("metadata", {}) or {})
        text = cast(str, metadata.get("text", ""))
        if text:
            contexts.append(text)
            sources.append(
                {
                    "id": match.get("id"),
                    "score": match.get("score"),
                    "metadata": metadata,
                }
            )

    prompt = build_prompt(payload.message, contexts)
    answer = generate_answer(prompt)

    return ChatResponse(answer=answer, sources=sources)
