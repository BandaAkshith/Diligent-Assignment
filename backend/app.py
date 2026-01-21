import os
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from pinecone import Pinecone, ServerlessSpec
from llama_cpp import Llama

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "assistant-knowledge")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_HOST = os.getenv("PINECONE_HOST", "")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "")
LLM_CTX = int(os.getenv("LLM_CTX", "4096"))
LLM_THREADS = int(os.getenv("LLM_THREADS", "6"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

app = FastAPI(title="Personal AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = SentenceTransformer(EMBEDDING_MODEL)

pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
index = None

if pc:
    if PINECONE_HOST:
        index = pc.Index(host=PINECONE_HOST)
    else:
        existing = [idx["name"] for idx in pc.list_indexes()]
        if PINECONE_INDEX not in existing:
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            )
        index = pc.Index(PINECONE_INDEX)

llm = None
if LLM_MODEL_PATH:
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_ctx=LLM_CTX,
        n_threads=LLM_THREADS,
        verbose=False,
    )


class IngestRequest(BaseModel):
    text: str
    id: Optional[str] = None
    metadata: Optional[dict] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


def embed_text(text: str) -> List[float]:
    vector = embedder.encode([text], normalize_embeddings=True)[0]
    return vector.tolist()


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
    if not llm:
        return "LLM_MODEL_PATH is not configured. Please set it in your environment."
    output = llm(
        prompt,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        stop=["User:", "System:"],
    )
    return output["choices"][0]["text"].strip()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pinecone": bool(index),
        "llm": bool(llm),
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

    contexts = []
    sources = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {}) or {}
        text = metadata.get("text", "")
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
