import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

LLM_PROVIDER = os.getenv("LLM_PROVIDER")
LLAMA_MODEL = os.getenv("LLAMA_MODEL")
