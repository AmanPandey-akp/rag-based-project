"""Central configuration. Every value can be overridden with an environment variable."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where your notes live (.md / .txt / .pdf) and where the vector index is stored.
NOTES_DIR = Path(os.getenv("STUDYMATE_NOTES_DIR", BASE_DIR / "data" / "notes"))
DB_DIR = Path(os.getenv("STUDYMATE_DB_DIR", BASE_DIR / "vector_store"))
COLLECTION_NAME = "studymate"

# Free local LLM served by Ollama (https://ollama.com). No API key, no cost, works offline.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("STUDYMATE_MODEL", "llama3.2:3b")
LLM_TEMPERATURE = float(os.getenv("STUDYMATE_TEMPERATURE", "0.2"))
LLM_CONTEXT_TOKENS = int(os.getenv("STUDYMATE_NUM_CTX", "4096"))
LLM_TIMEOUT_SECONDS = int(os.getenv("STUDYMATE_LLM_TIMEOUT", "300"))

# Chunking (measured in words).
CHUNK_WORDS = int(os.getenv("STUDYMATE_CHUNK_WORDS", "160"))
CHUNK_OVERLAP = int(os.getenv("STUDYMATE_CHUNK_OVERLAP", "40"))

# Retrieval.
TOP_K = int(os.getenv("STUDYMATE_TOP_K", "4"))          # chunks sent to the LLM
CANDIDATES = int(os.getenv("STUDYMATE_CANDIDATES", "12"))  # per retriever before fusion
RRF_K = 60                                               # standard Reciprocal Rank Fusion constant

# If even the closest chunk is farther than this (cosine distance, 0 = identical,
# 2 = opposite) the question is treated as "not in my notes" and the LLM is not called.
MAX_DISTANCE = float(os.getenv("STUDYMATE_MAX_DISTANCE", "0.75"))

# Number of previous chat turns used for follow-up questions.
HISTORY_TURNS = 3
