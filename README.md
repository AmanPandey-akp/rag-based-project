# StudyMate: a free, offline RAG assistant for engineering notes

Ask questions about your own B.Tech notes (DBMS, OS, CN, DSA, anything) and get answers
**grounded in your documents, with citations**. Everything runs on your own computer:

- **LLM:** open-weight model served by [Ollama](https://ollama.com), with no API key, no billing and no usage limit.
- **Embeddings:** a small local model (all-MiniLM-L6-v2). It is downloaded once, then works offline.
- **Vector DB:** ChromaDB, stored in a local folder.

## What makes it more than a toy

| Feature | Why it matters |
|---|---|
| Reads **PDF, Markdown and TXT** (PDFs are page-aware) | Real notes are PDFs |
| **Heading-aware chunking with overlap** | Each chunk keeps its topic path, e.g. `Deadlocks > Coffman conditions` |
| **Hybrid search**: vector + BM25 keyword, merged with Reciprocal Rank Fusion | Finds both meaning ("avoid waiting forever") and exact terms ("B+ tree", "TCP") |
| **Incremental indexing** (SHA-256 per file) | Re-running only embeds new or changed files and drops deleted ones |
| **Cited answers** `[1][2]` plus a sources panel | You can verify every claim |
| **Out-of-scope guard** | If nothing in your notes is close enough, the LLM is not called and it says "I couldn't find that in your notes" |
| **Follow-up questions** (chat memory + query rewriting) | "How do I prevent *it*?" works |
| **Streaming** answers in terminal and web UI | Feels fast even on a laptop |
| **Evaluation script** (Hit@k, MRR, refusal rate) | Measure changes instead of guessing |
| **Unit tests** (13, run offline in seconds) | Safe to modify |

## Setup (about 10 minutes, one time)

**1. Install Ollama** (free): download from <https://ollama.com/download> (Windows / macOS / Linux) and start it.

**2. Download a model** (about 2 GB, once):
```bash
ollama pull llama3.2:3b
```
Short on RAM (4 GB or less)? Use `ollama pull llama3.2:1b` or `ollama pull qwen2.5:1.5b`.
Have 16 GB or more? Try `ollama pull qwen2.5:7b` for better answers.
Switch model with `set STUDYMATE_MODEL=qwen2.5:7b` (Windows cmd) or `export STUDYMATE_MODEL=qwen2.5:7b` (Mac/Linux).

**3. Python environment** (Python 3.10 or newer):
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac / Linux
pip install -r requirements.txt
```

**4. Check everything:**
```bash
python cli.py doctor
```

## Use it

```bash
python cli.py ingest                      # index data/notes (the first run downloads the embedding model, about 80 MB)
python cli.py ask "What are the four Coffman conditions?"
python cli.py chat                        # interactive, remembers follow-ups
streamlit run app.py                      # web UI with file upload
python cli.py eval                        # retrieval quality report (no LLM needed)
pytest                                    # run the tests
```

**Use your own notes:** drop `.pdf`, `.md` or `.txt` files into `data/notes/` (or upload them in the web UI) and run
`python cli.py ingest`. Only the new files are processed. Scanned PDFs with no text layer cannot be read (they need OCR).

## How it works

```
 INDEXING (once per file change)
 notes/*.pdf|md|txt -> load -> heading-aware chunks (+overlap) -> embed -> ChromaDB
                                                    \-> BM25 keyword index (built in memory)

 QUESTION
 follow-up? -> rewrite as standalone question (LLM)
            -> vector search (top 12) + BM25 search (top 12) -> Reciprocal Rank Fusion -> top 4
            -> closest chunk too far away? -> "I couldn't find that in your notes."  (no LLM call)
            -> prompt = rules + numbered context + question -> local LLM (streamed) -> cited answer
```

Code map: `studymate/loaders.py` (read files), `chunking.py`, `store.py` (index and hybrid search), `rag.py` (pipeline and prompts),
`llm.py` (Ollama client), `evaluate.py`, plus `cli.py` and `app.py` (interfaces).

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `STUDYMATE_MODEL` | `llama3.2:3b` | Any model you have pulled in Ollama |
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama is running |
| `STUDYMATE_NOTES_DIR` | `data/notes` | Folder with your notes |
| `STUDYMATE_DB_DIR` | `vector_store` | Where the index is saved |
| `STUDYMATE_CHUNK_WORDS` / `_CHUNK_OVERLAP` | `160` / `40` | Chunk size and overlap (re-index after changing: delete `vector_store/`) |
| `STUDYMATE_TOP_K` | `4` | Passages given to the LLM |
| `STUDYMATE_MAX_DISTANCE` | `0.75` | Out-of-scope cut-off (see below) |

### Tuning the out-of-scope guard
`MAX_DISTANCE` is a cosine distance: lower means more similar. The default `0.75` is a starting guess that I could not calibrate against the real
embedding model while building this. Run `python cli.py eval` on your machine. It prints the best distance for the out-of-scope questions
in `eval/questions.json`. If unrelated questions would still be answered, lower the value; if good questions get refused, raise it.
Add your own questions to `eval/questions.json` (`"source": null` marks an out-of-scope question).

## Troubleshooting

- **"Cannot reach Ollama"**: open the Ollama app or run `ollama serve`, then `python cli.py doctor`.
- **"Model ... is not downloaded"**: run `ollama pull <model>` using the exact name shown.
- **First run is slow or fails**: the embedding model is downloaded once, so connect to the internet for that first `ingest`.
- **Slow answers**: use a smaller model (`llama3.2:1b`). A GPU is not required.
- **Answers ignore your newest notes**: run `python cli.py ingest` (the web UI has a "Re-scan" button).

## Ideas to extend it (good for a final-year project)

- Add a cross-encoder re-ranker on the top 12 candidates.
- OCR for scanned PDFs (`pytesseract`).
- Per-subject collections with a subject filter, and a "generate quiz from this topic" mode.
- FastAPI backend and Docker packaging.
- Automatic answer-faithfulness checks added to `evaluate.py`.
