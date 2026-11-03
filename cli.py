"""StudyMate command line.

    python cli.py doctor            check Ollama, the model and the notes folder
    python cli.py ingest            (re)index data/notes  (only changed files are re-embedded)
    python cli.py ask "question"    one-shot question
    python cli.py chat              interactive chat with follow-up memory
    python cli.py eval              measure retrieval quality (no LLM needed)
"""

from __future__ import annotations

import argparse
import sys

from studymate import config
from studymate.llm import LLMUnavailable, OllamaClient
from studymate.rag import StudyMate
from studymate.store import KnowledgeBase


def _print_stats(stats: dict) -> None:
    print(
        f"Indexed: {stats['added']} new, {stats['updated']} updated, "
        f"{stats['skipped']} unchanged, {stats['removed']} removed "
        f"({stats['chunks']} new chunks)."
    )
    for name in stats["empty"]:
        print(f"  warning: no extractable text in {name} (scanned PDF?)")


def _ensure_index(kb: KnowledgeBase) -> None:
    stats = kb.index_folder()
    if stats["added"] or stats["updated"] or stats["removed"]:
        _print_stats(stats)
    if kb.count() == 0:
        sys.exit(f"No notes found. Put .md/.txt/.pdf files in {config.NOTES_DIR} and retry.")


def _answer(app: StudyMate, question: str, history, show_sources: bool = True) -> str:
    retrieval = app.retrieve(question, history)
    if retrieval.query != question:
        print(f"(searching for: {retrieval.query})")
    chunks = []
    print()
    try:
        for piece in app.stream_answer(question, retrieval, history):
            chunks.append(piece)
            print(piece, end="", flush=True)
    except LLMUnavailable as exc:
        print(f"[LLM unavailable] {exc}\nShowing the most relevant passages instead:\n")
        for i, h in enumerate(retrieval.hits, 1):
            print(f"[{i}] {h.label}\n{h.text}\n")
        return ""
    print()
    if show_sources and retrieval.in_scope:
        print("\nSources:")
        for i, h in enumerate(retrieval.hits, 1):
            d = "" if h.distance is None else f"  (distance {h.distance:.2f})"
            print(f"  [{i}] {h.label}{d}")
    return "".join(chunks)


def cmd_doctor(_args) -> int:
    ok = True
    llm = OllamaClient()
    try:
        models = llm.installed_models()
        print(f"[ok]   Ollama is running at {llm.base_url}")
        if llm.has_model():
            print(f"[ok]   Model '{llm.model}' is installed")
        else:
            ok = False
            print(f"[fail] Model '{llm.model}' missing. Run: ollama pull {llm.model}")
            if models:
                print(f"       Installed models: {', '.join(models)}")
    except LLMUnavailable as exc:
        ok = False
        print(f"[fail] {exc}")
    notes = list(config.NOTES_DIR.glob("**/*")) if config.NOTES_DIR.exists() else []
    print(f"[{'ok' if notes else 'fail'}]   Notes folder: {config.NOTES_DIR} ({len(notes)} files)")
    return 0 if ok and notes else 1


def cmd_ingest(_args) -> int:
    kb = KnowledgeBase()
    _print_stats(kb.index_folder())
    print(f"Total chunks in index: {kb.count()}  |  files: {', '.join(kb.sources())}")
    return 0


def cmd_ask(args) -> int:
    kb = KnowledgeBase()
    _ensure_index(kb)
    app = StudyMate(kb, OllamaClient(args.model))
    _answer(app, " ".join(args.question), [])
    return 0


def cmd_chat(args) -> int:
    kb = KnowledgeBase()
    _ensure_index(kb)
    app = StudyMate(kb, OllamaClient(args.model))
    history: list[tuple[str, str]] = []
    print("StudyMate chat - ask about your notes. Type 'exit' to quit, 'clear' to reset memory.")
    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            return 0
        if q.lower() == "clear":
            history.clear()
            print("Memory cleared.")
            continue
        answer = _answer(app, q, history)
        if answer:
            history.append((q, answer))


def cmd_eval(args) -> int:
    from studymate.evaluate import evaluate

    kb = KnowledgeBase()
    _ensure_index(kb)
    result = evaluate(kb, top_k=args.top_k)
    print("\nResults:")
    for key, value in result.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="StudyMate - local RAG for engineering notes")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check the setup").set_defaults(func=cmd_doctor)
    sub.add_parser("ingest", help="index the notes folder").set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask a single question")
    p.add_argument("question", nargs="+")
    p.add_argument("--model", default=None, help=f"Ollama model (default {config.LLM_MODEL})")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive chat")
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("eval", help="evaluate retrieval quality")
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
