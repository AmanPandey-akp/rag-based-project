"""Heading-aware, overlapping word-window chunking.

Why this beats "split on blank lines":
  * Markdown headings are kept, so every chunk knows which topic it belongs to
    ("Deadlocks > Coffman conditions"). The heading is prepended to the chunk text,
    which helps both the embedding and the LLM.
  * Long sections are cut into windows of ~CHUNK_WORDS words that overlap, so a fact
    sitting on a boundary is never lost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    section: str
    index: int


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text into (section_path, body) pairs."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))
        buffer.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buffer.append(line)
    flush()
    return sections


def window_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    if len(words) <= size:
        return [words] if words else []
    step = size - overlap
    windows = []
    for start in range(0, len(words), step):
        windows.append(words[start : start + size])
        if start + size >= len(words):
            break
    return windows


def chunk_document(
    pages: list[tuple[int | None, str]],
    source: str,
    chunk_words: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    chunk_words = chunk_words or config.CHUNK_WORDS
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    chunks: list[Chunk] = []
    for page, text in pages:
        for section, body in split_sections(text):
            for window in window_words(body.split(), chunk_words, overlap):
                body_text = " ".join(window)
                full = f"{section}\n{body_text}" if section else body_text
                idx = len(chunks)
                chunks.append(
                    Chunk(
                        id=f"{source}::{idx}",
                        text=full,
                        source=source,
                        page=page,
                        section=section,
                        index=idx,
                    )
                )
    return chunks
