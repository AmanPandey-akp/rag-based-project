"""The RAG pipeline: condense follow-up -> retrieve -> guard -> augment -> generate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from . import config
from .llm import LLMUnavailable, OllamaClient
from .store import Hit, KnowledgeBase

SYSTEM_PROMPT = """You are StudyMate, a precise teaching assistant for engineering students.
Answer the question using ONLY the numbered context passages provided.
Rules:
- If the context does not contain the answer, reply exactly: "I couldn't find that in your notes."
- Cite the passages you used like [1] or [2][3] right after the claims they support.
- Be clear and concise. Use short bullet points or numbered steps when it helps.
- Never invent facts, numbers or definitions that are not in the context."""

CONDENSE_PROMPT = """Rewrite the follow-up question as one standalone question, using the chat history
for context. Output ONLY the rewritten question, nothing else.

Chat history:
{history}

Follow-up question: {question}

Standalone question:"""

NOT_FOUND = "I couldn't find that in your notes."


@dataclass
class Retrieval:
    query: str                      # the (possibly rewritten) query actually searched
    hits: list[Hit] = field(default_factory=list)
    in_scope: bool = True           # False => closest chunk is too far away

    @property
    def best_distance(self) -> float | None:
        d = [h.distance for h in self.hits if h.distance is not None]
        return min(d) if d else None


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.label})\n{h.text}" for i, h in enumerate(hits, start=1))


class StudyMate:
    def __init__(self, kb: KnowledgeBase, llm: OllamaClient | None = None):
        self.kb = kb
        self.llm = llm or OllamaClient()

    # ----------------------------------------------------------- retrieval
    def _condense(self, question: str, history: list[tuple[str, str]]) -> str:
        if not history:
            return question
        recent = history[-config.HISTORY_TURNS :]
        text = "\n".join(f"Student: {q}\nAssistant: {a[:300]}" for q, a in recent)
        try:
            rewritten = self.llm.chat(
                [{"role": "user", "content": CONDENSE_PROMPT.format(history=text, question=question)}],
                temperature=0.0,
            ).strip()
            return rewritten.splitlines()[0] if rewritten else question
        except LLMUnavailable:
            return question

    def retrieve(self, question: str, history: list[tuple[str, str]] | None = None,
                 top_k: int | None = None) -> Retrieval:
        query = self._condense(question, history or [])
        hits = self.kb.search(query, top_k=top_k)
        result = Retrieval(query=query, hits=hits)
        best = result.best_distance
        result.in_scope = bool(hits) and best is not None and best <= config.MAX_DISTANCE
        return result

    # ---------------------------------------------------------- generation
    def build_messages(self, question: str, hits: list[Hit],
                       history: list[tuple[str, str]] | None = None) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for q, a in (history or [])[-config.HISTORY_TURNS :]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        messages.append(
            {
                "role": "user",
                "content": f"CONTEXT:\n{format_context(hits)}\n\nQUESTION: {question}",
            }
        )
        return messages

    def stream_answer(self, question: str, retrieval: Retrieval,
                      history: list[tuple[str, str]] | None = None) -> Iterator[str]:
        """Yield answer text. Out-of-scope questions never reach the LLM."""
        if not retrieval.in_scope:
            yield NOT_FOUND
            return
        messages = self.build_messages(question, retrieval.hits, history)
        yield from self.llm.stream(messages)
