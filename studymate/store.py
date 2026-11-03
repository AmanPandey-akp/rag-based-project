"""Knowledge base: incremental indexing + hybrid (vector + BM25) retrieval.

Hybrid search matters for engineering content. Pure vector search is great at meaning
("how do processes avoid waiting forever?" -> deadlock) but can miss exact technical
tokens ("B+ tree", "TCP", "ACID", "LRU"). BM25 keyword search is the opposite. We run
both and merge the rankings with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from . import config
from .chunking import Chunk, chunk_document
from .loaders import file_hash, list_files, load_file

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on", "for", "and",
    "or", "it", "its", "this", "that", "with", "as", "by", "at", "from", "what", "which", "who",
    "how", "why", "when", "do", "does", "did", "can", "i", "you", "me", "my", "explain", "tell",
}
_TOKEN = re.compile(r"[a-z0-9+#]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class Hit:
    id: str
    text: str
    source: str
    page: int | None
    section: str
    score: float                  # fused RRF score (higher = better)
    distance: float | None = None  # cosine distance from the vector search (lower = better)

    @property
    def label(self) -> str:
        parts = [self.source]
        if self.page:
            parts.append(f"p.{self.page}")
        if self.section:
            parts.append(self.section)
        return " | ".join(parts)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = config.RRF_K) -> dict[str, float]:
    """Merge several ranked id-lists into one score per id."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class KnowledgeBase:
    def __init__(self, db_dir: Path | str | None = None, embedding_function=None,
                 collection_name: str | None = None):
        self.db_dir = str(db_dir or config.DB_DIR)
        if embedding_function is None:
            from chromadb.utils import embedding_functions

            # Small local model (all-MiniLM-L6-v2). Downloaded once, then fully offline.
            embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.client = chromadb.PersistentClient(
            path=self.db_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name or config.COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_loaded_count = -1

    # ------------------------------------------------------------- indexing
    def _indexed_sources(self) -> dict[str, str]:
        """Map source filename -> file hash currently stored in the index."""
        data = self.collection.get(include=["metadatas"])
        return {m["source"]: m.get("file_hash", "") for m in data["metadatas"]}

    def _add_chunks(self, chunks: list[Chunk], fhash: str) -> None:
        for start in range(0, len(chunks), 64):
            batch = chunks[start : start + 64]
            self.collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "source": c.source,
                        "page": c.page if c.page is not None else 0,
                        "section": c.section,
                        "file_hash": fhash,
                    }
                    for c in batch
                ],
            )

    def index_folder(self, folder: Path | str | None = None) -> dict:
        """Index new/changed files, skip unchanged ones, drop deleted ones."""
        folder = Path(folder or config.NOTES_DIR)
        existing = self._indexed_sources()
        stats = {"added": 0, "updated": 0, "skipped": 0, "removed": 0, "chunks": 0, "empty": []}
        seen: set[str] = set()

        for path in list_files(folder):
            source = str(path.relative_to(folder)).replace("\\", "/")
            seen.add(source)
            fhash = file_hash(path)
            if existing.get(source) == fhash:
                stats["skipped"] += 1
                continue
            if source in existing:
                self.collection.delete(where={"source": source})
                stats["updated"] += 1
            else:
                stats["added"] += 1
            chunks = chunk_document(load_file(path), source)
            if not chunks:
                stats["empty"].append(source)
                continue
            self._add_chunks(chunks, fhash)
            stats["chunks"] += len(chunks)

        for source in set(existing) - seen:
            self.collection.delete(where={"source": source})
            stats["removed"] += 1

        self._bm25 = None  # force rebuild
        return stats

    def count(self) -> int:
        return self.collection.count()

    def sources(self) -> list[str]:
        return sorted(self._indexed_sources())

    # ------------------------------------------------------------ retrieval
    def _ensure_bm25(self) -> None:
        n = self.collection.count()
        if self._bm25 is not None and self._bm25_loaded_count == n:
            return
        from rank_bm25 import BM25Okapi

        data = self.collection.get(include=["documents"])
        self._bm25_ids = data["ids"]
        self._bm25_docs = data["documents"]
        corpus = [tokenize(d) for d in self._bm25_docs]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._bm25_loaded_count = n

    def _keyword_ranking(self, query: str, limit: int) -> list[str]:
        self._ensure_bm25()
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._bm25_ids[i] for i in order[:limit] if scores[i] > 0]

    def search(self, query: str, top_k: int | None = None,
               candidates: int | None = None) -> list[Hit]:
        top_k = top_k or config.TOP_K
        candidates = candidates or config.CANDIDATES
        total = self.collection.count()
        if total == 0:
            return []

        res = self.collection.query(
            query_texts=[query],
            n_results=min(candidates, total),
            include=["documents", "metadatas", "distances"],
        )
        vec_ids = res["ids"][0]
        info: dict[str, tuple[str, dict, float | None]] = {
            i: (d, m, dist)
            for i, d, m, dist in zip(
                vec_ids, res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        }

        kw_ids = self._keyword_ranking(query, candidates)
        missing = [i for i in kw_ids if i not in info]
        if missing:
            extra = self.collection.get(ids=missing, include=["documents", "metadatas"])
            for i, d, m in zip(extra["ids"], extra["documents"], extra["metadatas"]):
                info[i] = (d, m, None)

        fused = reciprocal_rank_fusion([vec_ids, kw_ids])
        best = sorted(fused, key=fused.get, reverse=True)[:top_k]
        hits = []
        for doc_id in best:
            text, meta, dist = info[doc_id]
            hits.append(
                Hit(
                    id=doc_id,
                    text=text,
                    source=meta["source"],
                    page=meta.get("page") or None,
                    section=meta.get("section", ""),
                    score=fused[doc_id],
                    distance=dist,
                )
            )
        return hits
