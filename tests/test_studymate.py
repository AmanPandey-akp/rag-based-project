"""Unit tests. They use a tiny deterministic fake embedder so they run offline in seconds."""

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest
from chromadb import EmbeddingFunction

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studymate import config
from studymate.chunking import chunk_document, split_sections, window_words
from studymate.rag import NOT_FOUND, StudyMate, format_context
from studymate.store import KnowledgeBase, reciprocal_rank_fusion, tokenize


class FakeEmbedding(EmbeddingFunction):
    """Hashing bag-of-words embedder: same words -> close vectors. No downloads needed."""

    DIM = 256

    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            v = [0.0] * self.DIM
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    @staticmethod
    def name():
        return "fake-hash-embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEmbedding()

    def default_space(self):
        return "cosine"

    def supported_spaces(self):
        return ["cosine", "l2", "ip"]


class FakeLLM:
    def __init__(self):
        self.calls = []

    def stream(self, messages, temperature=None):
        self.calls.append(messages)
        yield "answer "
        yield "[1]"

    def chat(self, messages, temperature=None):
        self.calls.append(messages)
        return "rewritten standalone question"


@pytest.fixture
def notes(tmp_path):
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "os.md").write_text(
        "# OS\n\n## Deadlocks\nA deadlock happens when processes wait forever. "
        "The four Coffman conditions are mutual exclusion, hold and wait, no preemption "
        "and circular wait.\n\n## Paging\nPaging divides memory into fixed size frames "
        "and the TLB caches page table translations.\n",
        encoding="utf-8",
    )
    (folder / "net.txt").write_text(
        "TCP uses a three way handshake with SYN and ACK packets to open a connection.",
        encoding="utf-8",
    )
    return folder


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(db_dir=tmp_path / "db", embedding_function=FakeEmbedding())


# ------------------------------------------------------------- chunking
def test_split_sections_keeps_heading_path():
    sections = split_sections("# A\nintro\n## B\nbody b\n## C\nbody c")
    assert [s for s, _ in sections] == ["A", "A > B", "A > C"]


def test_window_words_overlap_and_coverage():
    words = [str(i) for i in range(100)]
    windows = window_words(words, 30, 10)
    assert windows[0][-10:] == windows[1][:10]
    assert windows[-1][-1] == "99"
    assert set(w for win in windows for w in win) == set(words)


def test_window_words_rejects_bad_overlap():
    with pytest.raises(ValueError):
        window_words(["a"] * 10, 5, 5)


def test_chunk_document_prepends_section_and_unique_ids():
    text = "# Topic\n" + " ".join(f"w{i}" for i in range(300))
    chunks = chunk_document([(None, text)], "x.md", chunk_words=100, overlap=20)
    assert len(chunks) > 2
    assert all(c.text.startswith("Topic\n") for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


# ------------------------------------------------------------ retrieval
def test_rrf_prefers_items_ranked_high_in_both_lists():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert ranked[:2] == ["a", "b"] or ranked[:2] == ["b", "a"]
    assert scores["d"] < scores["a"]


def test_tokenize_drops_stopwords_keeps_technical_tokens():
    assert tokenize("What is a B+ tree in C#?") == ["b+", "tree", "c#"]


def test_index_search_finds_right_file(kb, notes):
    stats = kb.index_folder(notes)
    assert stats["added"] == 2 and stats["chunks"] >= 3
    hits = kb.search("What are the Coffman conditions for deadlock?", top_k=2)
    assert hits[0].source == "os.md"
    assert "Deadlocks" in hits[0].section
    hits = kb.search("three way handshake", top_k=2)
    assert hits[0].source == "net.txt"


def test_incremental_indexing(kb, notes):
    kb.index_folder(notes)
    again = kb.index_folder(notes)
    assert again["skipped"] == 2 and again["added"] == 0 and again["chunks"] == 0

    (notes / "net.txt").write_text("UDP is connectionless and fast.", encoding="utf-8")
    changed = kb.index_folder(notes)
    assert changed["updated"] == 1 and changed["skipped"] == 1
    assert "UDP" in kb.search("udp connectionless", top_k=1)[0].text

    (notes / "net.txt").unlink()
    removed = kb.index_folder(notes)
    assert removed["removed"] == 1
    assert kb.sources() == ["os.md"]


def test_empty_index_returns_no_hits(kb):
    assert kb.search("anything") == []


# ------------------------------------------------------------------ rag
def test_out_of_scope_question_never_calls_llm(kb, notes, monkeypatch):
    kb.index_folder(notes)
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.5)
    llm = FakeLLM()
    app = StudyMate(kb, llm)
    r = app.retrieve("recipe for chocolate cake with frosting")
    assert not r.in_scope
    assert "".join(app.stream_answer("recipe", r)) == NOT_FOUND
    assert llm.calls == []


def test_in_scope_question_builds_cited_prompt(kb, notes, monkeypatch):
    kb.index_folder(notes)
    monkeypatch.setattr(config, "MAX_DISTANCE", 0.95)
    llm = FakeLLM()
    app = StudyMate(kb, llm)
    q = "What are the four Coffman conditions?"
    r = app.retrieve(q)
    assert r.in_scope
    answer = "".join(app.stream_answer(q, r))
    assert answer == "answer [1]"
    user_msg = llm.calls[-1][-1]["content"]
    assert user_msg.startswith("CONTEXT:\n[1] (")
    assert q in user_msg
    assert llm.calls[-1][0]["role"] == "system"


def test_followup_is_condensed_using_history(kb, notes, monkeypatch):
    kb.index_folder(notes)
    llm = FakeLLM()
    app = StudyMate(kb, llm)
    r = app.retrieve("and how do I prevent it?", history=[("What is deadlock?", "A wait cycle.")])
    assert r.query == "rewritten standalone question"


def test_format_context_numbers_passages(kb, notes):
    kb.index_folder(notes)
    text = format_context(kb.search("paging TLB", top_k=2))
    assert "[1] (" in text and "[2] (" in text
