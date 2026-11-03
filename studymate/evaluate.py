"""Measure retrieval quality without any LLM (fast, free, fully offline).

Metrics
  Hit@k         : fraction of in-scope questions whose correct file appears in the top-k.
  MRR           : mean reciprocal rank of the first correct file (1.0 = always first).
  Refusal rate  : fraction of out-of-scope questions correctly rejected by the distance guard.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config
from .store import KnowledgeBase

DEFAULT_QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.json"


def evaluate(kb: KnowledgeBase, questions_path: Path | str = DEFAULT_QUESTIONS,
             top_k: int | None = None, verbose: bool = True) -> dict:
    top_k = top_k or config.TOP_K
    items = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    in_scope = [i for i in items if i["source"]]
    out_scope = [i for i in items if not i["source"]]

    hits_at_k = 0
    reciprocal = 0.0
    for item in in_scope:
        hits = kb.search(item["q"], top_k=top_k)
        sources = [h.source for h in hits]
        if item["source"] in sources:
            hits_at_k += 1
            reciprocal += 1.0 / (sources.index(item["source"]) + 1)
        elif verbose:
            print(f"  MISS: {item['q']!r} expected {item['source']} got {sources}")

    refused = 0
    for item in out_scope:
        hits = kb.search(item["q"], top_k=top_k)
        dists = [h.distance for h in hits if h.distance is not None]
        best = min(dists) if dists else None
        is_refused = best is None or best > config.MAX_DISTANCE
        refused += is_refused
        if verbose:
            shown = "n/a" if best is None else f"{best:.3f}"
            print(f"  out-of-scope {item['q']!r}: best distance {shown} -> "
                  f"{'refused' if is_refused else 'WOULD ANSWER (threshold too loose)'}")

    result = {
        "questions_in_scope": len(in_scope),
        f"hit@{top_k}": hits_at_k / len(in_scope) if in_scope else 0.0,
        "mrr": reciprocal / len(in_scope) if in_scope else 0.0,
        "questions_out_of_scope": len(out_scope),
        "refusal_rate": refused / len(out_scope) if out_scope else 0.0,
        "max_distance": config.MAX_DISTANCE,
    }
    return result
