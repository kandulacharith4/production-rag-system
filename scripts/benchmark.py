"""Retrieval ablation benchmark.

Runs four retrieval configurations over the golden set and reports recall@k,
MRR, and median latency. This is the headline number for the portfolio writeup:
it demonstrates that hybrid + rerank actually improves precision over naive
vector search, with real numbers instead of vibes.

Configurations:
  - vector_only   — dense embedding search only
  - bm25_only     — lexical BM25 only
  - hybrid        — min-max fused BM25 + vector
  - hybrid_rerank — hybrid, then cross-encoder rerank (the shipped default)

Relevance labeling:
A retrieved chunk is considered "relevant" if it contains ALL `must_mention`
keywords for the question (case-insensitive substring). This is a cheap,
deterministic proxy — no LLM judge needed, so results are reproducible by
anyone cloning the repo.

Outputs:
  - stdout: pretty table
  - benchmark_results.md: markdown table ready to paste into the README or LinkedIn
  - benchmark_results.json: raw numbers
"""
from __future__ import annotations

import copy
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.retrieval import retrieve  # noqa: E402

K_VALUES = [1, 3, 5]


def load_golden() -> list[dict]:
    path = ROOT / "eval" / "golden.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def is_relevant(chunk_text: str, case: dict) -> bool:
    keys = [k.lower() for k in case.get("must_mention", [])]
    if not keys:
        return False
    t = chunk_text.lower()
    return all(k in t for k in keys)


def score_config(name: str, cfg: dict, cases: list[dict]) -> dict:
    recalls = {k: [] for k in K_VALUES}
    rrs = []  # reciprocal ranks for MRR
    latencies = []

    for case in cases:
        t0 = time.perf_counter()
        results = retrieve(case["question"], cfg)
        latencies.append((time.perf_counter() - t0) * 1000)

        relevant_ranks = [i + 1 for i, r in enumerate(results) if is_relevant(r.text, case)]
        for k in K_VALUES:
            hit = any(rank <= k for rank in relevant_ranks)
            recalls[k].append(1.0 if hit else 0.0)
        rrs.append(1.0 / relevant_ranks[0] if relevant_ranks else 0.0)

    return {
        "name": name,
        "recall": {k: sum(v) / len(v) for k, v in recalls.items()},
        "mrr": sum(rrs) / len(rrs),
        "latency_ms_p50": statistics.median(latencies),
        "latency_ms_mean": statistics.mean(latencies),
    }


def build_variants(base: dict) -> list[tuple[str, dict]]:
    variants = []

    vec = copy.deepcopy(base)
    vec["retrieval"]["hybrid_alpha"] = 1.0
    vec["reranker"]["enabled"] = False
    variants.append(("vector_only", vec))

    bm = copy.deepcopy(base)
    bm["retrieval"]["hybrid_alpha"] = 0.0
    bm["reranker"]["enabled"] = False
    variants.append(("bm25_only", bm))

    hyb = copy.deepcopy(base)
    hyb["reranker"]["enabled"] = False
    variants.append(("hybrid", hyb))

    full = copy.deepcopy(base)
    full["reranker"]["enabled"] = True
    variants.append(("hybrid_rerank", full))

    return variants


def render_table(results: list[dict]) -> str:
    lines = []
    header = "| Config | Recall@1 | Recall@3 | Recall@5 | MRR | p50 latency (ms) |"
    sep =    "|---|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        lines.append(
            f"| `{r['name']}` "
            f"| {r['recall'][1]:.3f} "
            f"| {r['recall'][3]:.3f} "
            f"| {r['recall'][5]:.3f} "
            f"| {r['mrr']:.3f} "
            f"| {r['latency_ms_p50']:.0f} |"
        )
    return "\n".join(lines)


def main() -> int:
    base = load_config()
    cases = load_golden()
    if not cases:
        print("No golden cases found at eval/golden.jsonl", file=sys.stderr)
        return 1

    print(f"Running 4 retrieval configs over {len(cases)} golden questions...\n")
    results = []
    for name, cfg in build_variants(base):
        print(f"  -> {name}")
        results.append(score_config(name, cfg, cases))

    table = render_table(results)
    print("\n" + table + "\n")

    (ROOT / "benchmark_results.md").write_text(
        "# Retrieval Ablation Results\n\n"
        f"Corpus: `data/docs/` — {len(list((ROOT / 'data' / 'docs').rglob('*')))} files\n"
        f"Golden set: {len(cases)} questions\n\n"
        + table + "\n",
        encoding="utf-8",
    )
    (ROOT / "benchmark_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote benchmark_results.md and benchmark_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
