"""LLM-assisted golden set drafter.

Samples chunks from the ingested corpus, asks Claude (via CLI) to produce a
crisp question + short reference answer + a few must-mention keywords grounded
in each chunk, and writes them to `eval/candidates.jsonl` for human curation.

IMPORTANT: Do not promote candidates directly into `eval/golden.jsonl`. Review
each one — LLM-generated questions drift into the generic ("What is X?") and
the golden set is only useful if a human has verified the answer key. Treat
this as a starting point that saves typing, not a substitute for curation.

Usage:
  python scripts/download_corpus.py
  python cli.py ingest data/docs
  python scripts/generate_goldenset.py --per-doc 3 --max-chunks 80
  # then manually review eval/candidates.jsonl and copy good ones to golden.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config  # noqa: E402
from app.llm import complete  # noqa: E402

SYSTEM = (
    "You draft high-quality evaluation questions for a RAG system. "
    "Given a source chunk, produce ONE question that is specifically answerable "
    "from that chunk (not generic, not answerable from prior knowledge alone), "
    "a concise reference answer grounded in the chunk, and 2-4 lowercase "
    "keywords that any correct answer must mention. Respond ONLY with a JSON "
    "object matching this schema, no prose: "
    '{"question": str, "reference": str, "must_mention": [str, ...]}'
)

USER_TMPL = """SOURCE: {source}

CHUNK:
{chunk}

Respond with the JSON object only."""

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def parse_llm_json(text: str) -> dict | None:
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in ("question", "reference", "must_mention")):
        return None
    return obj


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--per-doc", type=int, default=2, help="questions per source document")
    p.add_argument("--max-chunks", type=int, default=60, help="hard cap on chunks processed (cost control)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_config()
    corpus_path = Path(cfg["storage"]["chroma_dir"]) / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"Corpus not found at {corpus_path}. Run ingest first.", file=sys.stderr)
        return 1

    rows = [json.loads(l) for l in corpus_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    for src, chunks in by_source.items():
        rng.shuffle(chunks)
        picked.extend(chunks[: args.per_doc])
    rng.shuffle(picked)
    picked = picked[: args.max_chunks]

    out_path = ROOT / "eval" / "candidates.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_skip = 0, 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(picked, 1):
            try:
                raw = complete(
                    system=SYSTEM,
                    user=USER_TMPL.format(source=chunk["source"], chunk=chunk["text"]),
                    model=cfg["generation"].get("model"),
                )
            except Exception as e:
                print(f"[{i}/{len(picked)}] LLM error: {e}", file=sys.stderr)
                n_skip += 1
                continue

            obj = parse_llm_json(raw)
            if not obj:
                n_skip += 1
                continue
            obj["_source"] = chunk["source"]
            obj["_ordinal"] = chunk["ordinal"]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_ok += 1
            print(f"[{i}/{len(picked)}] {obj['question'][:80]}")

    print(f"\nWrote {n_ok} candidates to {out_path} ({n_skip} skipped).")
    print("Review them, then copy the good ones into eval/golden.jsonl (drop _source/_ordinal fields).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
