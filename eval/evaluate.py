"""Offline evaluation: runs the RAG pipeline over a golden set and scores faithfulness.

Uses the `claude` CLI (via app.llm.complete) for both generation and LLM-as-judge
scoring, so no API key wiring is required beyond what Claude Code already uses.

Exits non-zero if aggregate faithfulness is below config.eval.faithfulness_threshold,
so this drops directly into CI.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_config, load_prompts  # noqa: E402
from app.generate import answer_question, REFUSAL  # noqa: E402
from app.llm import complete  # noqa: E402


_SCORE_RE = re.compile(r"SCORE:\s*(\d+)\s*/\s*(\d+)")


def judge_faithfulness(prompts: dict, model: str | None, answer: str, context: str) -> float:
    if answer.strip() == REFUSAL:
        # A correct refusal makes no unsupported claims, so it is trivially faithful.
        return 1.0
    text = complete(
        system=prompts["faithfulness_system"],
        user=prompts["faithfulness_user"].format(context=context, answer=answer),
        model=model,
    )
    m = _SCORE_RE.search(text)
    if not m:
        return 0.0
    sup, tot = int(m.group(1)), int(m.group(2))
    return sup / tot if tot else 0.0


def main() -> int:
    cfg = load_config()
    prompts = load_prompts()
    golden_path = ROOT / "eval" / "golden.jsonl"
    cases = [json.loads(l) for l in golden_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    judge_model = cfg["generation"].get("model")

    results = []
    for case in cases:
        q = case["question"]
        ans = answer_question(q, cfg)
        ctx = "\n\n".join(f"[{i+1}] {c.text}" for i, c in enumerate(ans.chunks))
        faith = judge_faithfulness(prompts, judge_model, ans.text, ctx)
        keys = [k.lower() for k in case.get("must_mention", [])]
        lower = ans.text.lower()
        hits = sum(1 for k in keys if k in lower)
        recall = hits / len(keys) if keys else 1.0
        results.append(
            {
                "question": q,
                "answer": ans.text,
                "refused": ans.refused,
                "citations": ans.citations,
                "faithfulness": faith,
                "keyword_recall": recall,
            }
        )
        print(f"[{faith:.2f} | recall {recall:.2f}] {q}")

    agg_faith = sum(r["faithfulness"] for r in results) / len(results)
    agg_recall = sum(r["keyword_recall"] for r in results) / len(results)

    report = {
        "n": len(results),
        "faithfulness_mean": agg_faith,
        "keyword_recall_mean": agg_recall,
        "threshold": cfg["eval"]["faithfulness_threshold"],
        "cases": results,
    }
    out_path = ROOT / "eval" / "report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFaithfulness mean: {agg_faith:.3f}  (threshold {cfg['eval']['faithfulness_threshold']})")
    print(f"Keyword recall mean: {agg_recall:.3f}")
    print(f"Wrote {out_path}")

    if agg_faith < cfg["eval"]["faithfulness_threshold"]:
        print("FAIL: faithfulness below threshold", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
