# LinkedIn Launch Kit

Everything you need to turn this repo into a post that lands. Fill in `{{...}}`
with your own numbers after running the benchmark; the rest is ready to go.

---

## 1. Before you post: run the numbers

```bash
python scripts/download_corpus.py     # 30 Python PEPs
python cli.py ingest data/docs
python scripts/generate_goldenset.py --per-doc 2 --max-chunks 80
# curate eval/candidates.jsonl into a 50-100 question eval/golden.jsonl
python scripts/benchmark.py           # writes benchmark_results.md
python eval/evaluate.py               # writes eval/report.json
pytest -q
```

Copy the table out of `benchmark_results.md` — that's your hero number.

---

## 2. The post (copy-paste, then personalize)

> I shipped a production-grade RAG system this week. Not a notebook, not a
> demo — a system with a CI gate, an evaluation harness, and measurable
> retrieval numbers I had to defend against ablations.
>
> The gap between "RAG demo that works on a happy-path query" and "RAG system
> a team can actually operate" is where most implementations fall apart. Here's
> what closing that gap actually looked like:
>
> 🔹 **Hybrid retrieval** — BM25 for exact terms, dense embeddings for intent,
> min-max fused. Lexical and semantic catch different failure modes.
>
> 🔹 **Cross-encoder reranking** — re-scores shortlisted candidates pairwise.
> Bumped Recall@1 from {{vec_r1}} → {{hyb_rerank_r1}} on my golden set.
>
> 🔹 **Citation enforcement** — every factual claim gets a [n] marker tied to
> a real retrieved chunk. If the model can't ground the answer, the pipeline
> returns a fixed refusal phrase instead of hallucinating. Non-negotiable for
> anything that lands in front of a user.
>
> 🔹 **Evaluation as a CI gate** — curated golden set of {{n}} Q/A pairs,
> LLM-as-judge faithfulness scoring, pipeline fails PRs if mean faithfulness
> drops below 0.85. If you can't block a regression, you don't have a quality
> bar, you have a hope.
>
> 🔹 **Versioned prompts** — YAML, in the repo, reviewable in PRs.
>
> Ablation across {{n}} questions on a {{corpus_size}}-document corpus:
>
> ```
> {{paste benchmark table here}}
> ```
>
> Stack: Python, ChromaDB, sentence-transformers, rank-bm25, FastAPI, Claude
> (via the Claude Code CLI), pytest. Dockerized, Makefile-driven.
>
> Code + setup: {{github-url}}
>
> What I learned: the interesting engineering in RAG isn't the retrieval — it's
> the scaffolding that tells you when retrieval is getting worse.
>
> #AI #RAG #MachineLearning #LLM #Python

---

## 3. Post variants by angle

### Angle A — "demo vs production" (best for hiring-manager reach)
Lead with the chart. First line: *"Most RAG demos pass the happy path and
collapse on the tail. Here's the scaffolding that catches that."*

### Angle B — "what I learned" (best for peer reach)
Lead with the lesson. First line: *"I thought the hard part of RAG was the
retrieval. It wasn't. It was the evaluation loop."*

### Angle C — "ablation numbers" (best if your numbers are strong)
Lead with the table. First line: *"Recall@1 went from {{X}} to {{Y}} by
adding two things most RAG tutorials skip."*

---

## 4. Screen-recording script (60 seconds)

1. **Terminal, 0:00–0:10** — `make corpus && make ingest` (speed up 4x)
2. **Terminal, 0:10–0:25** — `python cli.py ask "What does PEP 572 introduce?"`
   → show cited answer, highlight `[1]` + source line.
3. **Terminal, 0:25–0:40** — `python cli.py ask "Who won the 2022 World Cup?"`
   → show the refusal. This is the money shot — demonstrates guardrails.
4. **Terminal, 0:40–0:55** — `make benchmark` → show the table.
5. **Terminal, 0:55–1:00** — `make test && make eval` → green, "build passes".

Record with OBS, export mp4, upload natively to LinkedIn (don't link to
YouTube — native video gets 5x the reach).

---

## 5. Pinned-comment hooks

After you post, drop a follow-up comment with one of:

- "Happy to walk anyone through the evaluation harness — the LLM-as-judge
  faithfulness loop is the part I spent the most time tuning."
- "The bit I almost cut but am glad I kept: the citation enforcer that strips
  hallucinated chunk indices and forces a refusal. Caught more bugs than my
  tests did."
- "Numbers are on a 30-doc PEP corpus. If you've got a harder corpus you
  want me to benchmark on, drop it below."

These three are designed to pull comments from three different audiences
(engineers, infra people, potential collaborators).

---

## 6. What to include in the repo README

Hiring managers who click the link spend ~20 seconds deciding whether to read
more. The README should put in front of them, in order:

1. One-sentence pitch.
2. The benchmark table.
3. A 5-line architecture diagram.
4. A "how I'd harden this further" section — shows you know this isn't finished.

All of that is already wired into `README.md` — just plug your numbers in after
running the benchmark.
