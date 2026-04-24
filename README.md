# Production-Grade RAG Application

A domain-agnostic "Ask My Docs" system with hybrid retrieval, cross-encoder
reranking, citation enforcement, and a CI-gated evaluation pipeline. Built to
demonstrate the scaffolding that separates a RAG demo from a RAG system you
can actually operate.

> **The thesis:** the interesting engineering in RAG isn't the retrieval — it's
> the evaluation loop, the citation guardrails, and the CI gate that tell you
> when retrieval is getting worse.

---

## Quick demo

```bash
make install                              # pip deps
make corpus                               # download 30 Python PEPs
make ingest                               # embed + BM25 index
make ask Q="What does PEP 572 introduce?" # cited answer
make benchmark                            # ablation table
make eval                                 # CI-gate quality check
make test                                 # unit tests
make serve                                # FastAPI on :8000 -> /docs for Swagger
```

Or, Docker:

```bash
docker compose up --build
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What does PEP 572 introduce?"}'
```

---

## Architecture

```
        ┌──────── BM25 (lexical) ───────┐
query ──┤                               ├─► min-max fuse ─► cross-encoder rerank ─► top-k chunks
        └──── vector (semantic, Chroma)─┘                                                │
                                                                                         ▼
                                                                           LLM (Claude via CLI)
                                                                                         │
                                                                   citation enforcer ────┤
                                                                                         ▼
                                              cited answer  |or|  "I don't have enough information…"
```

- **Chunking** — 600 tokens / 100 overlap ([app/ingest.py](app/ingest.py))
- **Fusion** — min-max normalized BM25 + cosine, weighted by `hybrid_alpha` ([app/retrieval.py](app/retrieval.py))
- **Reranker** — `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top-N shortlist
- **Citation enforcement** — every claim must be followed by `[n]`; hallucinated
  indices are stripped; if the model cites nothing valid, the pipeline returns
  a fixed refusal phrase instead of guessing ([app/generate.py](app/generate.py))
- **Prompts** — versioned YAML at [prompts/prompts.yaml](prompts/prompts.yaml)
- **Generation** — Claude via the `claude` CLI ([app/llm.py](app/llm.py)) — no SDK dependency

---

## Retrieval ablation (`make benchmark`)

The headline number: four retrieval configurations scored over the golden set.
Plug your own numbers in after running `make benchmark` — the script writes
[benchmark_results.md](benchmark_results.md).

| Config | Recall@1 | Recall@3 | Recall@5 | MRR | p50 latency (ms) |
|---|---:|---:|---:|---:|---:|
| `vector_only` | *run the benchmark* | | | | |
| `bm25_only` | | | | | |
| `hybrid` | | | | | |
| `hybrid_rerank` | | | | | |

Relevance is labeled deterministically (a chunk is "relevant" if it contains
all `must_mention` keywords for the question), so these numbers are
reproducible by anyone who clones the repo.

---

## Quality gate (`make eval`)

A curated golden set in [eval/golden.jsonl](eval/golden.jsonl) drives an
LLM-as-judge faithfulness score (supported claims ÷ total claims). The
evaluator exits non-zero if mean faithfulness falls below the threshold set in
[config.yaml](config.yaml) (`eval.faithfulness_threshold`, default 0.85). This
runs on every PR via [.github/workflows/eval.yml](.github/workflows/eval.yml).

A correct refusal counts as fully faithful — the goal is "don't say false
things," not "always say something."

To grow the golden set: `make goldenset` runs an LLM-assisted drafter over
your corpus and writes candidates to `eval/candidates.jsonl`. Review, then
hand-curate into `golden.jsonl`. Never promote candidates unreviewed.

---

## Project layout

```
app/
  ingest.py       # chunk + embed + write Chroma + BM25 corpus
  retrieval.py    # hybrid fusion + cross-encoder rerank
  generate.py     # answer + citation enforcement
  llm.py          # Claude CLI subprocess wrapper
  api.py          # FastAPI (/ask, /health, /metrics)
  config.py       # config + prompt loader
cli.py            # `ingest` / `ask` commands
prompts/
  prompts.yaml    # versioned prompts
eval/
  golden.jsonl    # curated Q/A pairs
  evaluate.py     # faithfulness scoring + CI gate
scripts/
  download_corpus.py    # public PEP corpus
  generate_goldenset.py # LLM-drafted eval candidates
  benchmark.py          # retrieval ablation
tests/                  # chunking + citation unit tests
.github/workflows/
  eval.yml              # CI runs ingest -> tests -> eval
Dockerfile, docker-compose.yml, Makefile
```

---

## Observability

The API includes a lightweight timing middleware and an in-memory `/metrics`
endpoint exposing p50/p95/max per route — enough to see hot paths in a demo.
For real production I'd wire this to Prometheus, which is a 10-line swap on
the same middleware.

Every response also carries an `X-Response-Time-ms` header.

---

## How I'd harden this further

Explicit about what this *isn't*, because hiring managers ask:

- **Streaming responses** — the CLI wrapper buffers to stdout. For the API,
  switch to Anthropic's SSE streaming and forward as a FastAPI `StreamingResponse`.
- **Per-user isolation** — single-tenant today. Add a `tenant_id` to the
  Chroma metadata filter and a rate limiter in front of `/ask`.
- **Incremental ingestion** — `ingest` recreates the collection. Hash files,
  diff against a manifest, only upsert changed chunks.
- **Better reranker** — cross-encoder MiniLM is fine for the demo; Cohere
  Rerank 3 or a fine-tuned BGE-reranker would raise the ceiling.
- **Query rewriting** — for conversational use, add a step that rewrites
  follow-ups into standalone queries before retrieval.
- **Real observability stack** — Prometheus + OTel traces, not in-memory dicts.

---

## Credits

Built in one focused pass using [Claude Code](https://docs.claude.com/claude-code)
as both the coding assistant and the answer-generation backend. The same CLI
drives local development and the production `/ask` endpoint.
