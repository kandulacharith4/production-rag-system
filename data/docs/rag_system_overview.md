# RAG Application Overview

This project is a production-grade Retrieval-Augmented Generation (RAG) system for
answering questions over a corpus of documents with verifiable citations.

## Chunking

Documents are split into overlapping token windows. The default chunk size is **600
tokens** with a **100 token** overlap between adjacent chunks. The overlap exists so
that a sentence sliced at a chunk boundary is still fully present in one of the
neighboring chunks, which keeps retrieval context-complete.

## Embeddings

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. It is small,
fast, and a strong baseline for dense semantic retrieval.

## Vector store

Embeddings are stored in **ChromaDB** using a persistent client on local disk. The
collection uses cosine distance.

## Hybrid retrieval

At query time the system runs two retrievers in parallel:

1. **BM25** — classical keyword / lexical search over tokenized chunks. Good at
   catching exact terms, IDs, and rare phrases.
2. **Vector search** — dense semantic search over the Chroma collection. Good at
   matching meaning even when wording differs.

Their scores are min-max normalized and fused with a weighted sum controlled by
`retrieval.hybrid_alpha` (default 0.5).

## Reranking

The fused top candidates are rescored with the cross-encoder
`cross-encoder/ms-marco-MiniLM-L-6-v2`. A cross-encoder evaluates the query and each
candidate chunk together as a pair, which is significantly more precise than separate
embedding similarity.

## Citation enforcement

The generator prompt requires every factual claim to be followed by a citation of
the form `[n]` where `n` is the chunk number. If the model cites nothing valid, or
if it emits the fixed refusal phrase, the pipeline returns:

> "I don't have enough information in the provided sources to answer that."

This prevents ungrounded hallucinations from reaching the user.

## Prompt versioning

All prompts live in a versioned YAML file at `prompts/prompts.yaml`. Treating prompts
as configuration rather than hardcoded strings lets us diff, review, and roll back
prompt changes like any other code change.

## Evaluation

A curated golden set of question / reference pairs lives in `eval/golden.jsonl`.
The evaluation script `eval/evaluate.py` runs the full pipeline on every case and
scores **faithfulness** — whether each atomic claim in the answer is supported by
the retrieved chunks — using an LLM-as-judge with a deterministic scoring format.

CI fails if mean faithfulness drops below the threshold defined in
`config.yaml` under `eval.faithfulness_threshold` (default **0.85**).
