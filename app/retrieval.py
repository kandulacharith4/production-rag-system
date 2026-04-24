"""Hybrid retrieval: BM25 + vector search, fused and reranked with a cross-encoder."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from .config import load_config


@dataclass
class Retrieved:
    id: str
    text: str
    source: str
    ordinal: int
    score: float


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


@lru_cache(maxsize=1)
def _load_corpus(chroma_dir: str):
    path = Path(chroma_dir) / "corpus.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    tokenized = [_tokenize(r["text"]) for r in rows]
    bm25 = BM25Okapi(tokenized)
    return rows, bm25


@lru_cache(maxsize=1)
def _chroma_collection(chroma_dir: str, collection: str, embed_model: str):
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=chroma_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_model)
    return client.get_collection(collection, embedding_function=ef)


@lru_cache(maxsize=1)
def _reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _minmax(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def retrieve(query: str, config: dict | None = None) -> list[Retrieved]:
    cfg = config or load_config()
    chroma_dir = cfg["storage"]["chroma_dir"]

    # --- BM25 ---
    rows, bm25 = _load_corpus(chroma_dir)
    bm25_scores = bm25.get_scores(_tokenize(query))
    k_bm = cfg["retrieval"]["bm25_top_k"]
    bm25_top = sorted(range(len(rows)), key=lambda i: bm25_scores[i], reverse=True)[:k_bm]

    # --- Vector ---
    coll = _chroma_collection(
        chroma_dir, cfg["storage"]["collection"], cfg["embedding"]["model"]
    )
    k_vec = cfg["retrieval"]["vector_top_k"]
    vres = coll.query(query_texts=[query], n_results=k_vec)
    vec_ids = vres["ids"][0]
    vec_docs = vres["documents"][0]
    vec_meta = vres["metadatas"][0]
    # chroma cosine distances: lower = better. Convert to similarity.
    vec_sims = [1.0 - d for d in vres["distances"][0]]

    # --- Fusion: normalize then weighted sum over union of candidates ---
    alpha = cfg["retrieval"]["hybrid_alpha"]
    # Map chunk_id -> (text, source, ordinal)
    pool: dict[str, dict] = {}
    # BM25 candidates
    bm_id_score = {rows[i]["id"]: bm25_scores[i] for i in bm25_top}
    for i in bm25_top:
        r = rows[i]
        pool[r["id"]] = {
            "text": r["text"],
            "source": r["source"],
            "ordinal": r["ordinal"],
        }
    # Vector candidates
    vec_id_sim = {}
    for cid, doc, meta, sim in zip(vec_ids, vec_docs, vec_meta, vec_sims):
        vec_id_sim[cid] = sim
        pool.setdefault(cid, {"text": doc, "source": meta.get("source", ""), "ordinal": meta.get("ordinal", 0)})

    ids = list(pool)
    bm_vec = [bm_id_score.get(i, 0.0) for i in ids]
    vc_vec = [vec_id_sim.get(i, 0.0) for i in ids]
    bm_n = _minmax(bm_vec)
    vc_n = _minmax(vc_vec)
    fused = [alpha * v + (1 - alpha) * b for v, b in zip(vc_n, bm_n)]

    ranked_idx = sorted(range(len(ids)), key=lambda i: fused[i], reverse=True)
    # Take a healthy pre-rerank slate
    pre_k = max(cfg["retrieval"]["final_top_k"] * 4, cfg["retrieval"]["final_top_k"])
    shortlist = [(ids[i], fused[i]) for i in ranked_idx[:pre_k]]

    # --- Cross-encoder rerank ---
    final_k = cfg["retrieval"]["final_top_k"]
    if cfg["reranker"]["enabled"] and shortlist:
        ce = _reranker(cfg["reranker"]["model"])
        pairs = [(query, pool[cid]["text"]) for cid, _ in shortlist]
        ce_scores = ce.predict(pairs).tolist()
        reranked = sorted(
            zip(shortlist, ce_scores), key=lambda x: x[1], reverse=True
        )[:final_k]
        out = []
        for (cid, _), s in reranked:
            info = pool[cid]
            out.append(
                Retrieved(id=cid, text=info["text"], source=info["source"], ordinal=info["ordinal"], score=float(s))
            )
        return out

    return [
        Retrieved(
            id=cid,
            text=pool[cid]["text"],
            source=pool[cid]["source"],
            ordinal=pool[cid]["ordinal"],
            score=s,
        )
        for cid, s in shortlist[:final_k]
    ]
