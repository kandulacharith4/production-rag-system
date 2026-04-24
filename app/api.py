"""FastAPI server exposing /ask, /health, and /metrics."""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from pydantic import BaseModel

from .generate import answer_question

logger = logging.getLogger("rag.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

app = FastAPI(title="RAG Application", version="1.0.0")


# --- lightweight in-memory metrics ---------------------------------------------------
# Enough to show "we think about observability" without dragging in Prometheus.
_LATENCY_WINDOW: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=500))
_COUNTERS: dict[str, int] = defaultdict(int)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    route = request.url.path
    _LATENCY_WINDOW[route].append(elapsed_ms)
    _COUNTERS[f"{route}:{response.status_code}"] += 1
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    logger.info("%s %s -> %d in %.1fms", request.method, route, response.status_code, elapsed_ms)
    return response


# --- schema ------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    index: int
    source: str
    ordinal: int


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    latency_ms: float


# --- routes ------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    out = {"counters": dict(_COUNTERS), "latency_ms": {}}
    for route, samples in _LATENCY_WINDOW.items():
        if not samples:
            continue
        arr = sorted(samples)
        n = len(arr)
        out["latency_ms"][route] = {
            "count": n,
            "p50": arr[n // 2],
            "p95": arr[min(n - 1, int(n * 0.95))],
            "max": arr[-1],
        }
    return out


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    ans = answer_question(req.question)
    elapsed = (time.perf_counter() - t0) * 1000
    cites = [
        Citation(index=i, source=ans.chunks[i - 1].source, ordinal=ans.chunks[i - 1].ordinal)
        for i in ans.citations
        if 1 <= i <= len(ans.chunks)
    ]
    return AskResponse(answer=ans.text, refused=ans.refused, citations=cites, latency_ms=elapsed)
