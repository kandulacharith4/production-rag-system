"""Answer generation with citation enforcement."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import load_config, load_prompts
from .llm import complete
from .retrieval import Retrieved

REFUSAL = "I don't have enough information in the provided sources to answer that."
_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    text: str
    citations: list[int] = field(default_factory=list)
    chunks: list[Retrieved] = field(default_factory=list)
    refused: bool = False


def _format_context(chunks: list[Retrieved]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {c.source}, chunk {c.ordinal})\n{c.text}")
    return "\n\n".join(parts)


def _enforce_citations(text: str, n_chunks: int) -> tuple[str, list[int], bool]:
    """Strip invalid citation numbers. Return (cleaned_text, cited_indices, refused)."""
    if text.strip() == REFUSAL:
        return text.strip(), [], True

    cited = [int(m) for m in _CITATION_RE.findall(text)]
    valid = [c for c in cited if 1 <= c <= n_chunks]
    if not valid:
        return REFUSAL, [], True

    def _scrub(m: re.Match) -> str:
        n = int(m.group(1))
        return m.group(0) if 1 <= n <= n_chunks else ""

    cleaned = _CITATION_RE.sub(_scrub, text).strip()
    return cleaned, sorted(set(valid)), False


def answer_question(question: str, config: dict | None = None) -> Answer:
    from .retrieval import retrieve

    cfg = config or load_config()
    prompts = load_prompts()
    chunks = retrieve(question, cfg)

    if not chunks:
        return Answer(text=REFUSAL, refused=True)

    context = _format_context(chunks)
    raw = complete(
        system=prompts["answer_system"],
        user=prompts["answer_user"].format(context=context, question=question),
        model=cfg["generation"].get("model"),
    )
    cleaned, cites, refused = _enforce_citations(raw, len(chunks))
    return Answer(text=cleaned, citations=cites, chunks=chunks, refused=refused)
