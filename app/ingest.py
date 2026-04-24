"""Document ingestion: load files, chunk with overlap, embed, store in Chroma."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import tiktoken
from pypdf import PdfReader

from .config import load_config

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    ordinal: int


def _read_file(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if suf in {".md", ".txt", ".rst"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    # fallback: treat as utf-8 text
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size_tokens: int, overlap_tokens: int) -> list[str]:
    """Token-window chunking with overlap. Slices on token boundaries to avoid
    mid-sentence loss when combined with overlap."""
    if not text.strip():
        return []
    toks = _enc.encode(text)
    step = max(1, size_tokens - overlap_tokens)
    chunks = []
    for start in range(0, len(toks), step):
        window = toks[start : start + size_tokens]
        if not window:
            break
        chunks.append(_enc.decode(window))
        if start + size_tokens >= len(toks):
            break
    return chunks


def _chunk_id(source: str, ordinal: int, text: str) -> str:
    h = hashlib.sha1(f"{source}:{ordinal}:{text[:64]}".encode("utf-8")).hexdigest()[:16]
    return f"{Path(source).stem}-{ordinal}-{h}"


def iter_documents(root: Path) -> Iterable[Path]:
    exts = {".pdf", ".md", ".txt", ".rst"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def build_chunks(doc_root: Path, size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    out: list[Chunk] = []
    for path in iter_documents(doc_root):
        text = _read_file(path)
        pieces = chunk_text(text, size_tokens, overlap_tokens)
        for i, piece in enumerate(pieces):
            src = str(path.relative_to(doc_root.parent)) if doc_root.parent in path.parents else str(path)
            out.append(Chunk(id=_chunk_id(src, i, piece), text=piece, source=src, ordinal=i))
    return out


def ingest(doc_root: str | Path, config: dict | None = None) -> dict:
    """Ingest all documents under doc_root into Chroma + save BM25 corpus to disk."""
    import chromadb
    from chromadb.utils import embedding_functions

    cfg = config or load_config()
    doc_root = Path(doc_root)
    chunks = build_chunks(
        doc_root,
        size_tokens=cfg["chunk"]["size_tokens"],
        overlap_tokens=cfg["chunk"]["overlap_tokens"],
    )
    if not chunks:
        raise RuntimeError(f"No documents found under {doc_root}")

    chroma_dir = Path(cfg["storage"]["chroma_dir"])
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg["embedding"]["model"]
    )
    # recreate to stay in sync with BM25 corpus
    try:
        client.delete_collection(cfg["storage"]["collection"])
    except Exception:
        pass
    coll = client.create_collection(
        cfg["storage"]["collection"], embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )

    # Prepend source path to each document so both BM25 and the vector embedder
    # can associate "pep 572" or "pep-0572" queries with the right file.
    def _indexed_text(c: Chunk) -> str:
        return f"[Source: {c.source}]\n{c.text}"

    BATCH = 128
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        coll.add(
            ids=[c.id for c in batch],
            documents=[_indexed_text(c) for c in batch],
            metadatas=[{"source": c.source, "ordinal": c.ordinal} for c in batch],
        )

    # Persist a flat corpus file for BM25 (built at query time to avoid pickling models).
    corpus_path = chroma_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            row = asdict(c)
            row["text"] = _indexed_text(c)  # BM25 sees the same prefixed text
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"chunks": len(chunks), "sources": len({c.source for c in chunks})}


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/docs"
    stats = ingest(target)
    print(json.dumps(stats, indent=2))
