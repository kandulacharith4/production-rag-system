from app.ingest import chunk_text


def test_chunking_respects_overlap():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, size_tokens=200, overlap_tokens=50)
    assert len(chunks) > 1
    # Consecutive chunks should share some prefix/suffix content due to overlap.
    for a, b in zip(chunks, chunks[1:]):
        tail = a.split()[-20:]
        head = b.split()[:40]
        assert any(t in head for t in tail), "expected token overlap between consecutive chunks"


def test_chunking_empty_string():
    assert chunk_text("", 100, 20) == []


def test_chunking_single_window():
    text = "hello world " * 10
    chunks = chunk_text(text, size_tokens=500, overlap_tokens=50)
    assert len(chunks) == 1
