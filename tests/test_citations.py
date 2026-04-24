from app.generate import _enforce_citations, REFUSAL


def test_valid_citations_pass_through():
    text, cites, refused = _enforce_citations("The sky is blue [1] and bright [2].", n_chunks=3)
    assert not refused
    assert cites == [1, 2]
    assert "[1]" in text and "[2]" in text


def test_no_citation_triggers_refusal():
    text, cites, refused = _enforce_citations("The answer is 42.", n_chunks=3)
    assert refused
    assert text == REFUSAL
    assert cites == []


def test_hallucinated_citation_stripped():
    text, cites, refused = _enforce_citations("Fact A [1]. Fact B [9].", n_chunks=3)
    assert not refused
    assert cites == [1]
    assert "[9]" not in text
    assert "[1]" in text


def test_explicit_refusal_phrase_passes_through():
    text, cites, refused = _enforce_citations(REFUSAL, n_chunks=3)
    assert refused
    assert cites == []
    assert text == REFUSAL
