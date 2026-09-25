from app.services.ingestion.chunker import Chunker


def test_chunker_splits_long_text_into_multiple_chunks():
    chunker = Chunker(chunk_size=10, overlap=2)
    text = " ".join(f"word{i}" for i in range(35))

    chunks = chunker.split(text, document_id="doc-1", investigation_id="inv-1")

    assert len(chunks) > 1
    assert all(c.document_id == "doc-1" for c in chunks)
    assert all(c.investigation_id == "inv-1" for c in chunks)
    # chunk indices should be sequential starting at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunker_handles_empty_text():
    chunker = Chunker(chunk_size=10, overlap=2)
    assert chunker.split("", document_id="doc-1", investigation_id="inv-1") == []


def test_chunker_overlap_shares_words_between_consecutive_chunks():
    chunker = Chunker(chunk_size=10, overlap=4)
    text = " ".join(f"word{i}" for i in range(20))

    chunks = chunker.split(text, document_id="doc-1", investigation_id="inv-1")

    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words  # overlap should share at least one word
