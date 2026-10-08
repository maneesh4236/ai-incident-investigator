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


def test_document_chunks_keep_whole_lines_and_newlines():
    chunker = Chunker(chunk_size=12, overlap=4)
    lines = [f"line {i} has five words" for i in range(10)]
    chunks = chunker.split("\n".join(lines), document_id="d", investigation_id="i")
    assert len(chunks) > 1
    for chunk in chunks:
        for line in chunk.text.split("\n"):
            assert line in lines  # never a partial line


def test_points_from_groups_one_per_group_whole_events():
    from pathlib import Path

    from app.services.ingestion.event_processor import EventProcessor
    from app.services.ingestion.log_loader import LogEventParser

    path = Path(__file__).parent / "fixtures" / "bracketed_multiline.txt"
    events = LogEventParser().parse_file(str(path), "d")
    groups = EventProcessor().process(events)
    points = Chunker().points_from_groups(groups, {e.id: e for e in events}, "d", "i")
    assert len(points) == len(groups)
    original = set(path.read_text(encoding="utf-8").splitlines())
    for point in points:
        assert point.metadata["group_id"] and point.metadata["event_ids"]
        for line in point.text.split("\n"):
            assert line in original or line.startswith("(x") or line.endswith("omitted]")


def test_compact_event_text_keeps_header_exception_and_whole_lines():
    raw = "2030 ERROR svc boom\njava.lang.IllegalStateException: bad\n" + "\n".join(
        f"\tat com.example.C.m{i}(C.java:{i})" for i in range(50)
    ) + "\nCaused by: java.io.IOException: disk"
    text = Chunker.compact_event_text(raw, 400, max_frames=3)
    lines = text.split("\n")
    assert lines[0] == "2030 ERROR svc boom"
    assert "java.lang.IllegalStateException: bad" in lines
    assert "Caused by: java.io.IOException: disk" in lines
    assert sum(1 for l in lines if l.lstrip().startswith("at ")) <= 3
    assert lines[-1].endswith("more lines omitted]")
    assert all(l in raw.split("\n") for l in lines[:-1])
