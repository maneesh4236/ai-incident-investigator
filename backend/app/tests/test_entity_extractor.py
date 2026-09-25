from unittest.mock import MagicMock

from app.models.schemas import Chunk
from app.services.ingestion.entity_extractor import EntityExtractor


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        id="chunk-1",
        document_id="doc-1",
        investigation_id="inv-1",
        text=text,
        chunk_index=0,
    )


def test_heuristic_extraction_finds_known_components_and_errors():
    llm = MagicMock()
    llm.is_configured = False
    extractor = EntityExtractor(llm_client=llm)

    chunk = _make_chunk("Redis connections started timing out, causing a payment failure.")
    result = extractor.extract(chunk)

    entity_names = {e.name for e in result.entities}
    assert "Redis" in entity_names
    assert "Timeout" in entity_names or "Failure" in entity_names


def test_heuristic_extraction_detects_causal_language():
    llm = MagicMock()
    llm.is_configured = False
    extractor = EntityExtractor(llm_client=llm)

    chunk = _make_chunk("Connection pool exhaustion caused payment retry storm across the checkout service.")
    result = extractor.extract(chunk)

    assert any(r.type.value == "CAUSES" for r in result.relationships)


def test_llm_extraction_used_when_configured():
    llm = MagicMock()
    llm.is_configured = True
    llm.generate_json.return_value = {
        "entities": [{"name": "Redis", "type": "DATABASE"}],
        "relationships": [{"source": "Redis", "target": "Timeout", "type": "CAUSES", "confidence": 0.9}],
    }
    extractor = EntityExtractor(llm_client=llm)

    chunk = _make_chunk("Redis timeout observed at 12:01.")
    result = extractor.extract(chunk)

    assert result.entities[0].name == "Redis"
    assert result.relationships[0].type.value == "CAUSES"
