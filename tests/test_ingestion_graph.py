# test_ingestion_graph.py
# Tests for ingestion/pipeline.py - specifically ingest_chunks_into_graph()

from unittest.mock import MagicMock, patch
from ingestion.pipeline import build_chunks, ingest_chunks_into_graph
import tempfile
import os


# ---------------------------------------------------------------------------
# build_chunks (unit, no DB needed)
# ---------------------------------------------------------------------------

def test_build_chunks_returns_list():
    text = "Hello world. This is a test sentence. Another sentence here."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as f:
        f.write(text)
        tmp_path = f.name
    chunks = build_chunks(tmp_path)
    os.remove(tmp_path)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# ingest_chunks_into_graph (unit, mocked graph store)
# ---------------------------------------------------------------------------

def _make_graph_store():
    gs = MagicMock()
    gs.upsert_entity = MagicMock()
    gs.upsert_relationship = MagicMock()
    return gs


def _make_entity_extractor(entities):
    e = MagicMock()
    e.extract_entities = MagicMock(return_value=entities)
    return e


def _make_relation_extractor(triples):
    r = MagicMock()
    r.extract_relations = MagicMock(return_value=triples)
    return r


def test_ingest_chunks_into_graph_writes_entities():
    chunks = ["John Doe works at Acme Corp."]
    chunk_ids = [42]
    gs = _make_graph_store()
    ee = _make_entity_extractor({"PERSON": ["John Doe"], "ORG": ["Acme Corp"]})
    re = _make_relation_extractor([])

    summary = ingest_chunks_into_graph(chunks, chunk_ids, gs, ee, re)

    assert summary["entities_written"] == 2
    assert summary["relations_written"] == 0
    assert summary["errors"] == []
    gs.upsert_entity.assert_any_call("John Doe", "PERSON", chunk_id=42)
    gs.upsert_entity.assert_any_call("Acme Corp", "ORG", chunk_id=42)


def test_ingest_chunks_into_graph_writes_relations():
    chunks = ["John Doe works at Acme Corp."]
    chunk_ids = [7]
    gs = _make_graph_store()
    ee = _make_entity_extractor({})
    re = _make_relation_extractor([
        {"subject": "John Doe", "predicate": "works at", "object": "Acme Corp"}
    ])

    summary = ingest_chunks_into_graph(chunks, chunk_ids, gs, ee, re)

    assert summary["relations_written"] == 1
    gs.upsert_relationship.assert_called_once_with(
        "John Doe", "works at", "Acme Corp", chunk_id=7
    )


def test_ingest_chunks_into_graph_skips_incomplete_triples():
    chunks = ["Some chunk."]
    chunk_ids = [1]
    gs = _make_graph_store()
    ee = _make_entity_extractor({})
    re = _make_relation_extractor([
        {"subject": "A", "predicate": None, "object": "B"},  # missing predicate
        {"subject": "", "predicate": "rel", "object": "B"},  # empty subject
    ])

    summary = ingest_chunks_into_graph(chunks, chunk_ids, gs, ee, re)
    assert summary["relations_written"] == 0


def test_ingest_chunks_into_graph_misaligned_inputs():
    gs = _make_graph_store()
    summary = ingest_chunks_into_graph(["a", "b"], [1], gs)
    assert summary["entities_written"] == 0
    assert len(summary["errors"]) >= 1


def test_ingest_chunks_into_graph_entity_error_is_nonfatal():
    chunks = ["text chunk"]
    chunk_ids = [99]
    gs = _make_graph_store()
    ee = _make_entity_extractor({})
    ee.extract_entities.side_effect = Exception("spacy failed")
    re = _make_relation_extractor([])

    summary = ingest_chunks_into_graph(chunks, chunk_ids, gs, ee, re)
    assert len(summary["errors"]) == 1
    assert "spacy failed" in summary["errors"][0]


def test_ingest_chunks_into_graph_relation_error_is_nonfatal():
    chunks = ["text chunk"]
    chunk_ids = [99]
    gs = _make_graph_store()
    ee = _make_entity_extractor({})
    re = _make_relation_extractor([])
    re.extract_relations.side_effect = Exception("LLM timeout")

    summary = ingest_chunks_into_graph(chunks, chunk_ids, gs, ee, re)
    assert len(summary["errors"]) == 1
    assert "LLM timeout" in summary["errors"][0]
