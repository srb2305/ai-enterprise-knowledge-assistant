# test_phase1.py
# Tests for Phase 1: Ingestion, chunking, embedding, and vector retrieval

import os
import tempfile
import pytest
from ingestion.pipeline import ingest_document
from retrieval.vector_retriever import VectorStore
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
PGVECTOR_URL = os.getenv("PGVECTOR_URL")

@pytest.fixture(scope="module")
def vector_store():
    return VectorStore(PGVECTOR_URL)

def test_ingest_and_chunk():
    text = "John Doe works at Acme Corp. Jane Smith joined in 2020."
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    chunks = ingest_document(tmp_path)
    os.remove(tmp_path)
    assert isinstance(chunks, list)
    assert any("John Doe" in c or "Jane Smith" in c for c in chunks)

def test_vector_upsert_and_query(vector_store):
    chunks = ["John Doe works at Acme Corp.", "Jane Smith joined in 2020."]
    ids = vector_store.upsert_chunks(chunks)
    assert isinstance(ids, list)
    assert len(ids) == 2
    results = vector_store.query("Who works at Acme Corp?", top_k=2)
    assert isinstance(results, list)
    # results are now dicts with chunk_id, chunk, score, metadata
    for r in results:
        assert "chunk" in r
        assert "chunk_id" in r
        assert "score" in r
    assert any("Acme Corp" in r["chunk"] for r in results)
