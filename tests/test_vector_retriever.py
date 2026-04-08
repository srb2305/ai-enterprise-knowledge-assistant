# test_vector_retriever.py
# Tests for VectorStore (Phase 3)
import os
import pytest
from retrieval.vector_retriever import VectorStore

PGVECTOR_URL = os.getenv("PGVECTOR_URL")

@pytest.fixture(scope="module")
def vector_store():
    return VectorStore(PGVECTOR_URL)

def test_vector_query(vector_store):
    query = "Who works at Acme Corp?"
    results = vector_store.query(query, top_k=3)
    assert isinstance(results, list)
    for chunk, score, metadata in results:
        assert isinstance(chunk, str)
        assert isinstance(score, float) or isinstance(score, int)
