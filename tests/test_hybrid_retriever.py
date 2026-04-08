# test_hybrid_retriever.py
# Tests for HybridRetriever (Phase 3)
import os
import pytest
import asyncio
from retrieval.hybrid_retriever import HybridRetriever

PGVECTOR_URL = os.getenv("PGVECTOR_URL")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

@pytest.fixture(scope="module")
def hybrid_retriever():
    return HybridRetriever(PGVECTOR_URL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

def test_hybrid_retrieve(hybrid_retriever):
    query = "Who works at Acme Corp?"
    # Run the async retrieve method
    results = asyncio.run(hybrid_retriever.retrieve(query, top_k=3))
    assert isinstance(results, list)
    assert len(results) <= 3
    # Each result should be a tuple (chunk, chunk_id, score)
    for chunk, chunk_id, score in results:
        assert isinstance(chunk, str)
        assert chunk_id is not None
        assert isinstance(score, float) or isinstance(score, int)
