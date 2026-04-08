# test_graph_retriever.py
# Tests for GraphRetriever (Phase 3)
import os
import pytest
from retrieval.graph_retriever import GraphRetriever

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

@pytest.fixture(scope="module")
def graph_retriever():
    return GraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

def test_graph_retrieve(graph_retriever):
    query = "Who works at Acme Corp?"
    chunk_ids = graph_retriever.retrieve(query)
    assert isinstance(chunk_ids, list)
