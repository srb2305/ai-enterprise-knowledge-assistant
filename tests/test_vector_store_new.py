# test_vector_store_new.py
# Tests for VectorStore new methods: get_chunks_by_ids, upsert returning IDs,
# schema initialization, and dict-shaped query output.
# All tests mock psycopg2 so no live database is needed.

from unittest.mock import MagicMock, patch, call
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cursor(fetchone=None, fetchall=None):
    cur = MagicMock()
    cur.fetchone = MagicMock(return_value=fetchone)
    cur.fetchall = MagicMock(return_value=fetchall or [])
    return cur


def _make_connection(cursor):
    conn = MagicMock()
    conn.cursor = MagicMock(return_value=cursor)
    return conn


# ---------------------------------------------------------------------------
# upsert_chunks returns list of IDs
# ---------------------------------------------------------------------------

def test_upsert_chunks_returns_ids():
    cur = _make_cursor(fetchone=(1,))
    conn = _make_connection(cur)

    with patch("psycopg2.connect", return_value=conn), \
         patch("retrieval.vector_retriever.SentenceTransformer") as MockST:
        MockST.return_value.encode = MagicMock(return_value=[0.1] * 384)

        from retrieval.vector_retriever import VectorStore
        vs = VectorStore("postgresql://test/db")

        ids = vs.upsert_chunks(["chunk one", "chunk two"])

    assert ids == [1, 1]  # mocked to return (1,) twice


# ---------------------------------------------------------------------------
# query returns list of dicts
# ---------------------------------------------------------------------------

def test_query_returns_dicts():
    rows = [(10, "Acme Corp text", {"source": "doc1"}, 0.2)]
    cur = _make_cursor(fetchall=rows)
    conn = _make_connection(cur)

    with patch("psycopg2.connect", return_value=conn), \
         patch("retrieval.vector_retriever.SentenceTransformer") as MockST:
        MockST.return_value.encode = MagicMock(return_value=[0.0] * 384)

        from retrieval.vector_retriever import VectorStore
        vs = VectorStore("postgresql://test/db")
        results = vs.query("Acme Corp", top_k=1)

    assert isinstance(results, list)
    assert len(results) == 1
    r = results[0]
    assert r["chunk_id"] == 10
    assert r["chunk"] == "Acme Corp text"
    assert isinstance(r["score"], float)
    assert r["score"] == pytest.approx(0.8, abs=1e-4)


# ---------------------------------------------------------------------------
# get_chunks_by_ids
# ---------------------------------------------------------------------------

def test_get_chunks_by_ids_returns_list():
    rows = [(5, "chunk A", None), (6, "chunk B", {"page": 2})]
    cur = _make_cursor(fetchall=rows)
    conn = _make_connection(cur)

    with patch("psycopg2.connect", return_value=conn), \
         patch("retrieval.vector_retriever.SentenceTransformer"):

        from retrieval.vector_retriever import VectorStore
        vs = VectorStore("postgresql://test/db")
        result = vs.get_chunks_by_ids([5, 6])

    assert len(result) == 2
    assert result[0] == {"chunk_id": 5, "chunk": "chunk A", "metadata": None}
    assert result[1]["metadata"] == {"page": 2}


def test_get_chunks_by_ids_empty_input():
    conn = _make_connection(_make_cursor())
    with patch("psycopg2.connect", return_value=conn), \
         patch("retrieval.vector_retriever.SentenceTransformer"):

        from retrieval.vector_retriever import VectorStore
        vs = VectorStore("postgresql://test/db")
        result = vs.get_chunks_by_ids([])

    assert result == []


# ---------------------------------------------------------------------------
# get_chunk_text
# ---------------------------------------------------------------------------

def test_get_chunk_text_found():
    cur = _make_cursor(fetchone=("hello world",))
    conn = _make_connection(cur)

    with patch("psycopg2.connect", return_value=conn), \
         patch("retrieval.vector_retriever.SentenceTransformer"):

        from retrieval.vector_retriever import VectorStore
        vs = VectorStore("postgresql://test/db")
        text = vs.get_chunk_text(3)

    assert text == "hello world"


def test_get_chunk_text_not_found():
    cur = _make_cursor(fetchone=None)
    conn = _make_connection(cur)

    with patch("psycopg2.connect", return_value=conn), \
         patch("retrieval.vector_retriever.SentenceTransformer"):

        from retrieval.vector_retriever import VectorStore
        vs = VectorStore("postgresql://test/db")
        text = vs.get_chunk_text(999)

    assert text is None
