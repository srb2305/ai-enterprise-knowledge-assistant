# test_agent_pipeline.py
# Unit tests for agent/pipeline.py and agent/nodes.py
# All LLM and hybrid retriever calls are mocked.

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.graph_state import LangGraphState


# ---------------------------------------------------------------------------
# Helpers: patch out DB connections at module level
# ---------------------------------------------------------------------------

PATCHES = {
    "psycopg2.connect": MagicMock(),
    "retrieval.vector_retriever.SentenceTransformer": MagicMock(),
    "neo4j.GraphDatabase.driver": MagicMock(),
    "sentence_transformers.CrossEncoder": MagicMock(),
    "langchain_community.llms.OpenAI": MagicMock(),
}


# ---------------------------------------------------------------------------
# decision_node
# ---------------------------------------------------------------------------

def test_decision_node_returns_when_scores_high():
    from agent.nodes import decision_node
    state = LangGraphState(
        query="test",
        faithfulness_score=0.9,
        relevance_score=0.85,
        iteration_count=0,
    )
    assert decision_node(state, min_score=0.7, max_iterations=2) == "return"


def test_decision_node_iterates_when_scores_low():
    from agent.nodes import decision_node
    state = LangGraphState(
        query="test",
        faithfulness_score=0.3,
        relevance_score=0.4,
        iteration_count=0,
    )
    assert decision_node(state, min_score=0.7, max_iterations=2) == "iterate"


def test_decision_node_returns_at_max_iterations():
    from agent.nodes import decision_node
    state = LangGraphState(
        query="test",
        faithfulness_score=0.3,
        relevance_score=0.4,
        iteration_count=2,
    )
    assert decision_node(state, min_score=0.7, max_iterations=2) == "return"


def test_decision_node_none_scores_iterates():
    from agent.nodes import decision_node
    state = LangGraphState(query="test", iteration_count=0)
    assert decision_node(state, min_score=0.7, max_iterations=2) == "iterate"


# ---------------------------------------------------------------------------
# generate_node - fallback path when LLM raises
# ---------------------------------------------------------------------------

def test_generate_node_llm_fallback():
    with patch("agent.nodes.llm") as mock_llm:
        mock_llm.invoke.side_effect = Exception("no API key")
        from agent.nodes import generate_node
        state = LangGraphState(
            query="test",
            context_chunks=["The sky is blue."],
        )
        result = generate_node(state)
        assert result.answer is not None
        assert len(result.answer) > 0


def test_generate_node_empty_context_fallback():
    with patch("agent.nodes.llm") as mock_llm:
        mock_llm.invoke.side_effect = Exception("no API key")
        from agent.nodes import generate_node
        state = LangGraphState(query="test", context_chunks=[])
        result = generate_node(state)
        assert "No relevant context" in result.answer


# ---------------------------------------------------------------------------
# critique_node - fallback path when LLM raises
# ---------------------------------------------------------------------------

def test_critique_node_llm_fallback_sets_scores():
    with patch("agent.nodes.llm") as mock_llm:
        mock_llm.invoke.side_effect = Exception("timeout")
        from agent.nodes import critique_node
        state = LangGraphState(
            query="test",
            context_chunks=["John Doe works at Acme Corp"],
            answer="John Doe works at Acme Corp",
        )
        result = critique_node(state)
        assert result.faithfulness_score is not None
        assert result.relevance_score is not None
        assert 0.0 <= result.faithfulness_score <= 1.0
        assert 0.0 <= result.relevance_score <= 1.0


def test_critique_node_parses_valid_llm_json():
    with patch("agent.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = '{"faithfulness": 0.92, "relevance": 0.88}'
        from agent.nodes import critique_node
        state = LangGraphState(
            query="test",
            context_chunks=["context"],
            answer="answer",
        )
        result = critique_node(state)
        assert result.faithfulness_score == pytest.approx(0.92)
        assert result.relevance_score == pytest.approx(0.88)


def test_critique_node_invalid_json_defaults_to_zero():
    with patch("agent.nodes.llm") as mock_llm:
        mock_llm.invoke.return_value = "not json at all"
        from agent.nodes import critique_node
        state = LangGraphState(query="test", context_chunks=["c"], answer="a")
        result = critique_node(state)
        assert result.faithfulness_score == 0.0
        assert result.relevance_score == 0.0


# ---------------------------------------------------------------------------
# run_self_correction_pipeline (integration, all I/O mocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_exits_after_good_scores():
    """Pipeline should stop after first iteration when scores are high enough."""
    mock_results = [{"chunk_id": 1, "chunk": "context text", "score": 0.9, "metadata": None}]

    with patch("agent.nodes.hybrid_retriever") as mock_hr, \
         patch("agent.nodes.llm") as mock_llm:

        mock_hr.retrieve = AsyncMock(return_value=mock_results)
        # LLM: answer on first invoke, valid JSON critique on second
        mock_llm.invoke.side_effect = [
            "The answer from context.",
            '{"faithfulness": 0.95, "relevance": 0.90}',
        ]

        from agent.pipeline import run_self_correction_pipeline
        state = await run_self_correction_pipeline(
            query="test query",
            max_iterations=2,
            min_score=0.7,
            log_results=False,
        )

    assert state.answer == "The answer from context."
    assert state.faithfulness_score == pytest.approx(0.95)
    assert state.iteration_count == 0
    assert len(state.trace) == 1
    assert state.trace[0]["action"] == "return"


@pytest.mark.asyncio
async def test_pipeline_iterates_on_low_scores():
    """Pipeline should iterate twice when first scores are below threshold."""
    mock_results = [{"chunk_id": 1, "chunk": "ctx", "score": 0.5, "metadata": None}]

    with patch("agent.nodes.hybrid_retriever") as mock_hr, \
         patch("agent.nodes.llm") as mock_llm:

        mock_hr.retrieve = AsyncMock(return_value=mock_results)
        mock_llm.invoke.side_effect = [
            "weak answer",
            '{"faithfulness": 0.3, "relevance": 0.3}',  # iteration 0 low
            "better answer",
            '{"faithfulness": 0.85, "relevance": 0.80}',  # iteration 1 high
        ]

        from agent.pipeline import run_self_correction_pipeline
        state = await run_self_correction_pipeline(
            query="hard query",
            max_iterations=2,
            min_score=0.7,
            log_results=False,
        )

    assert state.answer == "better answer"
    assert state.iteration_count >= 1


@pytest.mark.asyncio
async def test_pipeline_respects_max_iterations():
    """Pipeline must stop at max_iterations even if scores stay low."""
    mock_results = [{"chunk_id": 1, "chunk": "ctx", "score": 0.5, "metadata": None}]

    with patch("agent.nodes.hybrid_retriever") as mock_hr, \
         patch("agent.nodes.llm") as mock_llm:

        mock_hr.retrieve = AsyncMock(return_value=mock_results)
        # Always low scores
        mock_llm.invoke.side_effect = [
            "ans", '{"faithfulness": 0.2, "relevance": 0.2}',
            "ans", '{"faithfulness": 0.2, "relevance": 0.2}',
            "ans", '{"faithfulness": 0.2, "relevance": 0.2}',
        ]

        from agent.pipeline import run_self_correction_pipeline
        state = await run_self_correction_pipeline(
            query="ambiguous",
            max_iterations=2,
            min_score=0.7,
            log_results=False,
        )

    assert state.iteration_count <= 2


@pytest.mark.asyncio
async def test_pipeline_state_has_sources_and_trace():
    mock_results = [{"chunk_id": 5, "chunk": "data", "score": 0.8, "metadata": None}]

    with patch("agent.nodes.hybrid_retriever") as mock_hr, \
         patch("agent.nodes.llm") as mock_llm:

        mock_hr.retrieve = AsyncMock(return_value=mock_results)
        mock_llm.invoke.side_effect = [
            "an answer",
            '{"faithfulness": 0.9, "relevance": 0.9}',
        ]

        from agent.pipeline import run_self_correction_pipeline
        state = await run_self_correction_pipeline(
            query="q", max_iterations=2, min_score=0.7, log_results=False
        )

    assert isinstance(state.sources, list)
    assert isinstance(state.trace, list)
    assert state.trace[0]["faithfulness"] is not None
