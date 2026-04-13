# test_evaluation.py
# Tests for evaluation/ragas_eval.py, evaluation/mlflow_logger.py,
# and benchmarks/eval_report.py — no live services required.

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# ragas_eval: _token_overlap
# ---------------------------------------------------------------------------

def test_token_overlap_identical():
    from evaluation.ragas_eval import _token_overlap
    assert _token_overlap("hello world", "hello world") == pytest.approx(1.0)


def test_token_overlap_disjoint():
    from evaluation.ragas_eval import _token_overlap
    assert _token_overlap("hello world", "foo bar") == pytest.approx(0.0)


def test_token_overlap_empty():
    from evaluation.ragas_eval import _token_overlap
    assert _token_overlap("", "hello") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ragas_eval: _evaluate_proxy
# ---------------------------------------------------------------------------

def test_evaluate_proxy_returns_all_keys():
    from evaluation.ragas_eval import _evaluate_proxy
    records = [
        {
            "question": "Who is John?",
            "answer": "John Doe works at Acme.",
            "ground_truth": "John Doe is an employee.",
            "contexts": ["John Doe works at Acme Corp."],
        }
    ]
    result = _evaluate_proxy(records)
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "num_samples"]:
        assert key in result
    assert result["num_samples"] == 1
    for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert 0.0 <= result[k] <= 1.0


def test_evaluate_proxy_empty_records():
    from evaluation.ragas_eval import _evaluate_proxy
    result = _evaluate_proxy([])
    assert result["num_samples"] == 0
    assert result["faithfulness"] == 0.0


# ---------------------------------------------------------------------------
# ragas_eval: load_golden_dataset
# ---------------------------------------------------------------------------

def test_load_golden_dataset_valid():
    from evaluation.ragas_eval import load_golden_dataset
    data = [{"question": "q1", "ground_truth": "a1"}]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
        json.dump(data, f)
        path = f.name
    loaded = load_golden_dataset(path)
    os.remove(path)
    assert loaded == data


def test_load_golden_dataset_invalid_raises():
    from evaluation.ragas_eval import load_golden_dataset
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
        json.dump({"key": "not a list"}, f)
        path = f.name
    with pytest.raises(ValueError):
        load_golden_dataset(path)
    os.remove(path)


# ---------------------------------------------------------------------------
# ragas_eval: prepare_eval_records uses provided answers when present
# ---------------------------------------------------------------------------

def test_prepare_eval_records_uses_preloaded_answers():
    from evaluation.ragas_eval import prepare_eval_records
    dataset = [
        {
            "question": "q",
            "ground_truth": "gt",
            "answer": "preloaded answer",
            "contexts": ["ctx1"],
        }
    ]
    query_fn = MagicMock(return_value=("api answer", ["api ctx"]))
    records = prepare_eval_records(dataset, query_fn)
    # should use preloaded values, NOT call query_fn
    query_fn.assert_not_called()
    assert records[0]["answer"] == "preloaded answer"


def test_prepare_eval_records_calls_query_fn_when_missing():
    from evaluation.ragas_eval import prepare_eval_records
    dataset = [{"question": "q", "ground_truth": "gt"}]
    query_fn = MagicMock(return_value=("api answer", ["ctx"]))
    records = prepare_eval_records(dataset, query_fn)
    query_fn.assert_called_once_with("q")
    assert records[0]["answer"] == "api answer"


# ---------------------------------------------------------------------------
# ragas_eval: run_ragas_eval writes JSON output file
# ---------------------------------------------------------------------------

def test_run_ragas_eval_writes_output_file():
    from evaluation.ragas_eval import run_ragas_eval

    # golden dataset with preloaded answers so no HTTP call is made
    dataset = [
        {
            "question": "q",
            "ground_truth": "gt",
            "answer": "an answer",
            "contexts": ["some context"],
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "golden.json")
        output_path = os.path.join(tmpdir, "results.json")
        with open(dataset_path, "w") as f:
            json.dump(dataset, f)

        results, saved_path = run_ragas_eval(
            dataset_path=dataset_path,
            output_path=output_path,
        )

    assert os.path.exists(output_path)
    assert results["num_samples"] == 1
    assert "generated_at" in results


# ---------------------------------------------------------------------------
# mlflow_logger: log_metrics and log_state do not raise
# ---------------------------------------------------------------------------

def test_log_metrics_does_not_raise():
    with patch("mlflow.set_experiment"), patch("mlflow.start_run"), \
         patch("mlflow.log_metric"), patch("mlflow.log_param"):
        from evaluation.mlflow_logger import log_metrics
        log_metrics({"faithfulness": 0.8, "relevance": 0.7}, experiment_name="test")


def test_log_state_does_not_raise():
    with patch("mlflow.set_experiment"), patch("mlflow.start_run"), \
         patch("mlflow.log_metric"), patch("mlflow.log_param"):
        from evaluation.mlflow_logger import log_state
        from agent.graph_state import LangGraphState
        state = LangGraphState(
            query="test",
            answer="answer",
            faithfulness_score=0.9,
            relevance_score=0.8,
            iteration_count=1,
        )
        log_state(state)


# ---------------------------------------------------------------------------
# eval_report: build_report produces markdown table
# ---------------------------------------------------------------------------

def test_build_report_contains_all_metrics():
    from benchmarks.eval_report import build_report
    v = {"faithfulness": 0.61, "answer_relevancy": 0.70, "context_precision": 0.55, "context_recall": 0.60}
    h = {"faithfulness": 0.84, "answer_relevancy": 0.82, "context_precision": 0.77, "context_recall": 0.80}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as fv:
        json.dump(v, fv)
        vpath = fv.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as fh:
        json.dump(h, fh)
        hpath = fh.name

    report = build_report(vpath, hpath)
    os.remove(vpath)
    os.remove(hpath)

    assert "faithfulness" in report
    assert "context_precision" in report
    assert "+0.2300" in report or "+0.23" in report  # delta for faithfulness


def test_build_report_save_writes_file():
    from benchmarks.eval_report import save_report

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "report.md")
        saved = save_report("# Test Report\n| a | b |", output_path=output_path)
        assert os.path.exists(saved)
        content = open(saved).read()
        assert "# Test Report" in content
