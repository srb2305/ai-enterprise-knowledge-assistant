"""Utility functions for MLflow logging."""

from __future__ import annotations

import mlflow


def log_metrics(metrics: dict, experiment_name: str = "GraphRAG-Eval"):
	"""Log a flat dictionary of metrics to MLflow."""
	mlflow.set_experiment(experiment_name)
	with mlflow.start_run():
		for k, v in metrics.items():
			if isinstance(v, (int, float)):
				mlflow.log_metric(k, float(v))
			else:
				mlflow.log_param(k, str(v))


def log_state(state, experiment_name: str = "GraphRAG"):
	"""Log LangGraph state details safely to MLflow."""
	mlflow.set_experiment(experiment_name)
	with mlflow.start_run():
		mlflow.log_param("query", getattr(state, "query", ""))
		mlflow.log_param("answer", getattr(state, "answer", ""))
		mlflow.log_metric("iteration_count", float(getattr(state, "iteration_count", 0)))
		mlflow.log_metric("faithfulness_score", float(getattr(state, "faithfulness_score", 0.0) or 0.0))
		mlflow.log_metric("relevance_score", float(getattr(state, "relevance_score", 0.0) or 0.0))
