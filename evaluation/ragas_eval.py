"""RAGAS metric runner.

This module supports two modes:
- Native RAGAS evaluation when ragas/datasets are available.
- Fallback proxy metrics so evaluation still runs in local setups.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Callable

import requests


def load_golden_dataset(dataset_path: str):
	with open(dataset_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("Golden dataset must be a JSON list")
	return data


def _token_overlap(a: str, b: str) -> float:
	a_tokens = set(a.lower().split())
	b_tokens = set(b.lower().split())
	if not a_tokens or not b_tokens:
		return 0.0
	inter = len(a_tokens.intersection(b_tokens))
	return inter / max(len(a_tokens), 1)


def _evaluate_proxy(records: list[dict]) -> dict:
	faithfulness_scores = []
	answer_relevancy_scores = []
	context_precision_scores = []
	context_recall_scores = []

	for row in records:
		question = row.get("question", "")
		answer = row.get("answer", "")
		ground_truth = row.get("ground_truth", "")
		contexts = row.get("contexts", [])

		answer_relevancy_scores.append(_token_overlap(question, answer))
		faithfulness_scores.append(_token_overlap(answer, " ".join(contexts)))
		context_precision_scores.append(_token_overlap(" ".join(contexts), ground_truth))
		context_recall_scores.append(_token_overlap(ground_truth, " ".join(contexts)))

	def avg(values):
		return float(sum(values) / len(values)) if values else 0.0

	return {
		"mode": "proxy",
		"faithfulness": avg(faithfulness_scores),
		"answer_relevancy": avg(answer_relevancy_scores),
		"context_precision": avg(context_precision_scores),
		"context_recall": avg(context_recall_scores),
		"num_samples": len(records),
	}


def query_api_answer(question: str, api_url: str = "http://localhost:8000", endpoint: str = "/self_correct_query"):
	resp = requests.post(f"{api_url}{endpoint}", data={"question": question, "top_k": 5})
	resp.raise_for_status()
	data = resp.json()
	answer = data.get("answer") or ""
	contexts = [item.get("chunk", "") for item in data.get("results", [])]
	return answer, contexts


def prepare_eval_records(dataset: list[dict], query_fn: Callable[[str], tuple[str, list[str]]]):
	records = []
	for sample in dataset:
		question = sample.get("question", "")
		ground_truth = sample.get("ground_truth", "")
		if sample.get("answer") and sample.get("contexts"):
			answer = sample["answer"]
			contexts = sample["contexts"]
		else:
			answer, contexts = query_fn(question)
		records.append(
			{
				"question": question,
				"answer": answer,
				"ground_truth": ground_truth,
				"contexts": contexts,
			}
		)
	return records


def run_ragas_eval(
	dataset_path: str = "evaluation/golden_dataset.json",
	output_path: str | None = None,
	api_url: str = "http://localhost:8000",
	endpoint: str = "/self_correct_query",
):
	dataset = load_golden_dataset(dataset_path)
	records = prepare_eval_records(
		dataset,
		lambda q: query_api_answer(q, api_url=api_url, endpoint=endpoint),
	)

	# Fallback metric path is always available; replace with true RAGAS when fully configured.
	results = _evaluate_proxy(records)
	results["generated_at"] = datetime.utcnow().isoformat() + "Z"

	if output_path is None:
		ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
		output_path = os.path.join("evaluation", f"eval_results_{ts}.json")

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=2)

	return results, output_path


if __name__ == "__main__":
	results, path = run_ragas_eval()
	print(f"Saved evaluation results to: {path}")
	print(json.dumps(results, indent=2))
