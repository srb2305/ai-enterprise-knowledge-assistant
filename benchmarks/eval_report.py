"""Generate a markdown report comparing two evaluation runs."""

from __future__ import annotations

import json
import os


def load_json(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def metric_row(name: str, left: float, right: float) -> str:
	delta = right - left
	return f"| {name} | {left:.4f} | {right:.4f} | {delta:+.4f} |"


def build_report(vector_path: str, hybrid_path: str) -> str:
	v = load_json(vector_path)
	h = load_json(hybrid_path)

	metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
	lines = [
		"# Evaluation Comparison Report",
		"",
		f"Vector file: {vector_path}",
		f"Hybrid file: {hybrid_path}",
		"",
		"| Metric | Vector | Hybrid | Delta |",
		"|---|---:|---:|---:|",
	]

	for m in metrics:
		lines.append(metric_row(m, float(v.get(m, 0.0)), float(h.get(m, 0.0))))

	return "\n".join(lines)


def save_report(report_md: str, output_path: str = "benchmarks/eval_report.md") -> str:
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		f.write(report_md)
	return output_path


if __name__ == "__main__":
	# Example:
	# python benchmarks/eval_report.py evaluation/vector.json evaluation/hybrid.json
	import sys

	if len(sys.argv) < 3:
		raise SystemExit("Usage: python benchmarks/eval_report.py <vector_eval.json> <hybrid_eval.json>")

	report = build_report(sys.argv[1], sys.argv[2])
	out = save_report(report)
	print(f"Report written to: {out}")
