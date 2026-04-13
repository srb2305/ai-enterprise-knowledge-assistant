# Compiled LangGraph pipeline

import asyncio

from agent.graph_state import LangGraphState
from agent.nodes import critique_node, decision_node, generate_node, log_node, retrieve_node


async def run_self_correction_pipeline(
	query: str,
	max_iterations: int = 2,
	min_score: float = 0.7,
	top_k: int = 5,
	hops: int = 2,
	log_results: bool = True,
) -> LangGraphState:
	"""
	Runs retrieve -> generate -> critique loop with conditional re-retrieval.
	Returns final state after decision or max iterations.
	"""
	state = LangGraphState(query=query)

	for i in range(max_iterations + 1):
		state.iteration_count = i
		state = await retrieve_node(state, top_k=top_k, hops=hops)
		state = generate_node(state)
		state = critique_node(state)

		action = decision_node(state, min_score=min_score, max_iterations=max_iterations)
		state.trace.append(
			{
				"iteration": i,
				"faithfulness": state.faithfulness_score,
				"relevance": state.relevance_score,
				"action": action,
			}
		)
		if action == "return":
			break

	if log_results:
		log_node(state)

	return state


def run_self_correction_pipeline_sync(
	query: str,
	max_iterations: int = 2,
	min_score: float = 0.7,
	top_k: int = 5,
	hops: int = 2,
	log_results: bool = True,
) -> LangGraphState:
	"""Synchronous wrapper for environments that do not manage an event loop."""
	return asyncio.run(
		run_self_correction_pipeline(
			query=query,
			max_iterations=max_iterations,
			min_score=min_score,
			top_k=top_k,
			hops=hops,
			log_results=log_results,
		)
	)
