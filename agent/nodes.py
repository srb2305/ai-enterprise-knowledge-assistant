import mlflow
from evaluation import mlflow_logger
def log_node(state: LangGraphState, experiment_name: str = "GraphRAG") -> LangGraphState:
	"""
	Log node for LangGraph: logs the state to MLflow and LangSmith (if available).
	"""
	# MLflow logging
	mlflow.set_experiment(experiment_name)
	with mlflow.start_run():
		mlflow.log_param("query", state.query)
		mlflow.log_param("answer", state.answer)
		mlflow.log_param("iteration_count", state.iteration_count)
		mlflow.log_metric("faithfulness_score", state.faithfulness_score or 0.0)
		mlflow.log_metric("relevance_score", state.relevance_score or 0.0)
		for i, chunk in enumerate(state.context_chunks):
			mlflow.log_param(f"context_chunk_{i}", chunk)
	# Optionally, call custom logger for more advanced logging
	if hasattr(mlflow_logger, "log_state"):
		mlflow_logger.log_state(state)
	# LangSmith logging can be added here if needed
	return state
def decision_node(state: LangGraphState, min_score: float = 0.7, max_iterations: int = 2) -> str:
	"""
	Decision node for LangGraph: checks scores and decides next action.
	Returns 'return' if answer is good, 'iterate' if another round is needed.
	"""
	if (
		(state.faithfulness_score is not None and state.faithfulness_score >= min_score)
		and (state.relevance_score is not None and state.relevance_score >= min_score)
	):
		return "return"
	if state.iteration_count >= max_iterations:
		return "return"
	return "iterate"
import json
def format_critique_prompt(context_chunks: list[str], answer: str) -> str:
	context = "\n\n".join(context_chunks)
	prompt = f"""You are a strict evaluator. Given the following context and answer, rate the answer's faithfulness (does it stick to the context, 0-1) and relevance (does it answer the question, 0-1). Return your response as JSON: {{'faithfulness': float, 'relevance': float}}.\n\nContext:\n{context}\n\nAnswer:\n{answer}\n\nRespond only with the JSON object."""
	return prompt
def critique_node(state: LangGraphState) -> LangGraphState:
	"""
	Critique node for LangGraph: uses LLM to rate faithfulness and relevance of the answer.
	"""
	prompt = format_critique_prompt(state.context_chunks, state.answer)
	critique = llm.invoke(prompt)
	try:
		result = json.loads(critique)
		state.faithfulness_score = float(result.get('faithfulness', 0.0))
		state.relevance_score = float(result.get('relevance', 0.0))
	except Exception:
		state.faithfulness_score = 0.0
		state.relevance_score = 0.0
	return state
# retrieve, generate, critique, log nodes

from agent.graph_state import LangGraphState
from retrieval.hybrid_retriever import HybridRetriever

import os

# Example: load from environment or config
PGVECTOR_URL = os.getenv("PGVECTOR_URL", "postgresql://postgres:root@localhost:5432/pgvector")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

hybrid_retriever = HybridRetriever(PGVECTOR_URL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# LLM setup (using langchain OpenAI, can be replaced with other providers)
from langchain_community.llms import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...your-key...")
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.2)

def format_prompt(query: str, context_chunks: list[str]) -> str:
	context = "\n\n".join(context_chunks)
	prompt = f"""You are an expert assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer in a concise, factual manner."""
	return prompt

async def retrieve_node(state: LangGraphState, top_k: int = 5, hops: int = 2) -> LangGraphState:
	"""
	Retrieve node for LangGraph: fetches top context chunks using HybridRetriever and updates the state.
	"""
	results = await hybrid_retriever.retrieve(state.query, top_k=top_k, hops=hops)
	# results: list of (chunk_id, text, score)
	context_chunks = [text for text, chunk_id, score in results]
	state.context_chunks = context_chunks
	return state


def generate_node(state: LangGraphState) -> LangGraphState:
	"""
	Generate node for LangGraph: uses LLM to generate an answer from query and context.
	"""
	prompt = format_prompt(state.query, state.context_chunks)
	answer = llm.invoke(prompt)
	state.answer = answer.strip()
	return state
