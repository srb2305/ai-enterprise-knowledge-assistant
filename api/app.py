# app.py
# Purpose: FastAPI backend for document ingestion and querying.
# Provides /ingest (file upload) and /query (question answering) endpoints.

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import tempfile
from ingestion.pipeline import ingest_document, ingest_chunks_into_graph
from retrieval.vector_retriever import VectorStore
from retrieval.graph_retriever import GraphRetriever
from graph.graph_store import GraphStore
import glob
import json
from retrieval.hybrid_retriever import HybridRetriever
from agent.pipeline import run_self_correction_pipeline

# Update this with your actual PostgreSQL connection string
PGVECTOR_URL = os.getenv("PGVECTOR_URL", "postgresql://postgres:postgres@localhost:5432/pgvector")
vector_store = VectorStore(PGVECTOR_URL)

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test")

graph_retriever = GraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
hybrid_retriever = HybridRetriever(PGVECTOR_URL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

app = FastAPI()

# For direct graph access (neighbors)
graph_store = GraphStore(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
@app.get("/graph/neighbours/{entity}")
async def get_graph_neighbours(entity: str, hops: int = 2):
	"""
	Returns neighbors of an entity up to N hops for visualization.
	"""
	neighbours = graph_store.get_neighbors(entity, hops=hops)
	return JSONResponse({"entity": entity, "neighbours": neighbours})


# Simple latest eval results endpoint (looks for latest .json in evaluation/ or returns empty)
@app.get("/eval/latest")
async def get_latest_eval():
	"""
	Returns the latest evaluation results (expects .json files in evaluation/).
	"""
	eval_files = sorted(glob.glob(os.path.join("evaluation", "*.json")), reverse=True)
	for f in eval_files:
		if f.endswith("golden_dataset.json"):
			continue  # skip golden dataset
		try:
			with open(f, "r", encoding="utf-8") as file:
				data = json.load(file)
			return JSONResponse({"filename": os.path.basename(f), "results": data})
		except Exception as e:
			continue
	return JSONResponse({"results": []})

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
	"""
	Ingests an uploaded document: loads, chunks, stores vectors,
	and writes entities/relations to graph.
	"""
	with tempfile.NamedTemporaryFile(delete=False) as tmp:
		tmp.write(await file.read())
		tmp_path = tmp.name
	chunks = ingest_document(tmp_path)
	chunk_ids = vector_store.upsert_chunks(chunks)
	graph_summary = ingest_chunks_into_graph(chunks, chunk_ids, graph_store)
	os.remove(tmp_path)
	return {
		"status": "success",
		"num_chunks": len(chunks),
		"num_vector_rows": len(chunk_ids),
		"graph": graph_summary,
	}

@app.post("/query")
async def query(question: str = Form(...), top_k: int = Form(5)):
	"""
	Answers a user query by retrieving top-k relevant chunks from pgvector.
	"""
	results = vector_store.query(question, top_k=top_k)
	return JSONResponse({"results": results})

@app.post("/graph_query")
async def graph_query(question: str = Form(...)):
	"""
	Answers a user query by retrieving relevant chunk IDs from the graph
	and fetching corresponding chunks from pgvector.
	"""
	chunk_ids = graph_retriever.retrieve(question)
	results = vector_store.get_chunks_by_ids(chunk_ids) if chunk_ids else []
	return JSONResponse({"chunk_ids": chunk_ids, "results": results})

@app.post("/hybrid_query")
async def hybrid_query(question: str = Form(...), top_k: int = Form(5)):
	"""
	Answers a user query using hybrid retrieval (vector + graph + rerank).
	"""
	results = await hybrid_retriever.retrieve(question, top_k=top_k)
	return JSONResponse({"results": results})


@app.post("/self_correct_query")
async def self_correct_query(
	question: str = Form(...),
	top_k: int = Form(5),
	hops: int = Form(2),
	max_iterations: int = Form(2),
	min_score: float = Form(0.7),
):
	"""
	Answers a user query using the retrieve -> generate -> critique loop.
	"""
	state = await run_self_correction_pipeline(
		query=question,
		max_iterations=max_iterations,
		min_score=min_score,
		top_k=top_k,
		hops=hops,
		log_results=True,
	)
	return JSONResponse(
		{
			"answer": state.answer,
			"faithfulness_score": state.faithfulness_score,
			"relevance_score": state.relevance_score,
			"iteration_count": state.iteration_count,
			"trace": state.trace,
			"results": state.sources,
		}
	)
