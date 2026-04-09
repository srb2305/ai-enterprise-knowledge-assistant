# app.py
# Purpose: FastAPI backend for document ingestion and querying.
# Provides /ingest (file upload) and /query (question answering) endpoints.

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import tempfile
from ingestion.pipeline import ingest_document
from retrieval.vector_retriever import VectorStore
from retrieval.graph_retriever import GraphRetriever
from graph.graph_store import GraphStore
import glob
import json
from retrieval.hybrid_retriever import HybridRetriever

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
	Ingests an uploaded document: loads, chunks, embeds, and stores in pgvector.
	"""
	with tempfile.NamedTemporaryFile(delete=False) as tmp:
		tmp.write(await file.read())
		tmp_path = tmp.name
	chunks = ingest_document(tmp_path)
	vector_store.upsert_chunks(chunks)
	os.remove(tmp_path)
	return {"status": "success", "num_chunks": len(chunks)}

@app.post("/query")
async def query(question: str = Form(...)):
	"""
	Answers a user query by retrieving top-k relevant chunks from pgvector.
	"""
	results = vector_store.query(question)
	return JSONResponse({"results": [
		{"chunk": chunk, "score": score, "metadata": metadata}
		for chunk, score, metadata in results
	]})

@app.post("/graph_query")
async def graph_query(question: str = Form(...)):
    """
    Answers a user query by retrieving relevant chunk IDs from the graph.
    """
    chunk_ids = graph_retriever.retrieve(question)
    return JSONResponse({"chunk_ids": chunk_ids})

@app.post("/hybrid_query")
async def hybrid_query(question: str = Form(...), top_k: int = Form(5)):
    """
    Answers a user query using hybrid retrieval (vector + graph + rerank).
    """
    results = await hybrid_retriever.retrieve(question, top_k=top_k)
    return JSONResponse({
        "results": [
            {"chunk": chunk, "chunk_id": chunk_id, "score": float(score)}
            for chunk, chunk_id, score in results
        ]
    })
