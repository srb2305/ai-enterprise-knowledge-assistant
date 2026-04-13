# vector_retriever.py
# Purpose: Handles embedding of chunks and vector search in pgvector (PostgreSQL extension).
# Provides functions to upsert (store) and retrieve chunks by vector similarity.

import asyncio

import psycopg2
from sentence_transformers import SentenceTransformer

class VectorStore:
	def __init__(self, db_url, model_name="all-MiniLM-L6-v2"):
		"""
		Connects to pgvector database and loads embedding model.
		Args:
			db_url: PostgreSQL connection string.
			model_name: SentenceTransformer model for embeddings.
		"""
		self.conn = psycopg2.connect(db_url)
		self.model = SentenceTransformer(model_name)
		self._ensure_schema()

	def _ensure_schema(self):
		"""Ensures pgvector extension and table exist before use."""
		cur = self.conn.cursor()
		cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
		cur.execute('''CREATE TABLE IF NOT EXISTS documents (
			id SERIAL PRIMARY KEY,
			chunk TEXT NOT NULL,
			embedding VECTOR(384) NOT NULL,
			metadata JSONB
		)''')
		self.conn.commit()

	def upsert_chunks(self, chunks, metadata_list=None):
		"""
		Embeds and stores chunks in pgvector. Creates table if not exists.
		Args:
			chunks: List of text chunks.
			metadata_list: List of dicts with metadata for each chunk (optional).
		"""
		cur = self.conn.cursor()
		inserted_ids = []
		for i, chunk in enumerate(chunks):
			emb = self.model.encode(chunk)
			metadata = metadata_list[i] if metadata_list else None
			cur.execute(
				"""
				INSERT INTO documents (chunk, embedding, metadata)
				VALUES (%s, %s, %s)
				RETURNING id
				""",
				(chunk, emb.tolist(), metadata)
			)
			inserted_ids.append(cur.fetchone()[0])
		self.conn.commit()
		return inserted_ids

	def query(self, query_text, top_k=5):
		"""
		Embeds the query and retrieves top_k most similar chunks from pgvector.
		Args:
			query_text: The user query string.
			top_k: Number of results to return.
		Returns:
			List of dicts: {chunk_id, chunk, score, metadata}.
		"""
		cur = self.conn.cursor()
		query_emb = self.model.encode(query_text)
		cur.execute(
			"""
			SELECT id, chunk, metadata, embedding <=> %s::vector AS distance
			FROM documents
			ORDER BY distance ASC
			LIMIT %s
			""",
			(query_emb.tolist(), top_k)
		)
		results = cur.fetchall()
		return [
			{
				"chunk_id": row[0],
				"chunk": row[1],
				"metadata": row[2],
				"score": float(1 - row[3]),
			}
			for row in results
		]

	async def query_async(self, query_text, top_k=5):
		# Async wrapper for query (for HybridRetriever)
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(None, self.query, query_text, top_k)

	def get_chunk_text(self, chunk_id):
		"""
		Retrieves a chunk by its ID.
		Args:
			chunk_id: The ID of the chunk.
		Returns:
			The chunk text if found, None otherwise.
		"""
		cur = self.conn.cursor()
		cur.execute("SELECT chunk FROM documents WHERE id = %s", (chunk_id,))
		row = cur.fetchone()
		return row[0] if row else None

	def get_chunks_by_ids(self, chunk_ids):
		"""
		Retrieves chunks by a list of IDs.
		Args:
			chunk_ids: List of chunk IDs.
		Returns:
			List of dicts with chunk_id, chunk, metadata.
		"""
		if not chunk_ids:
			return []
		cur = self.conn.cursor()
		cur.execute(
			"""
			SELECT id, chunk, metadata
			FROM documents
			WHERE id = ANY(%s)
			""",
			(chunk_ids,),
		)
		rows = cur.fetchall()
		return [
			{"chunk_id": row[0], "chunk": row[1], "metadata": row[2]}
			for row in rows
		]
