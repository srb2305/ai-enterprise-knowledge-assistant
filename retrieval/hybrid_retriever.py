# HybridRetriever: Combines vector and graph retrieval, reranks with CrossEncoder
import asyncio
from sentence_transformers import CrossEncoder
from retrieval.vector_retriever import VectorStore as VectorRetriever
from retrieval.graph_retriever import GraphRetriever

class HybridRetriever:
    def __init__(self, pgvector_url, neo4j_uri, neo4j_user, neo4j_password, rerank_model='cross-encoder/ms-marco-MiniLM-L6-v2'):
        self.vector_retriever = VectorRetriever(pgvector_url)
        self.graph_retriever = GraphRetriever(neo4j_uri, neo4j_user, neo4j_password)
        self.cross_encoder = CrossEncoder(rerank_model)

    async def retrieve(self, query, top_k=5, hops=2):
        # Run both retrievers in parallel
        vector_task = asyncio.create_task(self.vector_retriever.query_async(query, top_k=top_k))
        graph_task = asyncio.create_task(self.graph_retriever.retrieve_async(query, hops=hops))
        vector_results, graph_chunk_ids = await asyncio.gather(vector_task, graph_task)

        # Merge and deduplicate by chunk_id
        chunk_id_to_data = {}
        for item in vector_results:
            chunk_id_to_data[item["chunk_id"]] = {
                "chunk": item["chunk"],
                "metadata": item.get("metadata"),
            }

        for chunk_id in graph_chunk_ids:
            if chunk_id not in chunk_id_to_data:
                # Fetch text for chunk_id from vector store
                text = self.vector_retriever.get_chunk_text(chunk_id)
                if text:
                    chunk_id_to_data[chunk_id] = {"chunk": text, "metadata": None}

        candidates = [
            (chunk_id, data["chunk"], data.get("metadata"))
            for chunk_id, data in chunk_id_to_data.items()
            if data.get("chunk")
        ]
        if not candidates:
            return []

        # Rerank with CrossEncoder
        pairs = [(query, text) for _, text, _ in candidates]
        scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top = reranked[:top_k]
        # Return dicts for consistent API serialization
        return [
            {
                "chunk_id": item[0][0],
                "chunk": item[0][1],
                "metadata": item[0][2],
                "score": float(item[1]),
            }
            for item in top
        ]
