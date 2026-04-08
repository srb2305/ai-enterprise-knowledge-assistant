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
        chunk_id_to_text = {}
        for text, chunk_id in vector_results:
            chunk_id_to_text[chunk_id] = text
        for chunk_id in graph_chunk_ids:
            if chunk_id not in chunk_id_to_text:
                # Fetch text for chunk_id from vector store
                text = self.vector_retriever.get_chunk_text(chunk_id)
                chunk_id_to_text[chunk_id] = text

        candidates = list(chunk_id_to_text.items())
        # Rerank with CrossEncoder
        pairs = [(query, text) for _, text in candidates]
        scores = self.cross_encoder.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top = reranked[:top_k]
        # Return [(text, chunk_id, score)]
        return [(c[0][1], c[0][0], c[1]) for c in top]
