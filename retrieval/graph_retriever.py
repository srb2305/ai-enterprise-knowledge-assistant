# graph_retriever.py
# Purpose: Retrieves relevant chunks by extracting entities from queries and traversing the Neo4j graph.

from graph.entity_extractor import EntityExtractor
from graph.graph_store import GraphStore
import asyncio

class GraphRetriever:
	def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
		self.graph_store = GraphStore(neo4j_uri, neo4j_user, neo4j_password)
		self.entity_extractor = EntityExtractor()

	def retrieve(self, query, hops=2):
		"""
		Extracts entities from the query, finds neighbors in Neo4j, and returns linked chunk IDs.
		Args:
			query: User query string
			hops: Number of hops for neighbor search
		Returns:
			List of chunk IDs (and optionally entity info)
		"""
		entities = self.entity_extractor.extract_entities([query])
		chunk_ids = set()
		for entity_type, entity_list in entities.items():
			for entity in entity_list:
				neighbors = self.graph_store.get_neighbors(entity, hops=hops)
				for n in neighbors:
					if n.get('chunk_id'):
						chunk_ids.add(n['chunk_id'])
		return list(chunk_ids)

	async def retrieve_async(self, query, hops=2):
		"""
		Extracts entities from the query, finds neighbors in Neo4j, and returns linked chunk IDs.
		Args:
			query: User query string
			hops: Number of hops for neighbor search
		Returns:
			List of chunk IDs (and optionally entity info)
		"""
		import asyncio
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(None, self.retrieve, query, hops)
