# graph_store.py
# Purpose: Handles Neo4j integration for storing entities, relationships, and linking to chunk IDs.

from neo4j import GraphDatabase

class GraphStore:
	def __init__(self, uri, user, password):
		self.driver = GraphDatabase.driver(uri, auth=(user, password))

	def close(self):
		self.driver.close()

	def upsert_entity(self, entity_text, entity_type, chunk_id=None):
		"""
		Creates or updates an entity node in Neo4j, optionally linking to a chunk ID.
		"""
		with self.driver.session() as session:
			session.run(
				"""
				MERGE (e:Entity {name: $name, type: $type})
				SET e.chunk_id = $chunk_id
				""",
				name=entity_text, type=entity_type, chunk_id=chunk_id
			)

	def upsert_relationship(self, subj, pred, obj, chunk_id=None):
		"""
		Creates or updates a relationship (edge) between two entities, optionally linking to a chunk ID.
		"""
		with self.driver.session() as session:
			session.run(
				"""
				MERGE (s:Entity {name: $subj})
				MERGE (o:Entity {name: $obj})
				MERGE (s)-[r:RELATION {type: $pred}]->(o)
				SET r.chunk_id = $chunk_id
				""",
				subj=subj, obj=obj, pred=pred, chunk_id=chunk_id
			)

	def get_neighbors(self, entity_name, hops=2):
		"""
		Retrieves neighbors up to N hops from the given entity.
		"""
		with self.driver.session() as session:
			result = session.run(
				f"""
				MATCH (n:Entity {{name: $name}})-[*1..{hops}]-(m)
				RETURN DISTINCT m.name AS name, m.type AS type, m.chunk_id AS chunk_id
				""",
				name=entity_name
			)
			return [dict(record) for record in result]
