# pipeline.py
# Purpose: Orchestrates the ingestion pipeline: loads documents, chunks them semantically, and returns chunks.

from .document_loader import load_document
from .chunker import semantic_chunk
from graph.entity_extractor import EntityExtractor
from graph.relation_extractor import RelationExtractor


def build_chunks(file_path):
	"""
	Loads a document, splits it into semantic chunks, and returns the chunks.
	Args:
		file_path: Path to the document file.
	Returns:
		List of text chunks.
	"""
	docs = load_document(file_path)
	all_chunks = []
	for doc in docs:
		text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
		chunks = semantic_chunk(text)
		all_chunks.extend(chunks)
	return all_chunks

def ingest_document(file_path):
	"""
	Loads a document, splits it into semantic chunks, and returns the chunks.
	Args:
		file_path: Path to the document file.
	Returns:
		List of text chunks.
	"""
	return build_chunks(file_path)


def ingest_chunks_into_graph(chunks, chunk_ids, graph_store, entity_extractor=None, relation_extractor=None):
	"""
	Writes entities and relationships extracted from chunks into Neo4j.
	Args:
		chunks: List of chunk texts.
		chunk_ids: List of chunk IDs aligned with chunks.
		graph_store: GraphStore instance.
		entity_extractor: Optional EntityExtractor instance.
		relation_extractor: Optional RelationExtractor instance.
	Returns:
		Dict summary with counts and non-fatal errors.
	"""
	if not chunks or not chunk_ids or len(chunks) != len(chunk_ids):
		return {
			"entities_written": 0,
			"relations_written": 0,
			"errors": ["chunks and chunk_ids must be non-empty and aligned"],
		}

	entity_extractor = entity_extractor or EntityExtractor()
	relation_extractor = relation_extractor or RelationExtractor()

	entities_written = 0
	relations_written = 0
	errors = []

	for chunk_text, chunk_id in zip(chunks, chunk_ids):
		try:
			entities = entity_extractor.extract_entities([chunk_text])
			for entity_type, entity_list in entities.items():
				for entity in entity_list:
					graph_store.upsert_entity(entity, entity_type, chunk_id=chunk_id)
					entities_written += 1
		except Exception as exc:
			errors.append(f"entity extraction failed for chunk_id={chunk_id}: {exc}")

		try:
			triples = relation_extractor.extract_relations(chunk_text)
			for triple in triples:
				subj = triple.get("subject")
				pred = triple.get("predicate")
				obj = triple.get("object")
				if subj and pred and obj:
					graph_store.upsert_relationship(subj, pred, obj, chunk_id=chunk_id)
					relations_written += 1
		except Exception as exc:
			errors.append(f"relation extraction failed for chunk_id={chunk_id}: {exc}")

	return {
		"entities_written": entities_written,
		"relations_written": relations_written,
		"errors": errors,
	}
