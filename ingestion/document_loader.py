# document_loader.py
# Purpose: Loads documents of various types (PDF, DOCX, TXT, etc.) using LangChain loaders.
# This module abstracts away the file type detection and uses the appropriate loader for each document.

from langchain_community.document_loaders import Docx2txtLoader, TextLoader
try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
except ImportError:
    UnstructuredPDFLoader = None
import os

def load_document(file_path):
	"""
	Loads a document from the given file path using the appropriate LangChain loader.
	Supports PDF, DOCX, and TXT files. If extension is missing or unknown, tries to treat as text.
	Returns a list of document objects (one per page or chunk, depending on loader).
	"""
	ext = os.path.splitext(file_path)[1].lower()
	# If extension is missing, try to guess or treat as text
	if not ext:
		ext = ".txt"
	if ext == ".pdf":
		if UnstructuredPDFLoader is None:
			raise ImportError("unstructured package required for PDF loading: pip install unstructured")
		loader = UnstructuredPDFLoader(file_path)
	elif ext == ".docx":
		loader = Docx2txtLoader(file_path)
	elif ext == ".txt":
		loader = TextLoader(file_path)
	else:
		# Try to treat as text if unknown extension
		try:
			loader = TextLoader(file_path)
		except Exception as e:
			print(f"[WARN] Unsupported file type: {ext}. Error: {e}")
			raise ValueError(f"Unsupported file type: {ext}")
	return loader.load()
