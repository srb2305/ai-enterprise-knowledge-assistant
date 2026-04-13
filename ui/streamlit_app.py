from typing import List
import os
import networkx as nx
import matplotlib.pyplot as plt


import streamlit as st
import requests
import pandas as pd
import io

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Enterprise Knowledge Assistant (Hybrid GraphRAG)")

st.header("1. Document Ingestion")
uploaded_file = st.file_uploader(
	"Upload a document for ingestion",
	type=["pdf", "txt", "docx", "csv", "xls", "xlsx"]
)
if uploaded_file is not None:
	file_ext = uploaded_file.name.split(".")[-1].lower()
	if file_ext in ["csv", "xls", "xlsx"]:
		# Convert tabular data to plain text for ingestion
		try:
			if file_ext == "csv":
				df = pd.read_csv(uploaded_file)
			else:
				df = pd.read_excel(uploaded_file)
			text = df.to_string(index=False)
			text_bytes = text.encode("utf-8")
			file_for_upload = io.BytesIO(text_bytes)
			file_for_upload.name = uploaded_file.name + ".txt"
			files = {"file": (file_for_upload.name, file_for_upload, "text/plain")}
		except Exception as e:
			st.error(f"Failed to process spreadsheet: {e}")
			files = None
	else:
		files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
	if files:
		with st.spinner("Uploading and ingesting..."):
			resp = requests.post(f"{API_URL}/ingest", files=files)
			if resp.ok:
				st.success(f"Ingestion complete: {resp.json().get('num_chunks', 0)} chunks ingested.")
			else:
				st.error(f"Ingestion failed: {resp.text}")

st.header("2. Ask a Question")
question = st.text_input("Enter your question:")
query_mode = st.selectbox(
	"Retrieval mode",
	options=["vector", "hybrid", "self-correct"],
	index=1,
	help="vector: pgvector only, hybrid: vector+graph+rerank, self-correct: hybrid plus critique loop",
)
top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5)

if st.button("Query") and question:
	with st.spinner("Retrieving answer..."):
		if query_mode == "vector":
			endpoint = "/query"
			payload = {"question": question, "top_k": top_k}
		elif query_mode == "hybrid":
			endpoint = "/hybrid_query"
			payload = {"question": question, "top_k": top_k}
		else:
			endpoint = "/self_correct_query"
			payload = {"question": question, "top_k": top_k, "max_iterations": 2, "min_score": 0.7}

		resp = requests.post(f"{API_URL}{endpoint}", data=payload)
		if resp.ok:
			data = resp.json()
			results = data.get("results", [])

			if query_mode == "self-correct":
				st.subheader("Answer")
				st.write(data.get("answer", ""))
				st.caption(
					f"Faithfulness: {data.get('faithfulness_score', 0):.2f} | "
					f"Relevance: {data.get('relevance_score', 0):.2f} | "
					f"Iterations: {data.get('iteration_count', 0)}"
				)
				if data.get("trace"):
					with st.expander("Iteration Trace"):
						st.json(data.get("trace"))

			if results:
				for idx, res in enumerate(results):
					score = res.get("score")
					header = f"**Chunk {idx+1}"
					if score is not None:
						header += f" (Score: {float(score):.2f})"
					header += "**"
					st.markdown(f"{header}\n{res.get('chunk', '')}")
					if res.get("metadata"):
						st.caption(str(res["metadata"]))
					if res.get("chunk_id") is not None:
						st.caption(f"chunk_id: {res['chunk_id']}")
			else:
				st.info("No results found.")
		else:
			st.error(f"Query failed: {resp.text}")
			
st.header("3. Graph Neighbors Visualization")
entity = st.text_input("Enter entity name to visualize neighbors:", key="entity_input")
num_hops = st.slider("Number of hops", min_value=1, max_value=4, value=2)
if st.button("Show Neighbors") and entity:
	with st.spinner("Fetching neighbors from graph..."):
		resp = requests.get(f"{API_URL}/graph/neighbours/{entity}", params={"hops": num_hops})
		if resp.ok:
			data = resp.json()
			neighbours = data.get("neighbours", [])
			if neighbours:
				# Visualize with networkx and matplotlib
				G = nx.Graph()
				G.add_node(entity, color='red')
				for n in neighbours:
					n_name = n.get("m.name") or n.get("name")
					if n_name:
						G.add_node(n_name)
						G.add_edge(entity, n_name)
				fig, ax = plt.subplots()
				nx.draw(G, with_labels=True, node_color='lightblue', ax=ax)
				st.pyplot(fig)
				st.write("Neighbors:", [n.get("name") for n in neighbours])
			else:
				st.info("No neighbors found.")
		else:
			st.error(f"Failed to fetch neighbors: {resp.text}")

st.header("4. Latest Evaluation Results")
if st.button("Show Latest Eval Results"):
	with st.spinner("Fetching evaluation results..."):
		resp = requests.get(f"{API_URL}/eval/latest")
		if resp.ok:
			data = resp.json()
			st.write(f"**File:** {data.get('filename', 'N/A')}")
			st.json(data.get("results", {}))
		else:
			st.error(f"Failed to fetch eval results: {resp.text}")


