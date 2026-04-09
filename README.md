# Enterprise Knowledge Assistant with Hybrid GraphRAG — Step-by-Step Guide

This guide will walk you through setting up, running, and understanding the full workflow of the Hybrid GraphRAG project, from scratch to advanced usage. It is designed for beginners and includes all commands and explanations.

---

## 1. Project Overview

This project builds an Enterprise Knowledge Assistant using a hybrid Retrieval-Augmented Generation (RAG) approach. It combines vector search (pgvector) and a knowledge graph (Neo4j) to answer complex queries over large document collections, with a Streamlit UI and FastAPI backend.

---

## 2. Prerequisites

- **Python 3.11** (recommended)
- **Docker** (for running PostgreSQL/pgvector, Neo4j, Prometheus, Grafana)
- **Git** (to clone the repository)
- **pgAdmin** (optional, for database inspection)

---

## 3. Setup Instructions

### 3.1. Clone the Repository
```sh
git clone <your-repo-url>
cd <project-folder>
```

### 3.2. Create and Activate Virtual Environment
```sh
python -m venv venv311
venv311\Scripts\activate  # On Windows
# or
source venv311/bin/activate  # On Linux/Mac
```

### 3.3. Install Python Dependencies
```sh
pip install -r requirements.txt
```

### 3.4. Start Databases and Services with Docker
```sh
docker compose up -d
```
This will start PostgreSQL (with pgvector), Neo4j, and other services as defined in `docker/docker-compose.yml`.

### 3.5. Database Setup (Optional, for manual inspection)
- Open **pgAdmin** or use `psql`:
```sh
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d pgvector
# password: postgres
```
- Check tables:
```sql
\dt
```
- Create extension and table if needed:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    chunk TEXT,
    metadata JSONB,
    embedding VECTOR(384)
);
```

---

## 4. Running the Application

### 4.1. Run Backend (FastAPI)
```sh
uvicorn api.app:app --reload
```
- Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for API documentation and testing.

### 4.2. Run Frontend (Streamlit UI)
```sh
streamlit run ui/streamlit_app.py
```
- Visit [http://localhost:8501/](http://localhost:8501/) in your browser.

---

## 5. Ingesting Data

### 5.1. Download Sample Data
```sh
python scripts/download_hf_data.py
```
- This downloads 10 news articles into the `data/` folder.

### 5.2. Upload Documents
- Use the Streamlit UI to upload `.pdf`, `.txt`, `.docx`, `.csv`, `.xls`, or `.xlsx` files.
- The system will chunk, embed, and store them in the database.

---

## 6. Querying and Visualization

### 6.1. Ask Questions
- Use the "Ask a Question" section in Streamlit to query your ingested documents.

### 6.2. Graph Visualization
- Use the "Graph Neighbors Visualization" section to explore entity relationships.

### 6.3. Evaluation Results
- Use the "Latest Evaluation Results" button to view RAGAS/MLflow evaluation metrics.

---

## 7. Testing

### 7.1. Run Tests
```sh
python -m tests.test_phase1
python -m tests.test_phase2
python -m tests.test_self_correction_loop
python -m test_vector_retriever
```
- Additional test scripts are in the `tests/` folder.

---

## 8. Useful SQL Commands (for pgAdmin/psql)
- List tables:
  ```sql
  \dt
  ```
- View data:
  ```sql
  SELECT * FROM documents LIMIT 100;
  ```
- Drop table (if needed):
  ```sql
  DROP TABLE IF EXISTS documents;
  ```

---

## 9. Troubleshooting
- **Connection errors:** Ensure Docker containers and FastAPI are running.
- **File upload errors:** Check file type and extension; `.csv`/`.xls`/`.xlsx` are converted to text before ingestion.
- **Dependency errors:** Run `pip install -r requirements.txt` and ensure `python-multipart` is installed.
- **Database issues:** Check Docker logs and pgAdmin for errors.

---

## 10. Project Structure
- `api/` — FastAPI backend
- `ui/` — Streamlit frontend
- `ingestion/` — Document loaders and chunkers
- `graph/` — Entity/relation extraction and graph store
- `retrieval/` — Vector, graph, and hybrid retrievers
- `evaluation/` — RAGAS, MLflow, and evaluation scripts
- `data/` — Ingested documents
- `tests/` — Test scripts
- `scripts/` — Data download and utility scripts

---

## 11. References
- See `GraphRAG_Project_Blueprint.txt` for the full project plan and architecture.
- For more details, check the code comments and each module’s README (if available).

---

**You now have a complete, step-by-step guide to set up, run, and learn from this project!**
