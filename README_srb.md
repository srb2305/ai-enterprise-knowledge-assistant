# Enterprise Knowledge Assistant with Hybrid GraphRAG

See GraphRAG_Project_Blueprint.txt for full project plan.

docker compose up -d

venv311\Scripts\activate

pip install -r requirements.txt

python -m tests.test_phase2

############################### DATABASE#################
PGSQL RUN QUERY
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d pgvector
password: postgres

check table : \dt

CREATE EXTENSION IF NOT EXISTS vector;

#DROP TABLE IF EXISTS documents;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    chunk TEXT,
    metadata JSONB,
    embedding VECTOR(384)
);
############################### DATABASE END#################

########## RUN BACKEND ############
uvicorn api.app:app --reload
http://127.0.0.1:8000/docs
############# RUN ON STREAMLIT ################
inside venv > streamlit run ui/streamlit_app.py
http://localhost:8501/
############ Download dataset from HuggingFace###########
python scripts/download_hf_data.py

#############################################