# SOP Guided Conversational Agent

## Setup

```bash
poetry install
cp .env.example .env
```

Start the FastAPI backend and the Streamlit UI (two terminals):

```bash
poetry run uvicorn app.api:app --reload --port 8000
poetry run streamlit run ui/app.py
```

- Backend: http://localhost:8000  
- UI: http://localhost:8501  

Paste your Groq API key in the Streamlit sidebar (required). Create one at [console.groq.com/keys](https://console.groq.com/keys).

## Cloud demo (Render)

This repo includes `render.yaml` for a free Render web service (API + UI together).

1. Push to GitHub.
2. In [Render](https://dashboard.render.com), **New → Blueprint**, connect this repo, and apply.
3. Open the service URL Render assigns (ends in `.onrender.com`).
