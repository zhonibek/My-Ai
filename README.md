# AETHER AI

A Python and React application exploring local AI inference, retrieval, streaming chat APIs, and model-provider integration.

**Stack:** FastAPI, Uvicorn, PyTorch, Hugging Face Transformers, React, and Vite.

## What is in the repository

- Local inference engines and optional external model-provider adapters.
- API routes for chat, files, history, models, training, voice, and vision.
- Retrieval components for parsing, chunking, embeddings, and vector storage.
- A React chat frontend.
- API, retrieval, multilingual, inference, and benchmark scripts.

These components are experimental. Their presence in the repository does not establish production readiness, independent model training, or benchmark superiority. The default local model configuration references **Qwen/Qwen2.5-0.5B-Instruct**; application code and pretrained model weights are distinct contributions.

## Development setup

Requirements: Python 3.10+ and Node.js with npm. Model inference may download weights and require additional memory or provider configuration.

```bash
git clone https://github.com/zhonibek/My-Ai.git
cd My-Ai/backend
python -m venv .venv
```

Activate the virtual environment (`.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on macOS/Linux), then:

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

In a second terminal, from `My-Ai/frontend`:

```bash
npm ci
npm run dev
```

Open the frontend URL printed by Vite. Backend API documentation is available at **http://127.0.0.1:8000/docs**.

## Configuration

Review `backend/app/config.py`. Options include `LOCAL_MODEL_NAME_OR_PATH`, `USE_CUDA`, `DEVICE`, `MAX_NEW_TOKENS`, and optional search-provider credentials. Keep real credentials outside version control. A tracked `.env` file exists in the current repository and should be checked before sharing the project with reviewers.

## Review starting points

| Area | Location |
| --- | --- |
| API entry point | `backend/app/main.py` |
| Local inference | `backend/app/inference/` |
| Retrieval | `backend/app/rag/` |
| Provider integration | `backend/app/providers/` |
| Frontend | `frontend/src/App.jsx` |
| Validation scripts | `backend/test_*.py`, `backend/benchmark_optimization.py` |

## Validation status

Setup commands are derived from the repository configuration and launcher. They have not been executed as part of this documentation review. API and model tests need their stated dependencies and, in some cases, a running server or model downloads. Publish reproducible commands and measured results before making performance claims.
