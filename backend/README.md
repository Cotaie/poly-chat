# Poly Chat Backend

FastAPI backend for the local Poly Chat workbench.

## Run locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

## API

- `GET /health`
- `GET /api/models`
- `GET /api/models/{model_id}`
- `POST /api/generate`

The backend currently includes a placeholder adapter so the API contract can be
developed before real local model inference is wired in.
