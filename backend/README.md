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

## N-gram artifacts

The n-gram adapter reads local JSON artifacts referenced by model metadata. The
demo artifact at `models/artifacts/tiny-shakespeare-demo-ngram/model.json` uses:

- `order`: n-gram order, where `3` means trigram generation.
- `transitions`: context strings mapped to next-token counts.
- `fallback`: optional token counts used when no context match is found.

Other model families should be added as separate adapters behind the same
`POST /api/generate` contract.
