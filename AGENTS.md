# AGENTS.md

## Project Context

Poly Chat is a local chatbot workbench for comparing small language models trained on different datasets.

The project uses:

- React, TypeScript, and Deno for the frontend.
- FastAPI and Python for the backend.
- Local model metadata and artifact files for trained models.

The goal is to keep the project easy to explain to interviewers while still being technically grounded.

## Working Agreements

- Keep documentation concise and readable.
- Prefer simple, explicit architecture over premature abstraction.
- Do not add SQL, hosted LLM providers, or API-key-based provider integrations unless the project direction changes.
- Treat `SPEC.md` as the source of truth for product scope.
- Keep frontend and backend concerns separated.
- Use local model files and metadata as the default model registry approach.

## Review Expectations

- Call out contradictions with `SPEC.md`.
- Prioritize user-facing clarity, correctness, and maintainability.
- For implementation work, include a focused verification step when practical.
