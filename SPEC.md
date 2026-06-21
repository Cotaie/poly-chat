# Poly Chat Spec

## Purpose

Poly Chat is a local chatbot workbench for comparing small language models trained on different datasets. The goal is not to build a production chat product, but to make model behavior easy to inspect, compare, and discuss.

The project will start with manually trained models such as:

- n-gram
- RNN
- GRU
- LSTM
- transformer

Each model can be trained on different corpora, then loaded into the app for interactive testing.

## Tech Stack

- Frontend: React, TypeScript, Deno
- Backend: FastAPI, Python
- Model runtime: Python ML libraries such as PyTorch where needed
- Storage: local model files and metadata files

The frontend and backend live in the same repository.

## Core Idea

The user selects a trained model, enters a prompt, adjusts generation settings, and sees the model output. The same prompt can then be tested against another model to compare behavior, speed, and quality.

This makes the app useful for answering questions like:

- How does an n-gram model behave compared with an LSTM?
- How does changing the dataset affect generated text?
- How much does context window size matter?
- How do generation settings affect coherence, repetition, and creativity?
- Which model responds faster?

## User Interface

The first UI should be a practical workbench, not a landing page.

Main areas:

- Model list: available trained models
- Model details: architecture, dataset, tokenizer, parameter count, context window, notes
- Prompt area: text input for the user prompt
- Generation controls: settings that affect output
- Output area: generated response, timing, token counts, and errors
- Comparison area: recent outputs for comparing models side by side

Useful controls:

- Model selector
- Context window size
- Max output length
- Temperature
- Top-k
- Top-p
- Random seed, when supported
- Clear current session
- Re-run same prompt on another model

The context window should be user-adjustable, but only within the limits supported by the selected model.

## Model Metadata

Models are registered through metadata files so the frontend does not need hard-coded model information.

Example:

```json
{
  "id": "tiny-shakespeare-lstm",
  "name": "Tiny Shakespeare LSTM",
  "architecture": "lstm",
  "dataset": "tiny-shakespeare",
  "artifact_path": "models/artifacts/tiny-shakespeare-lstm/model.pt",
  "tokenizer": "char-level",
  "parameters": 1200000,
  "context_window": 256,
  "max_output_tokens": 200,
  "supported_options": ["temperature", "max_tokens", "context_window", "top_k", "seed"],
  "notes": "Character-level LSTM trained on Tiny Shakespeare."
}
```

## Backend Responsibilities

The FastAPI backend should:

- Load model metadata from local files
- Expose available models to the frontend
- Load model artifacts from disk
- Route generation requests to the correct model adapter
- Validate generation options
- Return generated text and basic performance metrics

Each model family should have an adapter with a common shape:

```text
load_model(model_info)
generate(model, prompt, options)
```

This keeps n-gram, RNN, GRU, LSTM, and transformer logic separate while giving the UI one consistent API.

## API Sketch

```http
GET /health
GET /api/models
GET /api/models/{model_id}
POST /api/generate
```

Example generation request:

```json
{
  "model_id": "tiny-shakespeare-lstm",
  "prompt": "To be or not to be",
  "options": {
    "temperature": 0.8,
    "max_tokens": 120,
    "context_window": 128,
    "top_k": 40,
    "seed": 42
  }
}
```

Example generation response:

```json
{
  "model_id": "tiny-shakespeare-lstm",
  "output": "To be or not to be...",
  "usage": {
    "input_tokens": 18,
    "output_tokens": 120
  },
  "timing": {
    "load_ms": 0,
    "generation_ms": 412
  }
}
```

## Repository Shape

```text
poly-chat/
  frontend/
  backend/
  models/
    registry/
    artifacts/
  experiments/
  SPEC.md
  README.md
```

- `frontend/`: React UI
- `backend/`: FastAPI app and inference adapters
- `models/registry/`: metadata files for trained models
- `models/artifacts/`: trained model files
- `experiments/`: training scripts, notebooks, and research notes

## Future Training Workflow

Training can start manually through scripts or notebooks. Later, the app can include a training workbench where the user can:

- Pick an architecture
- Choose a predefined corpus
- Upload a text corpus
- Configure training settings
- Start training from the browser
- Track progress, loss, speed, and estimated time
- Save the trained model artifact
- Compare the new model with previous models

Possible training settings:

- Tokenizer type
- Context length
- Embedding size
- Hidden size
- Number of layers
- Batch size
- Epochs
- Learning rate
- Train/validation split
- Random seed
- Device

Useful comparison metrics:

- Training duration
- Tokens or characters processed per second
- Final training loss
- Validation loss
- Model size
- Parameter count
- Inference latency
- Qualitative output quality

## Success Criteria

The project is successful when a user can:

- See available trained models
- Select a model and inspect its metadata
- Send a prompt
- Adjust generation settings
- View the generated output
- Compare outputs from different model architectures or datasets

The most important outcome is that the app helps explain how different model choices affect chatbot behavior.
