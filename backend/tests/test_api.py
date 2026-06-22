import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.routes import generate, get_model, health, list_models
from app.models import GenerationOptions, GenerationRequest
from app.registry import ModelRegistry


def make_registry(tmp_path: Path) -> ModelRegistry:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    (registry_dir / "tiny-shakespeare-ngram.json").write_text(
        json.dumps(
            {
                "id": "tiny-shakespeare-ngram",
                "name": "Tiny Shakespeare n-gram",
                "architecture": "ngram",
                "dataset": "tiny-shakespeare",
                "artifact_path": "models/artifacts/tiny-shakespeare-ngram/model.json",
                "tokenizer": "word-level",
                "parameters": 0,
                "context_window": 128,
                "max_output_tokens": 80,
                "supported_options": ["max_tokens", "context_window", "top_k", "seed"],
                "notes": "Test registry entry.",
            }
        ),
        encoding="utf-8",
    )
    return ModelRegistry(registry_dir)


def test_health() -> None:
    assert health() == {"status": "ok"}


def test_list_models_includes_registry_entry(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    models = list_models(registry)

    assert [model.id for model in models] == ["tiny-shakespeare-ngram"]


def test_get_model(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    model = get_model("tiny-shakespeare-ngram", registry)

    assert model.architecture == "ngram"


def test_get_missing_model_returns_404(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    with pytest.raises(HTTPException) as exc:
        get_model("unknown", registry)

    assert exc.value.status_code == 404


def test_generate_uses_placeholder_response(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    response = generate(
        GenerationRequest(
            model_id="tiny-shakespeare-ngram",
            prompt="To be or not to be",
            options=GenerationOptions(max_tokens=12, context_window=64),
        ),
        registry,
    )

    assert response.model_id == "tiny-shakespeare-ngram"
    assert response.metadata == {"mode": "placeholder"}
    assert response.usage["input_tokens"] == 6


def test_generate_rejects_options_outside_model_limits(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    with pytest.raises(HTTPException) as exc:
        generate(
            GenerationRequest(
                model_id="tiny-shakespeare-ngram",
                prompt="Hello",
                options=GenerationOptions(context_window=999),
            ),
            registry,
        )

    assert exc.value.status_code == 400
