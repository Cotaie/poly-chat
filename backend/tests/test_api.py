import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.routes import generate, get_model, health, list_models
from app.models import GenerationOptions, GenerationRequest
from app.registry import ModelRegistry


def make_registry(tmp_path: Path) -> ModelRegistry:
    registry_dir = tmp_path / "registry"
    artifact_dir = tmp_path / "artifacts" / "tiny-shakespeare-demo-ngram"
    registry_dir.mkdir()
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "model.json"
    artifact_path.write_text(
        json.dumps(
            {
                "order": 3,
                "transitions": {
                    "": {"To": 1},
                    "To": {"be": 1},
                    "To be": {"or": 5, "and": 1},
                    "be or": {"not": 1},
                    "or not": {"to": 1},
                    "not to": {"be": 1},
                },
                "fallback": {"To": 1},
            }
        ),
        encoding="utf-8",
    )
    (registry_dir / "tiny-shakespeare-demo-ngram.json").write_text(
        json.dumps(
            {
                "id": "tiny-shakespeare-demo-ngram",
                "name": "Tiny Shakespeare Demo n-gram",
                "architecture": "ngram",
                "dataset": "tiny-shakespeare",
                "artifact_path": str(artifact_path),
                "tokenizer": "word-level",
                "parameters": 0,
                "context_window": 128,
                "max_output_tokens": 80,
                "supported_options": ["max_tokens", "context_window", "top_k", "seed"],
                "notes": "Hand-authored demo artifact, not a trained model.",
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

    assert [model.id for model in models] == ["tiny-shakespeare-demo-ngram"]


def test_get_model(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    model = get_model("tiny-shakespeare-demo-ngram", registry)

    assert model.architecture == "ngram"


def test_get_missing_model_returns_404(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    with pytest.raises(HTTPException) as exc:
        get_model("unknown", registry)

    assert exc.value.status_code == 404


def test_generate_uses_ngram_adapter(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    response = generate(
        GenerationRequest(
            model_id="tiny-shakespeare-demo-ngram",
            prompt="To be",
            options=GenerationOptions(max_tokens=3, context_window=64, top_k=1),
        ),
        registry,
    )

    assert response.model_id == "tiny-shakespeare-demo-ngram"
    assert response.output == "To be or not to"
    assert response.metadata["mode"] == "ngram"
    assert response.metadata["order"] == 3
    assert response.usage["input_tokens"] == 2
    assert response.usage["output_tokens"] == 3


def test_default_ngram_registry_entry_generates() -> None:
    registry = ModelRegistry()

    response = generate(
        GenerationRequest(
            model_id="tiny-shakespeare-demo-ngram",
            prompt="To be",
            options=GenerationOptions(max_tokens=3, top_k=1, seed=42),
        ),
        registry,
    )

    assert response.output == "To be or not to"
    assert response.metadata["mode"] == "ngram"


def test_generate_rejects_options_outside_model_limits(tmp_path: Path) -> None:
    registry = make_registry(tmp_path)

    with pytest.raises(HTTPException) as exc:
        generate(
            GenerationRequest(
                model_id="tiny-shakespeare-demo-ngram",
                prompt="Hello",
                options=GenerationOptions(context_window=999),
            ),
            registry,
        )

    assert exc.value.status_code == 400
