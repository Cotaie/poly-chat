import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.routes import generate, get_model, health, list_models, train_ngram
from app.models import GenerationOptions, GenerationRequest, ModelInfo, NGramTrainingRequest
from app.ngram_training import build_ngram_artifact, tokenize_corpus, train_and_save_ngram_model
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


def test_build_ngram_artifact_counts_transitions() -> None:
    sentences = tokenize_corpus("to be or not to be")
    artifact = build_ngram_artifact(sentences, order=3)

    assert artifact["order"] == 3
    assert artifact["transitions"][""] == {"<BOS>": 1}
    assert artifact["transitions"]["<BOS> to"] == {"be": 1}
    assert artifact["transitions"]["to be"] == {"<EOS>": 1, "or": 1}
    assert artifact["transitions"]["not to"] == {"be": 1}
    assert artifact["transitions"]["to be"]["<EOS>"] == 1
    assert artifact["fallback"]["to"] == 2
    assert "<BOS>" not in artifact["fallback"]
    assert "<EOS>" not in artifact["fallback"]


def test_tokenize_corpus_adds_special_tokens_without_leaking_manual_tokens() -> None:
    assert tokenize_corpus("<BOS> hello model <EOS>") == [["<BOS>", "hello", "model", "<EOS>"]]


def test_train_and_save_ngram_model_creates_registry_and_artifact(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    artifacts_dir = tmp_path / "artifacts"
    request = NGramTrainingRequest(
        model_id="custom-ngram",
        name="Custom n-gram",
        dataset="uploaded-corpus",
        corpus_text="to be or not to be",
        order=3,
        context_window=64,
        max_output_tokens=24,
    )

    model, stats, artifact_path, registry_path = train_and_save_ngram_model(
        request,
        registry_dir=registry_dir,
        artifacts_dir=artifacts_dir,
    )

    assert model.id == "custom-ngram"
    assert model.artifact_path.endswith("custom-ngram/model.json")
    assert stats["tokens"] == 6
    assert stats["contexts"] > 0
    assert artifact_path.exists()
    assert registry_path.exists()


def test_generation_stops_at_eos_without_returning_special_tokens(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    artifact_dir = tmp_path / "artifacts" / "sentence-ngram"
    registry_dir.mkdir()
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "model.json"
    artifact_path.write_text(
        json.dumps(
            build_ngram_artifact(
                tokenize_corpus("The capital of France is Paris."),
                order=4,
            )
        ),
        encoding="utf-8",
    )
    (registry_dir / "sentence-ngram.json").write_text(
        json.dumps(
            {
                "id": "sentence-ngram",
                "name": "Sentence n-gram",
                "architecture": "ngram",
                "dataset": "test",
                "artifact_path": str(artifact_path),
                "tokenizer": "word-level",
                "parameters": 0,
                "context_window": 64,
                "max_output_tokens": 20,
                "supported_options": ["max_tokens", "context_window", "top_k", "seed"],
                "notes": "Test registry entry.",
            }
        ),
        encoding="utf-8",
    )

    response = generate(
        GenerationRequest(
            model_id="sentence-ngram",
            prompt="The capital of France",
            options=GenerationOptions(max_tokens=20, top_k=1),
        ),
        ModelRegistry(registry_dir),
    )

    assert response.output == "The capital of France is Paris."
    assert "<EOS>" not in response.output
    assert response.usage["output_tokens"] == 2


def test_train_ngram_route_returns_saved_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = ModelInfo(
        id="route-ngram",
        name="Route n-gram",
        architecture="ngram",
        dataset="route-corpus",
        artifact_path="models/artifacts/route-ngram/model.json",
        tokenizer="word-level",
        parameters=3,
        context_window=128,
        max_output_tokens=80,
        supported_options=["max_tokens", "context_window", "top_k", "seed"],
        notes="Test model.",
    )

    def fake_train_and_save_ngram_model(request: NGramTrainingRequest):
        return model, {"tokens": 6}, tmp_path / "model.json", tmp_path / "route-ngram.json"

    monkeypatch.setattr(
        "app.api.routes.train_and_save_ngram_model",
        fake_train_and_save_ngram_model,
    )

    response = train_ngram(
        NGramTrainingRequest(
            model_id="route-ngram",
            name="Route n-gram",
            dataset="route-corpus",
            corpus_text="hello local model hello local chat",
            order=2,
            overwrite=True,
        )
    )

    assert response.model.id == "route-ngram"
    assert response.stats == {"tokens": 6}
