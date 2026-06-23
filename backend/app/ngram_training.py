import json
from pathlib import Path

from app.adapters.ngram_transition_model import (
    BOS_TOKEN,
    EOS_TOKEN,
    SPECIAL_TOKENS,
    tokenize,
)
from app.models import ModelInfo, NGramTrainingRequest
from app.registry import DEFAULT_REGISTRY_DIR


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"


class NGramTrainingError(RuntimeError):
    pass


def train_and_save_ngram_model(
    request: NGramTrainingRequest,
    registry_dir: Path = DEFAULT_REGISTRY_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> tuple[ModelInfo, dict[str, int], Path, Path]:
    sentences = tokenize_corpus(request.corpus_text)
    if not sentences:
        raise NGramTrainingError("Corpus must contain at least one token.")

    artifact = build_ngram_artifact(sentences, request.order)
    artifact_dir = artifacts_dir / request.model_id
    artifact_path = artifact_dir / "model.json"
    registry_path = registry_dir / f"{request.model_id}.json"

    if not request.overwrite and (artifact_path.exists() or registry_path.exists()):
        raise NGramTrainingError(f"Model '{request.model_id}' already exists.")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    model_info = ModelInfo(
        id=request.model_id,
        name=request.name,
        architecture="ngram",
        dataset=request.dataset,
        artifact_path=repo_relative_path(artifact_path),
        tokenizer="word-level",
        parameters=count_transition_edges(artifact["transitions"]),
        context_window=request.context_window,
        max_output_tokens=request.max_output_tokens,
        supported_options=["max_tokens", "context_window", "top_k", "seed"],
        notes=request.notes
        or f"Word-level {request.order}-gram model trained from uploaded corpus text.",
    )

    registry_path.write_text(
        json.dumps(model_info.model_dump(), indent=2),
        encoding="utf-8",
    )

    stats = {
        "order": request.order,
        "tokens": count_visible_tokens(sentences),
        "vocabulary": count_visible_vocabulary(sentences),
        "contexts": len(artifact["transitions"]),
        "transition_edges": model_info.parameters,
    }

    return model_info, stats, artifact_path, registry_path


def tokenize_corpus(corpus_text: str) -> list[list[str]]:
    sentences: list[list[str]] = []
    for line in corpus_text.splitlines():
        tokens = [token for token in tokenize(line) if token not in SPECIAL_TOKENS]
        if tokens:
            sentences.append([BOS_TOKEN, *tokens, EOS_TOKEN])
    return sentences


def build_ngram_artifact(sentences: list[list[str]], order: int) -> dict[str, object]:
    context_size = max(order - 1, 0)
    transitions: dict[str, dict[str, int]] = {}
    fallback: dict[str, int] = {}

    for sentence_tokens in sentences:
        for index, token in enumerate(sentence_tokens):
            context_tokens = sentence_tokens[max(0, index - context_size) : index]
            context = " ".join(context_tokens)
            context_counts = transitions.setdefault(context, {})
            context_counts[token] = context_counts.get(token, 0) + 1
            if token not in SPECIAL_TOKENS:
                fallback[token] = fallback.get(token, 0) + 1

    return {
        "order": order,
        "transitions": sort_nested_counts(transitions),
        "fallback": sort_counts(fallback),
    }


def sort_nested_counts(counts_by_context: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        context: sort_counts(counts)
        for context, counts in sorted(counts_by_context.items(), key=lambda item: item[0])
    }


def sort_counts(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def count_transition_edges(transitions: object) -> int:
    if not isinstance(transitions, dict):
        return 0
    return sum(len(candidates) for candidates in transitions.values() if isinstance(candidates, dict))


def count_visible_tokens(sentences: list[list[str]]) -> int:
    return sum(1 for sentence in sentences for token in sentence if token not in SPECIAL_TOKENS)


def count_visible_vocabulary(sentences: list[list[str]]) -> int:
    return len({token for sentence in sentences for token in sentence if token not in SPECIAL_TOKENS})


def repo_relative_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()
