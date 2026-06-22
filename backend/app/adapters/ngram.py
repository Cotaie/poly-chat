import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any

from app.models import GenerationOptions, GenerationResponse, ModelInfo


REPO_ROOT = Path(__file__).resolve().parents[3]


class NGramAdapterError(RuntimeError):
    pass


def generate_ngram_response(
    model: ModelInfo,
    prompt: str,
    options: GenerationOptions,
) -> GenerationResponse:
    load_started = perf_counter()
    artifact = load_artifact(model.artifact_path)
    load_ms = round((perf_counter() - load_started) * 1000)

    generation_started = perf_counter()
    input_tokens = prompt.strip().split()
    generated_tokens = generate_tokens(
        artifact=artifact,
        prompt_tokens=input_tokens,
        max_tokens=options.max_tokens or min(32, model.max_output_tokens),
        context_window=options.context_window or model.context_window,
        top_k=options.top_k,
        seed=options.seed,
    )
    generation_ms = round((perf_counter() - generation_started) * 1000)

    return GenerationResponse(
        model_id=model.id,
        output=" ".join([*input_tokens, *generated_tokens]),
        usage={
            "input_tokens": len(input_tokens),
            "output_tokens": len(generated_tokens),
        },
        timing={"load_ms": load_ms, "generation_ms": generation_ms},
        metadata={
            "mode": "ngram",
            "order": int(artifact.get("order", 1)),
            "artifact_path": model.artifact_path,
        },
    )


def load_artifact(artifact_path: str) -> dict[str, Any]:
    path = Path(artifact_path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise NGramAdapterError(f"Model artifact was not found: {artifact_path}.") from exc
    except json.JSONDecodeError as exc:
        raise NGramAdapterError(f"Model artifact is not valid JSON: {artifact_path}.") from exc

    if not isinstance(artifact, dict):
        raise NGramAdapterError("N-gram artifact must be a JSON object.")
    if not isinstance(artifact.get("transitions"), dict):
        raise NGramAdapterError("N-gram artifact must include transitions.")

    return artifact


def generate_tokens(
    artifact: dict[str, Any],
    prompt_tokens: list[str],
    max_tokens: int,
    context_window: int,
    top_k: int | None,
    seed: int | None,
) -> list[str]:
    order = int(artifact.get("order", 1))
    transitions = artifact["transitions"]
    fallback = artifact.get("fallback", {})
    rng = random.Random(seed)
    tokens = prompt_tokens[-context_window:]
    generated: list[str] = []

    for _ in range(max_tokens):
        context_size = max(order - 1, 0)
        context = " ".join(tokens[-context_size:]) if context_size else ""
        candidates = transitions.get(context) or fallback
        if not isinstance(candidates, dict) or not candidates:
            break

        next_token = choose_next_token(candidates, top_k, rng)
        generated.append(next_token)
        tokens.append(next_token)

    return generated


def choose_next_token(
    candidates: dict[str, int],
    top_k: int | None,
    rng: random.Random,
) -> str:
    sorted_candidates = sorted(
        ((token, int(count)) for token, count in candidates.items() if int(count) > 0),
        key=lambda item: (-item[1], item[0]),
    )
    if top_k is not None:
        sorted_candidates = sorted_candidates[:top_k]
    if not sorted_candidates:
        raise NGramAdapterError("N-gram candidates must include at least one positive count.")

    total = sum(count for _, count in sorted_candidates)
    pick = rng.uniform(0, total)
    cumulative = 0.0

    for token, count in sorted_candidates:
        cumulative += count
        if pick <= cumulative:
            return token

    return sorted_candidates[-1][0]
