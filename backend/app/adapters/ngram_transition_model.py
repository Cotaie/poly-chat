import json
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
Candidate = tuple[str, int]


class NGramModelError(RuntimeError):
    pass


class NGramTransitionModel:
    def __init__(self, artifact: dict[str, Any]) -> None:
        if not isinstance(artifact.get("transitions"), dict):
            raise NGramModelError("N-gram artifact must include transitions.")

        self.order = int(artifact.get("order", 1))
        self.transitions: dict[str, Any] = artifact["transitions"]
        self.fallback: Any = artifact.get("fallback", {})

    @classmethod
    def load(cls, artifact_path: str) -> "NGramTransitionModel":
        path = resolve_artifact_path(artifact_path)
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise NGramModelError(f"Model artifact was not found: {artifact_path}.") from exc
        except json.JSONDecodeError as exc:
            raise NGramModelError(f"Model artifact is not valid JSON: {artifact_path}.") from exc

        if not isinstance(artifact, dict):
            raise NGramModelError("N-gram artifact must be a JSON object.")

        return cls(artifact)

    def generate_tokens(
        self,
        prompt_tokens: list[str],
        max_tokens: int,
        context_window: int,
        top_k: int | None,
        seed: int | None,
    ) -> list[str]:
        rng = random.Random(seed)
        tokens = prompt_tokens[-context_window:]
        generated: list[str] = []

        for _ in range(max_tokens):
            candidates = self.candidates_for(tokens)
            if not isinstance(candidates, dict) or not candidates:
                break

            next_token = self.choose_next_token(candidates, top_k, rng)
            generated.append(next_token)
            tokens.append(next_token)

        return generated

    def candidates_for(self, tokens: list[str]) -> Any:
        context = self.build_context(tokens)
        return self.transitions.get(context) or self.fallback

    def build_context(self, tokens: list[str]) -> str:
        context_size = max(self.order - 1, 0)
        if context_size == 0:
            return ""
        return detokenize(tokens[-context_size:])

    def choose_next_token(
        self,
        candidates: dict[str, int],
        top_k: int | None,
        rng: random.Random,
    ) -> str:
        sorted_candidates = self.sort_candidates(candidates)
        if top_k is not None:
            sorted_candidates = sorted_candidates[:top_k]
        if not sorted_candidates:
            raise NGramModelError("N-gram candidates must include at least one positive count.")

        total = sum(count for _, count in sorted_candidates)
        pick = rng.uniform(0, total)
        return self.choose_by_weight(sorted_candidates, pick)

    @staticmethod
    def sort_candidates(candidates: dict[str, int]) -> list[Candidate]:
        return sorted(
            ((token, int(count)) for token, count in candidates.items() if int(count) > 0),
            key=lambda item: (-item[1], item[0]),
        )

    @staticmethod
    def choose_by_weight(candidates: list[Candidate], pick: float) -> str:
        cumulative = 0.0

        for token, count in candidates:
            cumulative += count
            if pick <= cumulative:
                return token

        return candidates[-1][0]


def resolve_artifact_path(artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def tokenize(text: str) -> list[str]:
    return text.strip().split()


def detokenize(tokens: list[str]) -> str:
    return " ".join(tokens)
