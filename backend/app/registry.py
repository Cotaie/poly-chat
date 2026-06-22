import json
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from app.models import ModelInfo


DEFAULT_REGISTRY_DIR = Path(__file__).resolve().parents[2] / "models" / "registry"


class RegistryError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, registry_dir: Path = DEFAULT_REGISTRY_DIR) -> None:
        self.registry_dir = registry_dir

    def list_models(self) -> list[ModelInfo]:
        if not self.registry_dir.exists():
            return []

        models = [self._load_file(path) for path in sorted(self.registry_dir.glob("*.json"))]
        return sorted(models, key=lambda model: model.name.lower())

    def get_model(self, model_id: str) -> ModelInfo | None:
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def _load_file(self, path: Path) -> ModelInfo:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ModelInfo.model_validate(data)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"Invalid JSON in model registry file {path}.") from exc
        except ValidationError as exc:
            raise RegistryError(f"Invalid model metadata in {path}: {exc}") from exc


@lru_cache
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
