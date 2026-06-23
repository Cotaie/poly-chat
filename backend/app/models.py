from typing import Any

from pydantic import BaseModel, Field


class GenerationOptions(BaseModel):
    temperature: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=1)
    context_window: int | None = Field(default=None, ge=1)
    top_k: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0, le=1)
    seed: int | None = None


class ModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    dataset: str
    artifact_path: str
    tokenizer: str
    parameters: int = Field(ge=0)
    context_window: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    supported_options: list[str] = Field(default_factory=list)
    notes: str = ""

    def validate_options(self, options: GenerationOptions) -> str | None:
        provided_options = options.model_dump(exclude_none=True)
        provided_names = set(provided_options)
        supported_names = set(self.supported_options)

        unsupported = provided_names - supported_names
        if unsupported:
            unsupported_names = ", ".join(sorted(unsupported))
            return f"Unsupported option(s): {unsupported_names}."

        if options.context_window is not None and options.context_window > self.context_window:
            return f"context_window cannot exceed {self.context_window}."

        if options.max_tokens is not None and options.max_tokens > self.max_output_tokens:
            return f"max_tokens cannot exceed {self.max_output_tokens}."

        return None


class GenerationRequest(BaseModel):
    model_id: str
    prompt: str = Field(min_length=1)
    options: GenerationOptions = Field(default_factory=GenerationOptions)


class GenerationResponse(BaseModel):
    model_id: str
    output: str
    usage: dict[str, int]
    timing: dict[str, int]
    metadata: dict[str, Any] = Field(default_factory=dict)


class NGramTrainingRequest(BaseModel):
    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    corpus_text: str = Field(min_length=1)
    order: int = Field(ge=1, le=8)
    context_window: int = Field(default=128, ge=1)
    max_output_tokens: int = Field(default=80, ge=1)
    overwrite: bool = False
    notes: str = ""


class NGramTrainingResponse(BaseModel):
    model: ModelInfo
    artifact_path: str
    registry_path: str
    stats: dict[str, int]
