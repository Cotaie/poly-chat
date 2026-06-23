from fastapi import APIRouter, Depends, HTTPException

from app.generation import GenerationError, SUPPORTED_ARCHITECTURES, generate_response
from app.models import (
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    NGramTrainingRequest,
    NGramTrainingResponse,
)
from app.ngram_training import NGramTrainingError, train_and_save_ngram_model
from app.registry import ModelRegistry, get_model_registry

router = APIRouter()


@router.get("/health", response_model=dict[str, str])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/models", response_model=list[ModelInfo])
def list_models(
    registry: ModelRegistry = Depends(get_model_registry),
) -> list[ModelInfo]:
    return registry.list_models()


@router.get("/api/models/{model_id}", response_model=ModelInfo)
def get_model(
    model_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
) -> ModelInfo:
    model = registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' was not found.")
    return model


@router.post("/api/generate", response_model=GenerationResponse)
def generate(
    request: GenerationRequest,
    registry: ModelRegistry = Depends(get_model_registry),
) -> GenerationResponse:
    model = registry.get_model(request.model_id)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_id}' was not found.",
        )

    validation_error = model.validate_options(request.options)
    if validation_error is not None:
        raise HTTPException(status_code=400, detail=validation_error)

    if model.architecture.lower() not in SUPPORTED_ARCHITECTURES:
        raise HTTPException(
            status_code=501,
            detail=f"No generation support for architecture '{model.architecture}'.",
        )

    try:
        return generate_response(model, request.prompt, request.options)
    except GenerationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/ngram/train", response_model=NGramTrainingResponse)
def train_ngram(
    request: NGramTrainingRequest,
) -> NGramTrainingResponse:
    try:
        model, stats, artifact_path, registry_path = train_and_save_ngram_model(request)
    except NGramTrainingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return NGramTrainingResponse(
        model=model,
        artifact_path=str(artifact_path),
        registry_path=str(registry_path),
        stats=stats,
    )
