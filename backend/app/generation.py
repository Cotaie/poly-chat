from app.models import GenerationOptions, GenerationResponse, ModelInfo
from app.adapters.ngram import NGramAdapterError, generate_ngram_response


SUPPORTED_ARCHITECTURES = {"ngram"}


class GenerationError(RuntimeError):
    pass


def generate_response(
    model: ModelInfo,
    prompt: str,
    options: GenerationOptions,
) -> GenerationResponse:
    architecture = model.architecture.lower()
    if architecture == "ngram":
        try:
            return generate_ngram_response(model, prompt, options)
        except NGramAdapterError as exc:
            raise GenerationError(str(exc)) from exc

    raise GenerationError(f"No generation support for architecture '{model.architecture}'.")
