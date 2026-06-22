from time import perf_counter

from app.adapters.ngram_transition_model import (
    NGramModelError,
    NGramTransitionModel,
    detokenize,
    tokenize,
)
from app.models import GenerationOptions, GenerationResponse, ModelInfo


NGramAdapterError = NGramModelError


def generate_ngram_response(
    model: ModelInfo,
    prompt: str,
    options: GenerationOptions,
) -> GenerationResponse:
    load_started = perf_counter()
    transition_model = NGramTransitionModel.load(model.artifact_path)
    load_ms = round((perf_counter() - load_started) * 1000)

    generation_started = perf_counter()
    input_tokens = tokenize(prompt)
    generated_tokens = transition_model.generate_tokens(
        prompt_tokens=input_tokens,
        max_tokens=options.max_tokens or min(32, model.max_output_tokens),
        context_window=options.context_window or model.context_window,
        top_k=options.top_k,
        seed=options.seed,
    )
    generation_ms = round((perf_counter() - generation_started) * 1000)

    return GenerationResponse(
        model_id=model.id,
        output=detokenize([*input_tokens, *generated_tokens]),
        usage={
            "input_tokens": len(input_tokens),
            "output_tokens": len(generated_tokens),
        },
        timing={"load_ms": load_ms, "generation_ms": generation_ms},
        metadata={
            "mode": "ngram",
            "order": transition_model.order,
            "artifact_path": model.artifact_path,
        },
    )
