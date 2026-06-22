from time import perf_counter

from app.models import GenerationOptions, GenerationResponse, ModelInfo


SUPPORTED_ARCHITECTURES = {"ngram", "rnn", "gru", "lstm", "transformer"}


def generate_placeholder_response(
    model: ModelInfo,
    prompt: str,
    options: GenerationOptions,
) -> GenerationResponse:
    started = perf_counter()
    max_tokens = options.max_tokens or min(32, model.max_output_tokens)
    output = (
        f"[placeholder {model.architecture}] "
        f"{prompt.strip()} "
        f"(max_tokens={max_tokens})"
    )
    generation_ms = round((perf_counter() - started) * 1000)

    return GenerationResponse(
        model_id=model.id,
        output=output,
        usage={
            "input_tokens": len(prompt.split()),
            "output_tokens": len(output.split()),
        },
        timing={"load_ms": 0, "generation_ms": generation_ms},
        metadata={"mode": "placeholder"},
    )
