"""Model loading, quantization, and adapter configuration."""


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in ("load_base_model", "load_tokenizer"):
        from indus.model.loader import load_base_model, load_tokenizer

        return {"load_base_model": load_base_model, "load_tokenizer": load_tokenizer}[name]
    if name in ("apply_lora_adapter", "count_parameters"):
        from indus.model.adapter import apply_lora_adapter, count_parameters

        return {"apply_lora_adapter": apply_lora_adapter, "count_parameters": count_parameters}[
            name
        ]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_base_model",
    "load_tokenizer",
    "apply_lora_adapter",
    "count_parameters",
]
