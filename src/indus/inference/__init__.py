"""Inference optimization and serving utilities."""


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "InferenceEngine":
        from indus.inference.engine import InferenceEngine

        return InferenceEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["InferenceEngine"]
