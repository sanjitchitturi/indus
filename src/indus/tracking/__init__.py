"""MLflow experiment tracking integration."""


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "MLflowLogger":
        from indus.tracking.mlflow_logger import MLflowLogger

        return MLflowLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MLflowLogger"]
