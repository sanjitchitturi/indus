"""Training loop, callbacks, and gradient tracking."""


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "IndusTrainer":
        from indus.training.trainer import IndusTrainer

        return IndusTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["IndusTrainer"]
