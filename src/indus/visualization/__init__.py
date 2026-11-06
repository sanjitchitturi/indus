"""Plotting and visualization for training analysis."""


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    from indus.visualization import plots as _p

    if hasattr(_p, name):
        return getattr(_p, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "plot_training_loss",
    "plot_gradient_norms",
    "plot_eval_comparison",
    "plot_adapter_convergence",
    "generate_all_plots",
]
