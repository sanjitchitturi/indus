"""Custom exceptions for the Indus project."""


class IndusError(Exception):
    """Base exception for all Indus errors."""


class DataError(IndusError):
    """Raised when data loading or preprocessing fails."""


class ModelError(IndusError):
    """Raised when model loading or configuration is invalid."""


class ConfigError(IndusError):
    """Raised when configuration is missing or inconsistent."""


class TrainingError(IndusError):
    """Raised when a training step fails irrecoverably."""


class EvaluationError(IndusError):
    """Raised when metric computation or evaluation fails."""
