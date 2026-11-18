"""Domain exceptions used by the serving layer."""


class RetainAIError(Exception):
    """Base class for application-specific errors."""


class ArtifactError(RetainAIError):
    """Raised when model artifacts cannot be loaded or validated."""


class ModelUnavailableError(RetainAIError):
    """Raised when inference is requested before a model is ready."""


class InvalidFeatureError(RetainAIError, ValueError):
    """Raised when raw inference features violate the model contract."""
