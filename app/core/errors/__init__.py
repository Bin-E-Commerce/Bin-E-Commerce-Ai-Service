"""Public boundary cho exception nghiệp vụ và hạ tầng của AI Service."""

from app.core.errors.exceptions import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    BackgroundConfigurationError,
    ConfigurationError,
    IdempotencyKeyReusedError,
    InvalidInputError,
    InvalidProviderResponseError,
    OptimizationJobNotReadyError,
    ProviderUnavailableError,
    RateLimitExceededError,
    ResourceNotFoundError,
    UpstreamConflictError,
    UpstreamRequestError,
)

__all__ = [
    "AuthenticationError",
    "AppError",
    "AuthorizationError",
    "BackgroundConfigurationError",
    "ConfigurationError",
    "IdempotencyKeyReusedError",
    "InvalidInputError",
    "InvalidProviderResponseError",
    "OptimizationJobNotReadyError",
    "ProviderUnavailableError",
    "RateLimitExceededError",
    "ResourceNotFoundError",
    "UpstreamConflictError",
    "UpstreamRequestError",
]
