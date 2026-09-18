"""Framework integrations for the local Speko Gateway."""

from .client import CanonicalEvent, GatewayClient, GatewayError, GatewaySession
from .relay import (
    EvaluationAnswer,
    EvaluationQuestion,
    EvaluationResponse,
    RelayEvaluationClient,
    RelayLLMClient,
)

__all__ = [
    "CanonicalEvent",
    "EvaluationAnswer",
    "EvaluationQuestion",
    "EvaluationResponse",
    "GatewayClient",
    "GatewayError",
    "GatewaySession",
    "RelayEvaluationClient",
    "RelayLLMClient",
]
