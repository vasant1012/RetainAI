"""Retention policy, deliberately separated from ML inference."""

from dataclasses import dataclass
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass(frozen=True)
class RetentionAssessment:
    risk_level: RiskLevel
    recommended_action: str


class RetentionService:
    """Translate a 0–100 churn score into a configurable business action."""

    def __init__(self, medium_threshold: float = 50.0, high_threshold: float = 80.0):
        if not 0 <= medium_threshold < high_threshold <= 100:
            raise ValueError("Thresholds must satisfy 0 <= medium < high <= 100.")
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold

    def assess(self, score: float) -> RetentionAssessment:
        if not 0 <= score <= 100:
            raise ValueError("Churn score must be between 0 and 100.")
        if score >= self.high_threshold:
            return RetentionAssessment(
                RiskLevel.HIGH,
                "Trigger priority win-back outreach and a 20% incentive.",
            )
        if score >= self.medium_threshold:
            return RetentionAssessment(
                RiskLevel.MEDIUM,
                "Send a personalized re-engagement campaign.",
            )
        return RetentionAssessment(
            RiskLevel.LOW,
            "Continue standard lifecycle engagement.",
        )


retention_service = RetentionService()
