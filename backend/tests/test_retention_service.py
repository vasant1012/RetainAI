import pytest

from backend.core.retention_service import RetentionService, RiskLevel


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0, RiskLevel.LOW), (49.99, RiskLevel.LOW), (50, RiskLevel.MEDIUM), (79.99, RiskLevel.MEDIUM), (80, RiskLevel.HIGH), (100, RiskLevel.HIGH)],
)
def test_risk_boundaries(score: float, expected: RiskLevel) -> None:
    assert RetentionService().assess(score).risk_level is expected


@pytest.mark.parametrize("score", [-0.01, 100.01])
def test_invalid_scores_are_rejected(score: float) -> None:
    with pytest.raises(ValueError):
        RetentionService().assess(score)
