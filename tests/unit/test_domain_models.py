"""Domain-model invariants: no result may claim a security verdict it did not earn."""

from datetime import UTC, datetime
from typing import get_args

import pytest
from pydantic import ValidationError

from llm_guard_bench.domain.models import EvalResult, EvaluationStatus, SessionSummary

SECURITY_VERDICTS = ("PASSED", "VULNERABLE", "AMBIGUOUS")


def _summary() -> SessionSummary:
    return SessionSummary(
        session_id="test-session",
        started_at=datetime.now(UTC),
        config_snapshot={},
    )


def test_ambiguous_is_a_valid_status() -> None:
    """A judge that says 'unclear' must be recordable as AMBIGUOUS, not forced into PASSED."""
    result = EvalResult(status="AMBIGUOUS", stage="STAGE_2_JUDGE")
    assert result.status == "AMBIGUOUS"


@pytest.mark.parametrize("status", SECURITY_VERDICTS)
def test_pre_flight_cannot_carry_a_security_verdict(status: str) -> None:
    """PRE_FLIGHT means evaluation never started, so it cannot produce PASSED or VULNERABLE."""
    with pytest.raises(ValidationError):
        EvalResult(status=status, stage="PRE_FLIGHT")


def test_session_summary_counts_ambiguous_separately() -> None:
    summary = _summary()
    summary.increment("AMBIGUOUS")
    assert summary.ambiguous_count == 1
    assert summary.passed_count == 0


def test_every_status_is_counted_in_exactly_one_bucket() -> None:
    """Guard: adding a status without a matching counter must fail loudly."""
    counter_fields = [name for name in SessionSummary.model_fields if name.endswith("_count")]
    for status in get_args(EvaluationStatus):
        summary = _summary()
        summary.increment(status)
        assert summary.total_tests == 1
        assert sum(getattr(summary, name) for name in counter_fields) == 1, status


def test_unknown_status_is_rejected_not_silently_counted() -> None:
    summary = _summary()
    with pytest.raises(ValueError):
        summary.increment("BOGUS")
    assert summary.total_tests == 0
