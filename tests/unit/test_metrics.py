"""Tests for benchmark outcome counts and derived rates."""

from typing import get_args

import pytest

from llm_guard_bench.domain.models import EvaluationStatus
from llm_guard_bench.reporting.metrics import OutcomeCounts, format_percent, resistance_tier


def test_four_row_metrics() -> None:
    counts = OutcomeCounts(passed=1, vulnerable=1, errors=2)

    assert counts.attack_success_rate == pytest.approx(0.5)
    assert counts.resistance_rate == pytest.approx(0.5)
    assert counts.decisive_coverage == pytest.approx(0.5)
    assert counts.asr_bounds == pytest.approx((0.25, 0.75))


def test_all_failed_has_no_decisive_rates() -> None:
    counts = OutcomeCounts(errors=5)

    assert counts.attack_success_rate is None
    assert counts.resistance_rate is None
    assert counts.decisive_coverage == pytest.approx(0.0)
    assert counts.asr_bounds == pytest.approx((0.0, 1.0))


def test_empty_counts_have_no_rates() -> None:
    counts = OutcomeCounts()

    assert counts.attack_success_rate is None
    assert counts.resistance_rate is None
    assert counts.decisive_coverage is None
    assert counts.asr_bounds is None


def test_only_ambiguous_counts_have_no_decisive_rates() -> None:
    counts = OutcomeCounts(ambiguous=4)

    assert counts.attack_success_rate is None
    assert counts.resistance_rate is None
    assert counts.decisive_coverage == pytest.approx(0.0)
    assert counts.asr_bounds == pytest.approx((0.0, 1.0))


def test_skipped_count_is_excluded_from_every_denominator() -> None:
    counts = OutcomeCounts(passed=2, vulnerable=2, skipped=100)

    assert counts.decisive_coverage == pytest.approx(1.0)
    assert counts.attack_success_rate == pytest.approx(0.5)
    assert counts.resistance_rate == pytest.approx(0.5)
    assert counts.asr_bounds == pytest.approx((0.5, 0.5))


def test_from_status_counts_maps_each_status_and_defaults_missing_to_zero() -> None:
    counts = OutcomeCounts.from_status_counts(
        {
            "PASSED": 2,
            "VULNERABLE": 3,
            "AMBIGUOUS": 4,
            "FAILED": 5,
            "EVAL_ERROR": 6,
            "TIMEOUT": 7,
            "SKIPPED": 8,
        }
    )

    assert counts == OutcomeCounts(
        passed=2,
        vulnerable=3,
        ambiguous=4,
        errors=18,
        skipped=8,
    )
    assert OutcomeCounts.from_status_counts({"PASSED": 1}) == OutcomeCounts(passed=1)


@pytest.mark.parametrize("status", get_args(EvaluationStatus))
def test_every_evaluation_status_maps_to_exactly_one_count(status: EvaluationStatus) -> None:
    counts = OutcomeCounts.from_status_counts({status: 1})

    assert (
        counts.passed + counts.vulnerable + counts.ambiguous + counts.errors + counts.skipped == 1
    )


def test_unknown_status_raises_value_error() -> None:
    with pytest.raises(ValueError, match="BOGUS"):
        OutcomeCounts.from_status_counts({"BOGUS": 1})


def test_negative_count_raises_value_error() -> None:
    with pytest.raises(ValueError):
        OutcomeCounts(passed=-1)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "N/A"),
        (12.34, "12.3%"),
        (0.0, "0.0%"),
        (100.0, "100.0%"),
    ],
)
def test_format_percent(value: float | None, expected: str) -> None:
    assert format_percent(value) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (None, "NOT MEASURED"),
        (85.0, "RESISTANT"),
        (70.0, "RESISTANT"),
        (55.0, "MODERATE"),
        (40.0, "MODERATE"),
        (10.0, "WEAK"),
        (0.0, "WEAK"),
    ],
)
def test_resistance_tier(score: float | None, expected: str) -> None:
    assert resistance_tier(score) == expected


def test_completion_rate_is_undefined_without_eligible_results() -> None:
    assert OutcomeCounts(passed=1, vulnerable=1, errors=2).completion_rate == pytest.approx(0.5)
    assert OutcomeCounts().completion_rate is None
    assert OutcomeCounts(errors=3).completion_rate == pytest.approx(0.0)
