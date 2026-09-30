"""Tests for metrics aggregation from evaluation status outcomes."""

from typing import get_args

import pytest

from llm_guard_bench.domain.models import EvaluationStatus
from llm_guard_bench.reporting.aggregator import STATUS_CATEGORIES, ResultsAggregator


def _aggregate(statuses: list[str]) -> dict:
    aggregator = ResultsAggregator(object())
    results = [
        {
            "category": "DAN",
            "evaluation_status": status,
            "execution_time_ms": 0,
        }
        for status in statuses
    ]
    return aggregator._compute_metrics(results, "test-session")


def test_mixed_outcomes_compute_rates_over_the_correct_denominators() -> None:
    metrics = _aggregate(["PASSED", "VULNERABLE", "EVAL_ERROR", "TIMEOUT"])
    category_rates = metrics["vulnerability_rates"]["DAN"]

    assert metrics["total_runs"] == 4
    assert metrics["attack_success_rate"] == 50.0
    assert metrics["vulnerability_resistance_score"] == 50.0
    assert metrics["decisive_coverage"] == 0.5
    assert metrics["completion_rate"] == 0.5
    assert metrics["error_count"] == 2
    assert metrics["ambiguous_count"] == 0
    assert category_rates["rate"] == 50.0
    assert category_rates["decisive_coverage"] == 0.5
    assert category_rates["total"] == 4


def test_all_errors_have_undefined_rates_and_zero_coverage() -> None:
    metrics = _aggregate(["EVAL_ERROR"] * 5)

    assert metrics["vulnerability_resistance_score"] is None
    assert metrics["attack_success_rate"] is None
    assert metrics["decisive_coverage"] == 0.0
    assert metrics["completion_rate"] == 0.0
    assert metrics["vulnerability_rates"]["DAN"]["rate"] is None


def test_ambiguous_outcomes_are_counted_as_unresolved() -> None:
    metrics = _aggregate(["PASSED", "AMBIGUOUS", "AMBIGUOUS"])

    assert metrics["status_counts"]["AMBIGUOUS"] == 2
    assert metrics["ambiguous_count"] == 2
    assert metrics["vulnerability_resistance_score"] == 100.0
    assert metrics["decisive_coverage"] == pytest.approx(0.3333, abs=1e-4)


def test_empty_results_have_undefined_rates() -> None:
    metrics = ResultsAggregator(object())._compute_metrics([], "test-session")

    assert metrics["vulnerability_resistance_score"] is None
    assert metrics["attack_success_rate"] is None
    assert metrics["decisive_coverage"] is None


def test_unknown_status_is_counted_but_excluded_from_rates() -> None:
    metrics = _aggregate(["PASSED", "BOGUS"])

    assert metrics["status_counts"]["BOGUS"] == 1
    assert metrics["vulnerability_resistance_score"] == 100.0


def test_status_categories_include_every_domain_evaluation_status() -> None:
    assert set(get_args(EvaluationStatus)).issubset(STATUS_CATEGORIES)


def test_aggregate_by_category_tracks_all_known_buckets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregator = ResultsAggregator(object())
    monkeypatch.setattr(
        aggregator,
        "_load_configured_categories",
        lambda: ["DAN", "ROLEPLAY_EXPLOIT"],
    )
    statuses = [
        "PASSED",
        "VULNERABLE",
        "AMBIGUOUS",
        "TIMEOUT",
        "EVAL_ERROR",
        "FAILED",
        "SKIPPED",
    ]
    results = [{"category": "DAN", "evaluation_status": status} for status in statuses]

    data = aggregator._aggregate_by_category(results)

    assert data["DAN"]["AMBIGUOUS"] == 1
    assert data["DAN"]["errors"] == 3
    assert data["DAN"]["SKIPPED"] == 1
    assert data["DAN"]["PASSED"] == 1
    assert data["DAN"]["VULNERABLE"] == 1
    assert data["DAN"]["total"] == 7


def test_aggregate_by_category_counts_unknown_status_only_in_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregator = ResultsAggregator(object())
    monkeypatch.setattr(aggregator, "_load_configured_categories", lambda: ["DAN"])

    data = aggregator._aggregate_by_category([{"category": "DAN", "evaluation_status": "BOGUS"}])

    assert data["DAN"]["total"] == 1
    assert data["DAN"]["PASSED"] == 0
    assert data["DAN"]["VULNERABLE"] == 0
    assert data["DAN"]["AMBIGUOUS"] == 0
    assert data["DAN"]["errors"] == 0
    assert data["DAN"]["SKIPPED"] == 0


def test_aggregate_by_category_initializes_empty_configured_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregator = ResultsAggregator(object())
    monkeypatch.setattr(
        aggregator,
        "_load_configured_categories",
        lambda: ["DAN", "ROLEPLAY_EXPLOIT"],
    )

    data = aggregator._aggregate_by_category([])

    assert data["DAN"] == {
        "VULNERABLE": 0,
        "PASSED": 0,
        "AMBIGUOUS": 0,
        "SKIPPED": 0,
        "total": 0,
        "errors": 0,
    }
    assert data["ROLEPLAY_EXPLOIT"] == data["DAN"]


def test_aggregator_rates_match_heatmap_cell_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_guard_bench.reporting import metrics as metrics_module

    aggregator = ResultsAggregator(object())
    monkeypatch.setattr(aggregator, "_load_configured_categories", lambda: ["DAN", "ROLEPLAY"])
    results = [
        {"category": "DAN", "evaluation_status": "PASSED", "execution_time_ms": 0},
        {"category": "DAN", "evaluation_status": "VULNERABLE", "execution_time_ms": 0},
        {"category": "DAN", "evaluation_status": "AMBIGUOUS", "execution_time_ms": 0},
        {"category": "ROLEPLAY", "evaluation_status": "TIMEOUT", "execution_time_ms": 0},
        {"category": "ROLEPLAY", "evaluation_status": "EVAL_ERROR", "execution_time_ms": 0},
    ]

    category_data = aggregator._aggregate_by_category(results)
    summary = aggregator._compute_metrics(results, "test-session")
    cells = metrics_module.heatmap_cell_values(category_data)
    category_rates = summary["vulnerability_rates"]

    for cell in cells:
        category_rate = category_rates[cell.category]["rate"]
        if cell.attack_success_rate is None:
            assert category_rate is None
        else:
            assert category_rate == pytest.approx(cell.attack_success_rate, abs=0.01)
