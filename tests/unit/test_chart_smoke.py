"""Smoke tests for chart rendering with undefined and measured metrics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest

from llm_guard_bench.reporting.aggregator import ResultsAggregator


@pytest.fixture(autouse=True)
def close_figures() -> None:
    yield
    plt.close("all")


async def _render_chart(
    statuses: list[tuple[str, str]],
    output_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregator = ResultsAggregator(object())
    results = [
        {
            "category": category,
            "evaluation_status": status,
            "execution_time_ms": 1000,
            "model_name": "smoke-model",
        }
        for status, category in statuses
    ]
    monkeypatch.setattr(aggregator, "_query_results_by_session", lambda _: _results(results))
    monkeypatch.setattr(aggregator, "_load_results_from_json", lambda _: [])
    metrics = aggregator._compute_metrics(results, "smoke-session")

    await aggregator.plot_vulnerability_chart(
        "smoke-session",
        str(output_path),
        metrics=metrics,
    )


async def _results(records: list[dict]) -> list[dict]:
    return records


@pytest.mark.parametrize(
    ("statuses", "filename"),
    [
        ([("EVAL_ERROR", "DAN")] * 5, "all-errors.png"),
        (
            [
                ("PASSED", "DAN"),
                ("VULNERABLE", "DAN"),
                ("AMBIGUOUS", "ROLEPLAY_EXPLOIT"),
                ("TIMEOUT", "ROLEPLAY_EXPLOIT"),
            ],
            "mixed.png",
        ),
        ([], "empty.png"),
    ],
)
async def test_chart_renders_without_error_for_all_metric_states(
    statuses: list[tuple[str, str]],
    filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / filename

    await _render_chart(statuses, output_path, monkeypatch)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
