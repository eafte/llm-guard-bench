"""Tests that CLI runs without a verdict are not reported as clean."""

from collections.abc import Sequence

import pytest

from llm_guard_bench import cli
from llm_guard_bench.pipelines.pipeline import BenchmarkSummary


def _make_summary(
    attacks_pulled: int,
    results_written: int,
    errors: int,
    status_counts: dict[str, int],
) -> BenchmarkSummary:
    return BenchmarkSummary(
        attacks_pulled=attacks_pulled,
        results_written=results_written,
        errors=errors,
        status_counts=status_counts,
    )


def _make_orchestrator() -> cli.LLMGuardBenchOrchestrator:
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="t",
        judge="j",
        concurrency=1,
    )
    orchestrator.target_adapter = object()
    orchestrator.judge_adapter = object()
    orchestrator.db_manager = object()
    return orchestrator


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    summary: BenchmarkSummary,
) -> None:
    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run_benchmark_summary(self, **_kwargs: object) -> BenchmarkSummary:
            return summary

    monkeypatch.setattr(cli, "BenchmarkPipeline", FakePipeline)


async def test_run_benchmark_reports_no_verdict_and_returns_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = _make_summary(3, 3, 0, {"EVAL_ERROR": 3})
    _install_fake_pipeline(monkeypatch, summary)
    orchestrator = _make_orchestrator()

    result = await orchestrator.run_benchmark([])

    assert result is summary
    assert "No attack reached a verdict" in capsys.readouterr().out


@pytest.mark.parametrize(
    "summary",
    [
        _make_summary(3, 3, 0, {"PASSED": 2, "EVAL_ERROR": 1}),
        _make_summary(0, 0, 0, {}),
    ],
)
async def test_run_benchmark_does_not_report_no_verdict_when_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    summary: BenchmarkSummary,
) -> None:
    _install_fake_pipeline(monkeypatch, summary)
    orchestrator = _make_orchestrator()

    result = await orchestrator.run_benchmark([])

    assert result is summary
    assert "No attack reached a verdict" not in capsys.readouterr().out


STAGES = (
    "load_environment",
    "run_auto_flush",
    "initialize_database",
    "load_attack_definitions",
    "register_session",
    "initialize_adapters",
    "run_benchmark",
    "aggregate_and_export_results",
    "finalize_session",
    "cleanup",
)


def _stub_orchestrator_stages(
    monkeypatch: pytest.MonkeyPatch,
    summary: BenchmarkSummary,
    attacks: Sequence[object] = (object(),),
) -> cli.LLMGuardBenchOrchestrator:
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="target-model",
        judge="judge-model",
        concurrency=1,
        auto_flush=False,
    )
    for stage in STAGES:

        async def stub(*_args: object, _stage: str = stage) -> object | None:
            if _stage == "load_attack_definitions":
                return list(attacks)
            if _stage == "run_benchmark":
                return summary
            return None

        monkeypatch.setattr(orchestrator, stage, stub)
    return orchestrator


@pytest.mark.parametrize(
    ("summary", "expected_exit_code"),
    [
        (_make_summary(3, 3, 0, {"EVAL_ERROR": 3}), 1),
        (_make_summary(1, 1, 0, {"PASSED": 1}), 0),
    ],
)
async def test_orchestrate_exit_code_reflects_verdict_coverage(
    monkeypatch: pytest.MonkeyPatch,
    summary: BenchmarkSummary,
    expected_exit_code: int,
) -> None:
    orchestrator = _stub_orchestrator_stages(monkeypatch, summary)

    if expected_exit_code:
        with pytest.raises(SystemExit) as exc_info:
            await orchestrator.orchestrate()

        assert exc_info.value.code == expected_exit_code
    else:
        await orchestrator.orchestrate()
