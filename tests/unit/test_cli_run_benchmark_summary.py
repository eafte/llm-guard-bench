"""Tests for the CLI's bounded benchmark summary reporting."""

import pytest

from llm_guard_bench import cli
from llm_guard_bench.domain.models import AttackDefinition
from llm_guard_bench.pipelines.pipeline import BenchmarkSummary
from llm_guard_bench.storage.errors import StorageError


def _attack() -> AttackDefinition:
    return AttackDefinition(
        attack_id="attack-1",
        attack_name="Test attack",
        description="A harmless test attack.",
        category="DAN",
        severity="LOW",
        tags=["test"],
        turns=["A harmless test prompt."],
    )


def _make_summary() -> BenchmarkSummary:
    return BenchmarkSummary(
        attacks_pulled=5,
        results_written=4,
        errors=1,
        status_counts={"PASSED": 3, "VULNERABLE": 1},
    )


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure: Exception | None = None,
) -> tuple[BenchmarkSummary, list[dict[str, object]]]:
    summary = _make_summary()
    calls: list[dict[str, object]] = []

    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run_benchmark_summary(self, **kwargs: object) -> BenchmarkSummary:
            calls.append(kwargs)
            if failure is not None:
                raise failure
            return summary

    monkeypatch.setattr(cli, "BenchmarkPipeline", FakePipeline)
    return summary, calls


def _make_orchestrator() -> cli.LLMGuardBenchOrchestrator:
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="t",
        judge="j",
        concurrency=2,
        evaluation_delay_seconds=0.25,
    )
    orchestrator.target_adapter = object()
    orchestrator.judge_adapter = object()
    orchestrator.db_manager = object()
    return orchestrator


async def test_run_benchmark_returns_summary_and_forwards_pipeline_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, calls = _install_fake_pipeline(monkeypatch)
    orchestrator = _make_orchestrator()
    attacks = [_attack()]

    result = await orchestrator.run_benchmark(attacks)

    assert result is summary
    assert calls == [
        {
            "attacks": attacks,
            "model_name": "t",
            "concurrency_limit": 2,
            "session_id": orchestrator.session_id,
        }
    ]


async def test_run_benchmark_prints_bounded_summary_breakdown(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_pipeline(monkeypatch)
    orchestrator = _make_orchestrator()

    await orchestrator.run_benchmark([_attack()])

    output = capsys.readouterr().out
    assert "4 results written" in output
    assert "1 error" in output
    assert "PASSED: 3" in output
    assert "VULNERABLE: 1" in output
    assert output.index("PASSED: 3") < output.index("VULNERABLE: 1")
    assert "non-null" not in output


@pytest.mark.parametrize("failure", [StorageError("storage failed"), RuntimeError("failed")])
async def test_run_benchmark_propagates_summary_pipeline_exception(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    _install_fake_pipeline(monkeypatch, failure=failure)
    orchestrator = _make_orchestrator()

    with pytest.raises(type(failure)) as exc_info:
        await orchestrator.run_benchmark([_attack()])

    assert exc_info.value is failure
