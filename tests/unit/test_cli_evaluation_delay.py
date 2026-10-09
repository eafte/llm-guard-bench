"""Tests for the CLI evaluation-delay option and pipeline wiring."""

import sys
from types import SimpleNamespace

import pytest

from llm_guard_bench import cli
from llm_guard_bench.domain.models import AttackDefinition
from llm_guard_bench.pipelines.pipeline import BenchmarkSummary


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


def test_parse_arguments_defaults_evaluation_delay_to_one_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["llm-guard-bench", "--target", "t"])

    args = cli.parse_arguments()

    assert args.evaluation_delay == 1.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", 0.0), ("0.5", 0.5)],
)
def test_parse_arguments_accepts_non_negative_evaluation_delay(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: float,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["llm-guard-bench", "--target", "t", "--evaluation-delay", value],
    )

    args = cli.parse_arguments()

    assert args.evaluation_delay == expected


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "abc"])
def test_parse_arguments_rejects_invalid_evaluation_delay(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["llm-guard-bench", "--target", "t", "--evaluation-delay", value],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.parse_arguments()

    assert exc_info.value.code == 2
    assert "argument --evaluation-delay" in capsys.readouterr().err


@pytest.mark.parametrize("configured_delay", [0.25, None])
async def test_run_benchmark_passes_evaluation_delay_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    configured_delay: float | None,
) -> None:
    pipeline_kwargs: list[dict[str, object]] = []

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            pipeline_kwargs.append(kwargs)

        async def run_benchmark_summary(self, **_kwargs: object) -> BenchmarkSummary:
            return BenchmarkSummary(0, 0, 0, {})

    monkeypatch.setattr(cli, "BenchmarkPipeline", FakePipeline)
    orchestrator_kwargs: dict[str, object] = {
        "target": "t",
        "judge": "j",
        "concurrency": 1,
    }
    if configured_delay is not None:
        orchestrator_kwargs["evaluation_delay_seconds"] = configured_delay

    orchestrator = cli.LLMGuardBenchOrchestrator(**orchestrator_kwargs)

    await orchestrator.run_benchmark([_attack()])

    expected_delay = 1.0 if configured_delay is None else configured_delay
    assert pipeline_kwargs[0]["evaluation_delay_seconds"] == expected_delay


async def test_async_main_passes_evaluation_delay_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "parse_arguments",
        lambda: SimpleNamespace(
            target="t",
            judge="j",
            concurrency=1,
            categories=None,
            auto_flush=False,
            attacks_file=None,
            evaluation_delay=0.25,
        ),
    )
    monkeypatch.setattr(cli, "validate_configuration", lambda: None)

    class FakeOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.append(kwargs)

        async def orchestrate(self) -> None:
            return None

    monkeypatch.setattr(cli, "LLMGuardBenchOrchestrator", FakeOrchestrator)

    await cli.async_main()

    assert captured_kwargs[0]["evaluation_delay_seconds"] == 0.25
