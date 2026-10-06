"""Exit-code behavior tests for the CLI orchestrator."""

from collections.abc import Sequence

import pytest

from llm_guard_bench import cli
from llm_guard_bench.storage.migrations import MigrationError

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


def _make_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure_stage: str | None = None,
    failure: Exception | None = None,
    attacks: Sequence[object] = (),
) -> tuple[cli.LLMGuardBenchOrchestrator, list[str]]:
    events: list[str] = []
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="target-model",
        judge="judge-model",
        concurrency=1,
        auto_flush=False,
    )

    for stage in STAGES:

        async def stub(*_args: object, _stage: str = stage) -> object | None:
            events.append(_stage)
            if _stage == failure_stage and failure is not None:
                raise failure
            if _stage == "load_attack_definitions":
                return list(attacks)
            return None

        monkeypatch.setattr(orchestrator, stage, stub)

    return orchestrator, events


async def test_database_migration_failure_exits_and_skips_later_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, events = _make_orchestrator(
        monkeypatch,
        failure_stage="initialize_database",
        failure=MigrationError("schema_migrations missing"),
    )

    with pytest.raises(SystemExit) as exc_info:
        await orchestrator.orchestrate()

    assert exc_info.value.code == 1
    assert "load_attack_definitions" not in events
    assert "initialize_adapters" not in events
    assert "run_benchmark" not in events
    assert "cleanup" in events


async def test_environment_failure_exits_before_database_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, events = _make_orchestrator(
        monkeypatch,
        failure_stage="load_environment",
        failure=RuntimeError("environment unavailable"),
    )

    with pytest.raises(SystemExit) as exc_info:
        await orchestrator.orchestrate()

    assert exc_info.value.code == 1
    assert "initialize_database" not in events
    assert "cleanup" in events


async def test_missing_attack_definitions_exits_before_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, events = _make_orchestrator(
        monkeypatch,
        failure_stage="load_attack_definitions",
        failure=FileNotFoundError("prompts.json missing"),
    )

    with pytest.raises(SystemExit) as exc_info:
        await orchestrator.orchestrate()

    assert exc_info.value.code == 1
    assert "run_benchmark" not in events


async def test_empty_attack_definitions_exits_before_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, events = _make_orchestrator(monkeypatch, attacks=())

    with pytest.raises(SystemExit) as exc_info:
        await orchestrator.orchestrate()

    assert exc_info.value.code == 1
    assert "run_benchmark" not in events


async def test_adapter_initialization_failure_exits_before_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, events = _make_orchestrator(
        monkeypatch,
        attacks=(object(),),
        failure_stage="initialize_adapters",
        failure=RuntimeError("adapter initialization failed"),
    )

    with pytest.raises(SystemExit) as exc_info:
        await orchestrator.orchestrate()

    assert exc_info.value.code == 1
    assert "run_benchmark" not in events


async def test_benchmark_failure_exits_after_aggregating_partial_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, events = _make_orchestrator(
        monkeypatch,
        attacks=(object(),),
        failure_stage="run_benchmark",
        failure=RuntimeError("benchmark failed"),
    )

    with pytest.raises(SystemExit) as exc_info:
        await orchestrator.orchestrate()

    assert exc_info.value.code == 1
    assert "aggregate_and_export_results" in events
    assert "cleanup" in events


async def test_happy_path_runs_stages_in_order_and_reports_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    orchestrator, events = _make_orchestrator(monkeypatch, attacks=(object(),))

    await orchestrator.orchestrate()

    assert events == [
        "load_environment",
        "initialize_database",
        "load_attack_definitions",
        "register_session",
        "initialize_adapters",
        "run_benchmark",
        "aggregate_and_export_results",
        "finalize_session",
        "cleanup",
    ]
    assert "finalized successfully" in capsys.readouterr().out


async def test_benchmark_failure_does_not_report_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    orchestrator, _events = _make_orchestrator(
        monkeypatch,
        attacks=(object(),),
        failure_stage="run_benchmark",
        failure=RuntimeError("benchmark failed"),
    )

    with pytest.raises(SystemExit) as exc_info:
        await orchestrator.orchestrate()

    assert exc_info.value.code == 1
    assert "finalized successfully" not in capsys.readouterr().out
