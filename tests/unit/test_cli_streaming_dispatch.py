"""Tests for dispatching CLI runs between list and JSONL attack sources."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypeAlias

import pytest

from llm_guard_bench import cli
from llm_guard_bench.domain.models import AttackDefinition, SessionSummary
from llm_guard_bench.storage.errors import StorageError
from llm_guard_bench.streaming_io.loader import JsonlAttackSource

Event: TypeAlias = str | tuple[str, object]


def _attack(
    attack_id: str,
    *,
    category: str = "DAN",
) -> dict[str, object]:
    return {
        "attack_id": attack_id,
        "attack_name": f"Attack {attack_id}",
        "description": "A test attack.",
        "category": category,
        "severity": "LOW",
        "tags": ["t"],
        "turns": ["hi"],
    }


def _write_jsonl(tmp_path: Path, attacks: list[dict[str, object]]) -> Path:
    file_path = tmp_path / "attacks.jsonl"
    file_path.write_text(
        "".join(json.dumps(attack) + "\n" for attack in attacks),
        encoding="utf-8",
    )
    return file_path


def _write_json(tmp_path: Path, attacks: list[dict[str, object]]) -> Path:
    file_path = tmp_path / "attacks.json"
    file_path.write_text(json.dumps({"attacks": attacks}), encoding="utf-8")
    return file_path


class FakeDB:
    """Recording DB stub with optional failure on an attack upsert."""

    def __init__(
        self,
        events: list[Event],
        *,
        fail_on_attack_upsert: int | None = None,
    ) -> None:
        self.events = events
        self.fail_on_attack_upsert = fail_on_attack_upsert
        self.attack_upsert_count = 0

    async def upsert_session(self, summary: SessionSummary) -> None:
        self.events.append(("upsert_session", summary))

    async def upsert_attack_definition(self, attack: AttackDefinition) -> None:
        self.attack_upsert_count += 1
        self.events.append(("upsert_attack_definition", attack.attack_id))
        if self.attack_upsert_count == self.fail_on_attack_upsert:
            raise StorageError("disk full")


def _make_orchestrator(
    events: list[Event],
    monkeypatch: pytest.MonkeyPatch,
    attacks: list[AttackDefinition] | JsonlAttackSource,
    *,
    fail_on_attack_upsert: int | None = None,
    benchmark_failure: Exception | None = None,
) -> tuple[cli.LLMGuardBenchOrchestrator, list[object]]:
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="target-model",
        judge="judge-model",
        concurrency=1,
    )
    orchestrator.db_manager = FakeDB(events, fail_on_attack_upsert=fail_on_attack_upsert)
    benchmark_attacks: list[object] = []
    actual_register_streaming_source = orchestrator.register_streaming_source

    async def stage(
        name: str,
        result: object = None,
        *,
        failure: Exception | None = None,
    ) -> object:
        events.append(name)
        if failure is not None:
            raise failure
        return result

    async def load_environment() -> None:
        await stage("load_environment")

    async def initialize_database() -> None:
        await stage("initialize_database")

    async def load_attack_definitions() -> list[AttackDefinition] | JsonlAttackSource:
        events.append("load_attack_definitions")
        return attacks

    async def register_session(definitions: list[AttackDefinition]) -> None:
        events.append("register_session")
        await orchestrator._register_session_row()
        for attack in definitions:
            await orchestrator.db_manager.upsert_attack_definition(attack)

    async def register_streaming_source(
        source: Iterable[AttackDefinition],
    ) -> int:
        events.append("register_streaming_source")
        return await actual_register_streaming_source(source)

    async def initialize_adapters() -> None:
        await stage("initialize_adapters")

    async def run_benchmark(source: list[AttackDefinition] | JsonlAttackSource) -> list[object]:
        events.append("run_benchmark")
        benchmark_attacks.append(source)
        if benchmark_failure is not None:
            raise benchmark_failure
        return []

    async def aggregate_and_export_results() -> None:
        await stage("aggregate_and_export_results")

    async def finalize_session() -> None:
        await stage("finalize_session")

    async def cleanup() -> None:
        await stage("cleanup")

    monkeypatch.setattr(orchestrator, "load_environment", load_environment)
    monkeypatch.setattr(orchestrator, "initialize_database", initialize_database)
    monkeypatch.setattr(orchestrator, "load_attack_definitions", load_attack_definitions)
    monkeypatch.setattr(orchestrator, "register_session", register_session)
    monkeypatch.setattr(orchestrator, "register_streaming_source", register_streaming_source)
    monkeypatch.setattr(orchestrator, "initialize_adapters", initialize_adapters)
    monkeypatch.setattr(orchestrator, "run_benchmark", run_benchmark)
    monkeypatch.setattr(orchestrator, "aggregate_and_export_results", aggregate_and_export_results)
    monkeypatch.setattr(orchestrator, "finalize_session", finalize_session)
    monkeypatch.setattr(orchestrator, "cleanup", cleanup)
    return orchestrator, benchmark_attacks


@pytest.mark.parametrize("extension", [".jsonl", ".JSONL"])
async def test_load_attack_definitions_returns_lazy_jsonl_source(
    tmp_path: Path,
    extension: str,
) -> None:
    file_path = tmp_path / f"attacks{extension}"
    file_path.write_text(json.dumps(_attack("dan")) + "\n", encoding="utf-8")
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="t",
        judge="j",
        concurrency=1,
        categories=["DAN"],
        attacks_file=str(file_path),
    )

    source = await orchestrator.load_attack_definitions()

    assert isinstance(source, JsonlAttackSource)
    assert source._categories == ["DAN"]
    assert source.count is None

    unfiltered = cli.LLMGuardBenchOrchestrator(
        target="t",
        judge="j",
        concurrency=1,
        attacks_file=str(file_path),
    )
    unfiltered_source = await unfiltered.load_attack_definitions()

    assert isinstance(unfiltered_source, JsonlAttackSource)
    assert unfiltered_source._categories is None
    assert unfiltered_source.count is None


async def test_load_attack_definitions_keeps_json_list_path(tmp_path: Path) -> None:
    file_path = _write_json(tmp_path, [_attack("first"), _attack("second")])
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="t",
        judge="j",
        concurrency=1,
        attacks_file=str(file_path),
    )

    attacks = await orchestrator.load_attack_definitions()

    assert isinstance(attacks, list)
    assert [attack.attack_id for attack in attacks] == ["first", "second"]


async def test_orchestrate_registers_jsonl_before_adapters_and_passes_same_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = _write_jsonl(tmp_path, [_attack("a"), _attack("b"), _attack("c")])
    source = JsonlAttackSource(str(file_path))
    events: list[Event] = []
    orchestrator, benchmark_attacks = _make_orchestrator(events, monkeypatch, source)

    await orchestrator.orchestrate()

    assert events.index("register_streaming_source") < events.index("initialize_adapters")
    assert events.index("initialize_adapters") < events.index("run_benchmark")
    assert "register_session" not in events
    assert benchmark_attacks == [source]


async def test_orchestrate_exits_for_empty_jsonl_without_adapters_or_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = _write_jsonl(tmp_path, [])
    source = JsonlAttackSource(str(file_path))
    events: list[Event] = []
    orchestrator, _benchmark_attacks = _make_orchestrator(events, monkeypatch, source)

    with pytest.raises(SystemExit) as exc_info:
        await orchestrator.orchestrate()

    assert exc_info.value.code == 1
    assert "register_streaming_source" in events
    assert sum(isinstance(event, tuple) and event[0] == "upsert_session" for event in events) == 1
    assert "initialize_adapters" not in events
    assert "run_benchmark" not in events


async def test_orchestrate_exits_on_streaming_registration_storage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = _write_jsonl(tmp_path, [_attack("a"), _attack("b"), _attack("c")])
    source = JsonlAttackSource(str(file_path))
    events: list[Event] = []
    orchestrator, _benchmark_attacks = _make_orchestrator(
        events,
        monkeypatch,
        source,
        fail_on_attack_upsert=2,
    )

    with pytest.raises(SystemExit) as exc_info:
        await orchestrator.orchestrate()

    assert exc_info.value.code == 1
    assert "register_streaming_source" in events
    assert [
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "upsert_attack_definition"
    ] == [
        ("upsert_attack_definition", "a"),
        ("upsert_attack_definition", "b"),
    ]
    assert "run_benchmark" not in events


async def test_orchestrate_keeps_list_registration_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attacks = [AttackDefinition(**_attack("a"))]
    events: list[Event] = []
    orchestrator, _benchmark_attacks = _make_orchestrator(events, monkeypatch, attacks)

    await orchestrator.orchestrate()

    assert "register_session" in events
    assert "register_streaming_source" not in events


async def test_run_benchmark_reports_count_for_unsized_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    file_path = _write_jsonl(tmp_path, [_attack("a"), _attack("b"), _attack("c")])
    source = JsonlAttackSource(str(file_path))
    assert len(list(source)) == 3
    assert source.count == 3
    captured_attacks: list[object] = []

    class FakePipeline:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run_benchmark(self, **kwargs: object) -> list[object]:
            captured_attacks.append(kwargs["attacks"])
            return []

    monkeypatch.setattr(cli, "BenchmarkPipeline", FakePipeline)
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="t",
        judge="j",
        concurrency=1,
    )

    await orchestrator.run_benchmark(source)

    assert captured_attacks == [source]
    assert "Total Attack Vectors: 3" in capsys.readouterr().out
