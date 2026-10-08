"""Tests for streaming registration of JSONL attack sources."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TypeAlias

import pytest

from llm_guard_bench import cli
from llm_guard_bench.domain.models import AttackDefinition, SessionSummary
from llm_guard_bench.storage.errors import StorageError
from llm_guard_bench.streaming_io.loader import JsonlAttackSource

Event: TypeAlias = str | tuple[str, object]


def _attack(attack_id: str) -> dict[str, object]:
    return {
        "attack_id": attack_id,
        "attack_name": f"Attack {attack_id}",
        "description": "A test attack.",
        "category": "DAN",
        "severity": "LOW",
        "tags": ["t"],
        "turns": ["hi"],
    }


def _write_jsonl(tmp_path: Path, lines: list[bytes]) -> Path:
    file_path = tmp_path / "attacks.jsonl"
    file_path.write_bytes(b"".join(lines))
    return file_path


def _json_line(attack_id: str) -> bytes:
    return (json.dumps(_attack(attack_id)) + "\n").encode("utf-8")


class FakeDB:
    """Recording DB stub that can fail on a selected attack upsert."""

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
    *,
    fail_on_attack_upsert: int | None = None,
) -> cli.LLMGuardBenchOrchestrator:
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="target-model",
        judge="judge-model",
        concurrency=1,
    )
    orchestrator.db_manager = FakeDB(
        events,
        fail_on_attack_upsert=fail_on_attack_upsert,
    )
    return orchestrator


async def test_streaming_registration_upserts_session_then_attacks_in_order(
    tmp_path: Path,
) -> None:
    events: list[Event] = []
    file_path = _write_jsonl(tmp_path, [_json_line("a"), _json_line("b"), _json_line("c")])
    source = JsonlAttackSource(str(file_path))
    orchestrator = _make_orchestrator(events)

    registered = await orchestrator.register_streaming_source(source)

    assert registered == 3
    assert events[0][0] == "upsert_session"
    assert events[1:] == [
        ("upsert_attack_definition", "a"),
        ("upsert_attack_definition", "b"),
        ("upsert_attack_definition", "c"),
    ]


async def test_streaming_registration_consumes_source_lazily_in_order(
    tmp_path: Path,
) -> None:
    events: list[Event] = []
    file_path = _write_jsonl(tmp_path, [_json_line("a"), _json_line("b")])
    source = JsonlAttackSource(str(file_path))

    def observed_source() -> Iterator[AttackDefinition]:
        for attack in source:
            events.append(f"yield:{attack.attack_id}")
            yield attack

    orchestrator = _make_orchestrator(events)

    registered = await orchestrator.register_streaming_source(observed_source())

    assert registered == 2
    assert [
        event for event in events if isinstance(event, str) or event[0] != "upsert_session"
    ] == [
        "yield:a",
        ("upsert_attack_definition", "a"),
        "yield:b",
        ("upsert_attack_definition", "b"),
    ]


async def test_streaming_registration_skips_invalid_lines(tmp_path: Path) -> None:
    events: list[Event] = []
    file_path = _write_jsonl(
        tmp_path,
        [_json_line("first"), b"{invalid json\n", _json_line("last")],
    )
    source = JsonlAttackSource(str(file_path))
    orchestrator = _make_orchestrator(events)

    registered = await orchestrator.register_streaming_source(source)

    assert registered == 2
    assert events[1:] == [
        ("upsert_attack_definition", "first"),
        ("upsert_attack_definition", "last"),
    ]


async def test_empty_source_registers_session_and_returns_zero(tmp_path: Path) -> None:
    events: list[Event] = []
    file_path = _write_jsonl(tmp_path, [])
    source = JsonlAttackSource(str(file_path))
    orchestrator = _make_orchestrator(events)

    registered = await orchestrator.register_streaming_source(source)

    assert registered == 0
    assert len(events) == 1
    assert events[0][0] == "upsert_session"


async def test_storage_error_closes_source_iterator() -> None:
    events: list[Event] = []
    source_closed = False
    attacks = [AttackDefinition(**_attack(attack_id)) for attack_id in ("a", "b", "c")]

    class CloseTrackingSource:
        def __iter__(self) -> Iterator[AttackDefinition]:
            nonlocal source_closed
            try:
                yield from attacks
            finally:
                source_closed = True

    orchestrator = _make_orchestrator(events, fail_on_attack_upsert=2)

    with pytest.raises(StorageError):
        await orchestrator.register_streaming_source(CloseTrackingSource())

    assert source_closed


async def test_streaming_registration_upserts_session_once_for_many_attacks(
    tmp_path: Path,
) -> None:
    events: list[Event] = []
    file_path = _write_jsonl(
        tmp_path,
        [_json_line(f"attack-{index}") for index in range(5)],
    )
    source = JsonlAttackSource(str(file_path))
    orchestrator = _make_orchestrator(events)

    registered = await orchestrator.register_streaming_source(source)

    assert registered == 5
    assert sum(isinstance(event, tuple) and event[0] == "upsert_session" for event in events) == 1
