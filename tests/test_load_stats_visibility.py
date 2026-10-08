"""Tests that JSONL rejection stats are surfaced during streamed runs."""

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

from llm_guard_bench import cli
from llm_guard_bench.domain.models import AttackDefinition, SessionSummary
from llm_guard_bench.streaming_io.loader import JsonlAttackSource

_REJECTION_LABELS = {
    "invalid_json",
    "invalid_utf8",
    "invalid_attack",
    "not_an_object",
    "line_too_long",
}


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


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


class _FakeDB:
    """Minimal temporary database stub for orchestrator registration."""

    async def upsert_session(self, _summary: SessionSummary) -> None:
        pass

    async def upsert_attack_definition(self, _attack: AttackDefinition) -> None:
        pass


class _FakeAdapter:
    """Adapter stub; tests never make model calls."""


async def _run_streamed_source(
    source: JsonlAttackSource,
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="target-model",
        judge="judge-model",
        concurrency=1,
    )
    monkeypatch.setattr(orchestrator, "db_manager", _FakeDB())

    async def no_op() -> None:
        return None

    async def load_source() -> JsonlAttackSource:
        return source

    async def initialize_adapters() -> None:
        orchestrator.target_adapter = _FakeAdapter()
        orchestrator.judge_adapter = _FakeAdapter()

    benchmark_attack_ids: list[str] = []

    async def run_benchmark(attacks: Iterable[AttackDefinition]) -> list[object]:
        benchmark_attack_ids.extend(attack.attack_id for attack in attacks)
        return []

    monkeypatch.setattr(orchestrator, "load_environment", no_op)
    monkeypatch.setattr(orchestrator, "initialize_database", no_op)
    monkeypatch.setattr(orchestrator, "load_attack_definitions", load_source)
    monkeypatch.setattr(orchestrator, "initialize_adapters", initialize_adapters)
    monkeypatch.setattr(orchestrator, "run_benchmark", run_benchmark)
    monkeypatch.setattr(orchestrator, "aggregate_and_export_results", no_op)
    monkeypatch.setattr(orchestrator, "finalize_session", no_op)
    monkeypatch.setattr(orchestrator, "cleanup", no_op)

    await orchestrator.orchestrate()
    return benchmark_attack_ids


def _rejection_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and ("reject" in record.getMessage().lower() or "invalid" in record.getMessage().lower())
    ]


async def test_streamed_run_logs_rejected_lines_with_sanitized_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    file_path = tmp_path / "attacks.jsonl"
    _write_jsonl(
        file_path,
        [
            json.dumps(_attack("first")),
            "RAW_INVALID_JSON_SECRET {",
            json.dumps(_attack("second")),
            json.dumps("RAW_NON_OBJECT_SECRET"),
            json.dumps(_attack("third")),
        ],
    )
    source = JsonlAttackSource(str(file_path))

    await _run_streamed_source(source, monkeypatch)

    warnings = _rejection_warnings(caplog)
    assert len(warnings) == 1
    warning = warnings[0]
    assert re.search(r"(accepted\D{0,12}3|3\D{0,12}accepted)\b", warning, re.IGNORECASE)
    assert re.search(r"(rejected\D{0,12}2|2\D{0,12}rejected)\b", warning, re.IGNORECASE)
    reported_labels = {label for label in _REJECTION_LABELS if label in warning}
    assert reported_labels == {"invalid_json", "not_an_object"}
    assert "RAW_INVALID_JSON_SECRET" not in warning
    assert "RAW_NON_OBJECT_SECRET" not in warning
    assert source.last_stats is not None
    assert source.last_stats.accepted == 3
    assert source.last_stats.rejected_invalid == 2


async def test_streamed_run_does_not_warn_when_all_lines_are_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    file_path = tmp_path / "attacks.jsonl"
    _write_jsonl(file_path, [json.dumps(_attack("first")), json.dumps(_attack("second"))])
    source = JsonlAttackSource(str(file_path))

    await _run_streamed_source(source, monkeypatch)

    assert _rejection_warnings(caplog) == []


async def test_rejection_warning_is_logged_once_across_both_source_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    file_path = tmp_path / "attacks.jsonl"
    _write_jsonl(
        file_path,
        [json.dumps(_attack("first")), "RAW_INVALID_SECRET {", json.dumps(_attack("last"))],
    )
    source = JsonlAttackSource(str(file_path))

    benchmark_attack_ids = await _run_streamed_source(source, monkeypatch)

    assert source.count == 2
    assert benchmark_attack_ids == ["first", "last"]
    assert len(_rejection_warnings(caplog)) == 1
