"""Tests for injecting DatabaseManager's results directory."""

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_guard_bench.domain.models import AttackDefinition, SessionSummary
from llm_guard_bench.domain.models import (
    TestResult as ResultRecord,
)
from llm_guard_bench.storage import db as db_module
from llm_guard_bench.storage.db import DatabaseManager

ATTACK_ID = "test-attack"
SESSION_ID = "test-session"


@pytest.fixture
async def database_manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[DatabaseManager]:
    sentinel = tmp_path / "global_sentinel"
    monkeypatch.setattr(db_module, "RESULTS_DIR", sentinel, raising=False)
    manager = DatabaseManager(tmp_path / "test.db")
    try:
        yield manager
    finally:
        await manager.disconnect()


def _make_attack() -> AttackDefinition:
    return AttackDefinition(
        attack_id=ATTACK_ID,
        attack_name="Test attack",
        description="A test attack.",
        category="DAN",
        severity="LOW",
        tags=["test"],
        turns=["A harmless test prompt."],
    )


def _make_result() -> ResultRecord:
    return ResultRecord(
        session_id=SESSION_ID,
        timestamp=datetime.now(UTC),
        model_name="test-model",
        attack_id=ATTACK_ID,
        category="DAN",
        adversarial_prompt="A harmless test prompt.",
        system_prompt=None,
        evaluation_status="PASSED",
        evaluation_stage="STAGE_1_KEYWORD",
    )


async def _add_parents(manager: DatabaseManager) -> None:
    await manager.upsert_session(
        SessionSummary(
            session_id=SESSION_ID,
            started_at=datetime.now(UTC),
            config_snapshot={},
        )
    )
    await manager.upsert_attack_definition(_make_attack())


async def test_constructor_does_not_create_database_or_global_results_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = tmp_path / "global_sentinel"
    monkeypatch.setattr(db_module, "RESULTS_DIR", sentinel, raising=False)

    DatabaseManager(tmp_path / "newdir" / "t.db")

    assert not (tmp_path / "newdir").exists()
    assert not sentinel.exists()


async def test_results_directory_defaults_to_database_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        db_module,
        "RESULTS_DIR",
        tmp_path / "global_sentinel",
        raising=False,
    )
    manager = DatabaseManager(tmp_path / "t.db")

    assert manager.results_dir == tmp_path


async def test_results_directory_can_be_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        db_module,
        "RESULTS_DIR",
        tmp_path / "global_sentinel",
        raising=False,
    )
    output_dir = tmp_path / "out"
    manager = DatabaseManager(tmp_path / "t.db", results_dir=output_dir)

    assert manager.results_dir == output_dir


async def test_connect_creates_database_and_results_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        db_module,
        "RESULTS_DIR",
        tmp_path / "global_sentinel",
        raising=False,
    )
    database_dir = tmp_path / "db"
    results_dir = tmp_path / "out"
    manager = DatabaseManager(database_dir / "t.db", results_dir=results_dir)
    try:
        await manager.connect()

        assert database_dir.is_dir()
        assert results_dir.is_dir()
    finally:
        await manager.disconnect()


async def test_insert_result_writes_jsonl_to_injected_results_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = tmp_path / "global_sentinel"
    monkeypatch.setattr(db_module, "RESULTS_DIR", sentinel, raising=False)
    database_dir = tmp_path / "db"
    results_dir = tmp_path / "out"
    manager = DatabaseManager(database_dir / "t.db", results_dir=results_dir)
    try:
        await manager.connect()
        await manager.initialize()
        await _add_parents(manager)
        await manager.insert_result(_make_result())

        result_path = results_dir / f"session_{SESSION_ID}.jsonl"
        lines = [line for line in result_path.read_text(encoding="utf-8").splitlines() if line]

        assert len(lines) == 1
        assert isinstance(json.loads(lines[0]), dict)
        assert not list(database_dir.glob("*.jsonl"))
        assert not sentinel.exists()
    finally:
        await manager.disconnect()
