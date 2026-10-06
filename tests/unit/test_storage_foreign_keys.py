"""Foreign-key enforcement tests for the DatabaseManager writer connection."""

import sqlite3
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
    monkeypatch.setattr(db_module, "RESULTS_DIR", tmp_path)
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


def _make_result(
    *,
    session_id: str = SESSION_ID,
    attack_id: str = ATTACK_ID,
) -> ResultRecord:
    return ResultRecord(
        session_id=session_id,
        timestamp=datetime.now(UTC),
        model_name="test-model",
        attack_id=attack_id,
        category="DAN",
        adversarial_prompt="A harmless test prompt.",
        system_prompt=None,
        evaluation_status="PASSED",
        evaluation_stage="STAGE_1_KEYWORD",
    )


async def _initialize_database(manager: DatabaseManager) -> None:
    await manager.connect()
    await manager.initialize()


async def _add_session(manager: DatabaseManager) -> None:
    await manager.upsert_session(
        SessionSummary(
            session_id=SESSION_ID,
            started_at=datetime.now(UTC),
            config_snapshot={},
        )
    )


async def _add_parents(manager: DatabaseManager) -> None:
    await _add_session(manager)
    await manager.upsert_attack_definition(_make_attack())


def _read_result_count(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        return connection.execute("SELECT COUNT(*) FROM test_results").fetchone()[0]


async def test_writer_connection_has_foreign_keys_enabled(
    database_manager: DatabaseManager,
) -> None:
    await _initialize_database(database_manager)
    assert database_manager._connection is not None

    cursor = await database_manager._connection.execute("PRAGMA foreign_keys")
    row = await cursor.fetchone()

    assert row == (1,)


async def test_insert_result_rejects_missing_session_and_attack_parents(
    database_manager: DatabaseManager,
) -> None:
    await _initialize_database(database_manager)

    with pytest.raises(sqlite3.IntegrityError):
        await database_manager.insert_result(_make_result())


async def test_insert_result_rejects_missing_attack_parent(
    database_manager: DatabaseManager,
) -> None:
    await _initialize_database(database_manager)
    await _add_session(database_manager)

    with pytest.raises(sqlite3.IntegrityError):
        await database_manager.insert_result(_make_result())


async def test_insert_result_succeeds_when_both_parents_exist(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await _initialize_database(database_manager)
    await _add_parents(database_manager)

    await database_manager.insert_result(_make_result())

    assert _read_result_count(tmp_path / "test.db") == 1


async def test_insert_succeeds_after_recovering_from_orphan_insert(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await _initialize_database(database_manager)

    with pytest.raises(sqlite3.IntegrityError):
        await database_manager.insert_result(_make_result())

    await _add_parents(database_manager)
    await database_manager.insert_result(_make_result())

    assert _read_result_count(tmp_path / "test.db") == 1
