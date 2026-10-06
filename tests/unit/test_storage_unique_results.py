"""Tests for migration 002 unique result constraints."""

import importlib.resources
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
from llm_guard_bench.storage.migrations import discover_migrations

SESSION_ID = "test-session"
OTHER_SESSION_ID = "other-session"
ATTACK_ID = "test-attack"


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
    model_name: str = "test-model",
) -> ResultRecord:
    return ResultRecord(
        session_id=session_id,
        timestamp=datetime.now(UTC),
        model_name=model_name,
        attack_id=ATTACK_ID,
        category="DAN",
        adversarial_prompt="A harmless test prompt.",
        system_prompt=None,
        evaluation_status="PASSED",
        evaluation_stage="STAGE_1_KEYWORD",
    )


async def _initialize_database(manager: DatabaseManager) -> None:
    await manager.connect()
    await manager.initialize()


async def _add_session(manager: DatabaseManager, session_id: str = SESSION_ID) -> None:
    await manager.upsert_session(
        SessionSummary(
            session_id=session_id,
            started_at=datetime.now(UTC),
            config_snapshot={},
        )
    )


async def _add_parents(manager: DatabaseManager) -> None:
    await _add_session(manager)
    await manager.upsert_attack_definition(_make_attack())


def _read_migrations(database_path: Path) -> list[tuple[int, str]]:
    with sqlite3.connect(database_path) as connection:
        return connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()


def _read_result_count(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        return connection.execute("SELECT COUNT(*) FROM test_results").fetchone()[0]


async def test_initialize_applies_initial_and_unique_results_migrations(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await _initialize_database(database_manager)

    assert _read_migrations(tmp_path / "test.db") == [
        (1, "initial_schema"),
        (2, "unique_results"),
    ]


async def test_duplicate_result_for_same_session_attack_and_model_is_rejected(
    database_manager: DatabaseManager,
) -> None:
    await _initialize_database(database_manager)
    await _add_parents(database_manager)
    await database_manager.insert_result(_make_result())

    with pytest.raises(sqlite3.IntegrityError):
        await database_manager.insert_result(_make_result())


async def test_distinct_session_attack_model_combinations_are_accepted(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await _initialize_database(database_manager)
    await _add_parents(database_manager)
    await _add_session(database_manager, OTHER_SESSION_ID)

    await database_manager.insert_result(_make_result(model_name="model-one"))
    await database_manager.insert_result(_make_result(model_name="model-two"))
    await database_manager.insert_result(
        _make_result(session_id=OTHER_SESSION_ID, model_name="model-one")
    )

    assert _read_result_count(tmp_path / "test.db") == 3


async def test_result_insert_recovers_after_rejected_duplicate(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await _initialize_database(database_manager)
    await _add_parents(database_manager)
    await database_manager.insert_result(_make_result())

    with pytest.raises(sqlite3.IntegrityError):
        await database_manager.insert_result(_make_result())

    assert _read_result_count(tmp_path / "test.db") == 1
    await database_manager.insert_result(_make_result(model_name="different-model"))
    assert _read_result_count(tmp_path / "test.db") == 2


async def test_initialize_twice_keeps_exact_migration_rows(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await _initialize_database(database_manager)
    expected = [(1, "initial_schema"), (2, "unique_results")]
    assert _read_migrations(tmp_path / "test.db") == expected

    await database_manager.initialize()

    assert _read_migrations(tmp_path / "test.db") == expected


def test_discovered_migrations_include_unique_results() -> None:
    migrations = discover_migrations(db_module._MIGRATION_PATH.parent)

    assert [(migration.version, migration.name) for migration in migrations] == [
        (1, "initial_schema"),
        (2, "unique_results"),
    ]


def test_unique_results_migration_is_in_package_resources() -> None:
    migration = (
        importlib.resources.files("llm_guard_bench") / "storage" / "sql" / "002_unique_results.sql"
    )

    assert migration.is_file()


def test_unique_results_migration_has_no_transaction_control_statements() -> None:
    migration = (
        importlib.resources.files("llm_guard_bench") / "storage" / "sql" / "002_unique_results.sql"
    )
    assert migration.is_file()
    sql = migration.read_text(encoding="utf-8").upper()

    assert "PRAGMA" not in sql
    assert "BEGIN" not in sql
    assert "COMMIT" not in sql
