"""Tests for wiring versioned migrations into database initialization."""

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from llm_guard_bench.storage import db as db_module
from llm_guard_bench.storage.db import DatabaseManager
from llm_guard_bench.storage.migrations import MigrationError


@pytest.fixture
async def database_manager(
    tmp_path: Path,
) -> AsyncIterator[DatabaseManager]:
    manager = DatabaseManager(tmp_path / "test.db")
    try:
        yield manager
    finally:
        await manager.disconnect()


def _read_table_names(database_path: Path) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def _read_table_columns(database_path: Path) -> dict[str, list[str]]:
    table_names = _read_table_names(database_path)
    with sqlite3.connect(database_path) as connection:
        return {
            table: [
                row[1] for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            ]
            for table in table_names
        }


async def test_fresh_database_records_initial_schema_migration(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await database_manager.connect()
    await database_manager.initialize()

    with sqlite3.connect(tmp_path / "test.db") as connection:
        rows = connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()

    assert rows == [(1, "initial_schema"), (2, "unique_results")]


async def test_initialize_twice_records_initial_migration_only_once(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await database_manager.connect()
    await database_manager.initialize()
    await database_manager.initialize()

    with sqlite3.connect(tmp_path / "test.db") as connection:
        rows = connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()

    assert rows == [(1, "initial_schema"), (2, "unique_results")]


async def test_legacy_database_without_migration_tracking_is_preserved(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE test_results (id INTEGER);
            CREATE TABLE sessions (id INTEGER);
            CREATE TABLE attack_definitions (id INTEGER);
            """
        )
    original_columns = _read_table_columns(database_path)

    await database_manager.connect()
    with pytest.raises(MigrationError, match="schema_migrations"):
        await database_manager.initialize()

    assert database_path.is_file()
    assert "schema_migrations" not in _read_table_names(database_path)
    assert list(tmp_path.glob("*.backup")) == []
    assert _read_table_columns(database_path) == original_columns


async def test_incomplete_database_is_preserved_without_backup(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE sessions (id INTEGER)")
    original_columns = _read_table_columns(database_path)

    await database_manager.connect()
    with pytest.raises(MigrationError):
        await database_manager.initialize()

    assert database_path.is_file()
    assert list(tmp_path.glob("*.backup")) == []
    assert _read_table_columns(database_path) == original_columns


def test_initial_schema_sql_does_not_set_pragma() -> None:
    migration_sql = db_module._MIGRATION_PATH.read_text(encoding="utf-8")

    assert "PRAGMA" not in migration_sql.upper()


async def test_fresh_database_uses_wal_journal_mode(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await database_manager.connect()
    await database_manager.initialize()

    with sqlite3.connect(tmp_path / "test.db") as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert journal_mode.lower() == "wal"
