"""Database initialization tests using an isolated temporary SQLite file."""

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from llm_guard_bench.storage import db as db_module
from llm_guard_bench.storage.db import DatabaseManager


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


async def test_migration_file_exists_at_configured_path(
    database_manager: DatabaseManager,
) -> None:
    assert db_module._MIGRATION_PATH.is_file() is True


async def test_fresh_database_creates_required_tables(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await database_manager.connect()
    await database_manager.initialize()

    tables = _read_table_names(tmp_path / "test.db")
    assert {"test_results", "sessions", "attack_definitions"}.issubset(tables)


async def test_initialize_is_idempotent(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await database_manager.connect()
    await database_manager.initialize()
    await database_manager.initialize()

    tables = _read_table_names(tmp_path / "test.db")
    assert {"test_results", "sessions", "attack_definitions"}.issubset(tables)
