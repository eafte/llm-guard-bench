import hashlib
import sqlite3
from pathlib import Path

import pytest

from llm_guard_bench.storage.migrations import (
    Migration,
    MigrationError,
    apply_migrations,
    discover_migrations,
)


def _migration(version: int, name: str, sql: str) -> Migration:
    checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
    return Migration(version=version, name=name, sql=sql, checksum=checksum)


def test_discover_migrations_sorts_checksums_and_ignores_nonmatching_files(
    tmp_path: Path,
) -> None:
    first_sql = "CREATE TABLE first_table (id INTEGER PRIMARY KEY);"
    second_sql = "CREATE TABLE second_table (id INTEGER PRIMARY KEY);"
    (tmp_path / "002_second.sql").write_text(second_sql, encoding="utf-8")
    (tmp_path / "001_first.sql").write_text(first_sql, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a migration", encoding="utf-8")
    (tmp_path / "003.sql").write_text("not named", encoding="utf-8")
    (tmp_path / "004_bad-name.sql").write_text("not a name", encoding="utf-8")

    migrations = discover_migrations(tmp_path)

    assert [(migration.version, migration.name) for migration in migrations] == [
        (1, "first"),
        (2, "second"),
    ]
    assert [migration.sql for migration in migrations] == [first_sql, second_sql]
    assert [migration.checksum for migration in migrations] == [
        hashlib.sha256(first_sql.encode("utf-8")).hexdigest(),
        hashlib.sha256(second_sql.encode("utf-8")).hexdigest(),
    ]


def test_discover_migrations_rejects_duplicate_versions(tmp_path: Path) -> None:
    (tmp_path / "001_first.sql").write_text("SELECT 1;", encoding="utf-8")
    (tmp_path / "001_also_first.sql").write_text("SELECT 2;", encoding="utf-8")

    with pytest.raises(MigrationError, match="duplicate"):
        discover_migrations(tmp_path)


def test_apply_migrations_creates_tracking_table_and_applies_in_order() -> None:
    migrations = [
        _migration(1, "first", "CREATE TABLE first_table (id INTEGER PRIMARY KEY);"),
        _migration(2, "second", "CREATE TABLE second_table (id INTEGER PRIMARY KEY);"),
    ]
    with sqlite3.connect(":memory:") as connection:
        applied = apply_migrations(connection, migrations)

        assert applied == [1, 2]
        columns = connection.execute("PRAGMA table_info(schema_migrations)").fetchall()
        assert [column[1] for column in columns] == [
            "version",
            "name",
            "checksum",
            "applied_at",
        ]
        assert columns[0][5] == 1
        rows = connection.execute(
            "SELECT version, name, checksum, applied_at FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [(row[0], row[1], row[2]) for row in rows] == [
            (migration.version, migration.name, migration.checksum) for migration in migrations
        ]
        assert all(row[3] for row in rows)
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"first_table", "second_table"} <= tables


def test_second_apply_is_a_noop() -> None:
    migrations = [_migration(1, "first", "CREATE TABLE first_table (id INTEGER PRIMARY KEY);")]
    with sqlite3.connect(":memory:") as connection:
        assert apply_migrations(connection, migrations) == [1]

        assert apply_migrations(connection, migrations) == []

        assert connection.execute("SELECT version FROM schema_migrations").fetchall() == [(1,)]


def test_failing_migration_rolls_back_completely_and_keeps_prior_migrations() -> None:
    first = _migration(1, "first", "CREATE TABLE first_table (id INTEGER PRIMARY KEY);")
    failing = _migration(
        2,
        "failing",
        "CREATE TABLE partial_table (id INTEGER PRIMARY KEY);THIS IS NOT VALID SQL;",
    )
    with sqlite3.connect(":memory:") as connection:
        with pytest.raises(MigrationError):
            apply_migrations(connection, [first, failing])

        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "first_table" in tables
        assert "partial_table" not in tables
        assert connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(1,)]


def test_changing_applied_migration_sql_raises_checksum_mismatch() -> None:
    original = _migration(1, "first", "CREATE TABLE first_table (id INTEGER PRIMARY KEY);")
    changed = _migration(1, "first", "CREATE TABLE altered_table (id INTEGER PRIMARY KEY);")
    with sqlite3.connect(":memory:") as connection:
        assert apply_migrations(connection, [original]) == [1]

        with pytest.raises(MigrationError, match="checksum"):
            apply_migrations(connection, [changed])


def test_new_migration_applies_on_already_migrated_database() -> None:
    first = _migration(1, "first", "CREATE TABLE first_table (id INTEGER PRIMARY KEY);")
    second = _migration(2, "second", "CREATE TABLE second_table (id INTEGER PRIMARY KEY);")
    with sqlite3.connect(":memory:") as connection:
        assert apply_migrations(connection, [first]) == [1]

        assert apply_migrations(connection, [first, second]) == [2]

        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"first_table", "second_table"} <= tables
        assert connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(1,), (2,)]
