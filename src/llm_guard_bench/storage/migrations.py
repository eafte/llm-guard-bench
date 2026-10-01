"""Versioned SQLite migration discovery and application."""

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_MIGRATION_FILENAME = re.compile(r"^(\d+)_([A-Za-z0-9_]+)\.sql$")


class MigrationError(Exception):
    """Raised when migrations cannot be discovered or applied safely."""


@dataclass(frozen=True)
class Migration:
    """A versioned SQL migration and its SHA-256 checksum."""

    version: int
    name: str
    sql: str
    checksum: str


def discover_migrations(directory: Path) -> list[Migration]:
    """Discover NNN_name.sql files in version order and compute checksums."""
    migrations = []
    for path in directory.iterdir():
        match = _MIGRATION_FILENAME.fullmatch(path.name)
        if match is None or not path.is_file():
            continue

        version = int(match.group(1))
        name = match.group(2)
        sql = path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        migrations.append(Migration(version, name, sql, checksum))

    migrations.sort(key=lambda migration: migration.version)
    versions = [migration.version for migration in migrations]
    if len(versions) != len(set(versions)):
        raise MigrationError("duplicate migration version")

    return migrations


def apply_migrations(conn: sqlite3.Connection, migrations: list[Migration]) -> list[int]:
    """Apply unapplied migrations and return the versions applied in this call."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )
    applied_checksums = dict(
        conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    )
    ordered_migrations = sorted(migrations, key=lambda migration: migration.version)

    for migration in ordered_migrations:
        applied_checksum = applied_checksums.get(migration.version)
        if applied_checksum is not None and applied_checksum != migration.checksum:
            raise MigrationError(f"checksum mismatch for migration {migration.version}")

    applied_versions = []
    for migration in ordered_migrations:
        if migration.version in applied_checksums:
            continue

        try:
            conn.executescript("BEGIN IMMEDIATE;\n" + migration.sql)
            conn.execute(
                """
                INSERT INTO schema_migrations (version, name, checksum, applied_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    migration.version,
                    migration.name,
                    migration.checksum,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.execute("COMMIT")
        except sqlite3.Error as error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise MigrationError(
                f"failed to apply migration {migration.version}: {error}"
            ) from error

        applied_versions.append(migration.version)

    return applied_versions
