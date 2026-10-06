# db/db.py
# DatabaseManager: async SQLite interface using aiosqlite.
# All SQL statements reference the v3.0 column names: model_name, category.
# BULLETPROOF: Validates schema integrity and gracefully handles database repairs.

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import aiosqlite

from llm_guard_bench.domain.models import AttackDefinition, SessionSummary, TestResult
from llm_guard_bench.settings import DB_PATH, RESULTS_DIR
from llm_guard_bench.storage.errors import StorageError
from llm_guard_bench.storage.migrations import (
    MigrationError,
    apply_migrations,
    discover_migrations,
)

logger = logging.getLogger(__name__)

_MIGRATION_PATH = Path(__file__).parent / "sql" / "001_initial_schema.sql"
EXPECTED_BEHAVIOR_REFUSAL = "REFUSAL"

# ============================================================================
# REQUIRED TABLES (v3.0) — Enforced by validator
# ============================================================================
REQUIRED_TABLES = {
    "test_results",
    "sessions",
    "attack_definitions",
}


def _initialize_database(db_path: Path) -> None:
    """Apply versioned migrations to a SQLite database in a worker thread."""
    with closing(sqlite3.connect(db_path)) as conn:
        table_names = {
            row[0]
            for row in conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                """
            )
        }
        if "schema_migrations" not in table_names and table_names:
            raise MigrationError(
                "This database predates versioned migrations and must not be "
                "modified automatically; use a new database file. The "
                "schema_migrations table is missing."
            )

        conn.execute("PRAGMA journal_mode=WAL")
        apply_migrations(conn, discover_migrations(_MIGRATION_PATH.parent))


class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._schema_validated = False
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    async def connect(self) -> None:
        """
        Establish async connection to the SQLite database.
        Called by the orchestrator on startup.
        """
        if self._connection is None:
            connection = await aiosqlite.connect(self._db_path)
            await connection.execute("PRAGMA foreign_keys=ON")
            cursor = await connection.execute("PRAGMA foreign_keys")
            row = await cursor.fetchone()
            if row is None or row[0] != 1:
                await connection.close()
                self._connection = None
                raise StorageError("SQLite foreign_keys pragma did not take effect")

            self._connection = connection
            logger.info(f"Connected to database at {self._db_path}")
        else:
            logger.debug("Database connection already established")

    async def run_migrations(self) -> None:
        """
        Apply the initial schema migration (001_initial_schema.sql).
        Safe to call on an already-initialized database: all statements
        use CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS.
        Must be called once per session before any other DatabaseManager method.
        """
        await self.initialize()

    async def disconnect(self) -> None:
        """
        Close the async connection to the SQLite database.
        Called by the orchestrator on shutdown.
        """
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
        else:
            logger.debug("No active database connection to close")

    async def initialize(self) -> None:
        """
        Applies the initial schema migration (001_initial_schema.sql) with validation.
        Safe to call on an already-initialised database: all statements
        use CREATE TABLE IF NOT EXISTS and CREATE INDEX IF NOT EXISTS.

        BULLETPROOF: Validates schema after migration and raises if tables are missing.
        """
        await asyncio.to_thread(_initialize_database, self._db_path)
        logger.info("Database initialised at %s", self._db_path)

        # Validate schema after migration
        await self._validate_schema()

    async def _get_table_names(self) -> set[str]:
        """
        Get all table names in the database.
        BULLETPROOF: Used for validation and error diagnostics.
        """
        try:
            async with aiosqlite.connect(self._db_path) as conn:
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                rows = await cursor.fetchall()
                return {row[0] for row in rows}
        except Exception as e:
            logger.error(f"Failed to enumerate tables: {e}")
            return set()

    async def _validate_schema(self) -> None:
        """
        Validate that all required v3.0 tables exist.
        BULLETPROOF: Raises RuntimeError if schema is invalid.
        """
        existing_tables = await self._get_table_names()
        missing_tables = REQUIRED_TABLES - existing_tables

        if missing_tables:
            msg = (
                f"DATABASE SCHEMA MISMATCH DETECTED!\n"
                f"Missing tables: {missing_tables}\n"
                f"Expected: {REQUIRED_TABLES}\n"
                f"Found: {existing_tables}\n"
                f"Database path: {self._db_path}\n"
                f"Migration file: {_MIGRATION_PATH}\n"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        logger.debug(f"Schema validation passed. Tables: {existing_tables}")
        self._schema_validated = True

    async def insert_result(self, result: TestResult) -> None:
        """
        Writes a TestResult to:
          1. SQLite test_results table (via parameterised INSERT)
          2. JSONL file at results/session_{session_id}.jsonl (append)

        Both writes complete before this coroutine returns.
        If the JSONL write fails (e.g., disk full), the error is logged but
        not re-raised — the SQLite record is the authoritative copy.
        """
        if self._connection is None:
            raise RuntimeError("Database connection not established. Call connect() first.")

        conn = self._connection
        try:
            await conn.execute(
                """
                INSERT INTO test_results (
                    session_id,        timestamp,          model_name,
                    attack_id,         category,           adversarial_prompt,
                    system_prompt,     raw_llm_response,   evaluation_status,
                    evaluation_stage,  judge_verdict,      judge_parse_error,
                    execution_time_ms, total_time_ms,      prompt_tokens,
                    completion_tokens, http_status_code,   error_message
                ) VALUES (
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
                )
                """,
                (
                    result.session_id,
                    result.timestamp.isoformat(),
                    result.model_name,  # v3.0: was target_model_name
                    result.attack_id,
                    result.category,  # v3.0: was attack_category
                    result.adversarial_prompt,
                    result.system_prompt,
                    result.raw_llm_response,
                    result.evaluation_status,
                    result.evaluation_stage,
                    result.judge_verdict,
                    int(result.judge_parse_error),
                    result.execution_time_ms,
                    result.total_time_ms,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.http_status_code,
                    result.error_message,
                ),
            )
        except Exception:
            await conn.rollback()
            raise

        await conn.commit()
        logger.debug(f"Result persisted for session {result.session_id}, attack {result.attack_id}")

        # Append to JSONL
        jsonl_path = RESULTS_DIR / f"session_{result.session_id}.jsonl"
        try:
            with open(jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(result.model_dump_json() + "\n")
        except OSError as exc:
            logger.error(
                "JSONL write failed for session %s: %s. SQLite record preserved.",
                result.session_id,
                exc,
            )

    async def upsert_session(self, summary: SessionSummary) -> None:
        """
        Inserts or updates the session record.
        Called at session start (finished_at=NULL, all counts 0)
        and at session end (finished_at set, counts finalised).
        """
        if self._connection is None:
            raise RuntimeError("Database connection not established. Call connect() first.")

        conn = self._connection
        await conn.execute(
            """
            INSERT INTO sessions (
                session_id,      started_at,       finished_at,
                config_snapshot,
                total_tests,     passed_count,     vulnerable_count,
                ambiguous_count, failed_count,      eval_error_count,
                timeout_count,   skipped_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                config_snapshot = excluded.config_snapshot,
                total_tests = excluded.total_tests,
                passed_count = excluded.passed_count,
                vulnerable_count = excluded.vulnerable_count,
                ambiguous_count = excluded.ambiguous_count,
                failed_count = excluded.failed_count,
                eval_error_count = excluded.eval_error_count,
                timeout_count = excluded.timeout_count,
                skipped_count = excluded.skipped_count
            """,
            (
                summary.session_id,
                summary.started_at.isoformat(),
                summary.finished_at.isoformat() if summary.finished_at else None,
                json.dumps(summary.config_snapshot),
                summary.total_tests,
                summary.passed_count,
                summary.vulnerable_count,
                summary.ambiguous_count,
                summary.failed_count,
                summary.eval_error_count,
                summary.timeout_count,
                summary.skipped_count,
            ),
        )
        await conn.commit()

    async def finalize_session(self, session_id: str, finished_at: datetime) -> None:
        if self._connection is None:
            raise RuntimeError("Database connection not established. Call connect() first.")

        conn = self._connection
        missing_session = False
        try:
            cursor = await conn.execute(
                """
                UPDATE sessions
                SET finished_at = ?,
                    total_tests = (
                        SELECT COUNT(*) FROM test_results WHERE session_id = ?
                    ),
                    passed_count = (
                        SELECT COUNT(*) FROM test_results
                        WHERE session_id = ? AND evaluation_status = 'PASSED'
                    ),
                    vulnerable_count = (
                        SELECT COUNT(*) FROM test_results
                        WHERE session_id = ? AND evaluation_status = 'VULNERABLE'
                    ),
                    ambiguous_count = (
                        SELECT COUNT(*) FROM test_results
                        WHERE session_id = ? AND evaluation_status = 'AMBIGUOUS'
                    ),
                    failed_count = (
                        SELECT COUNT(*) FROM test_results
                        WHERE session_id = ? AND evaluation_status = 'FAILED'
                    ),
                    eval_error_count = (
                        SELECT COUNT(*) FROM test_results
                        WHERE session_id = ? AND evaluation_status = 'EVAL_ERROR'
                    ),
                    timeout_count = (
                        SELECT COUNT(*) FROM test_results
                        WHERE session_id = ? AND evaluation_status = 'TIMEOUT'
                    ),
                    skipped_count = (
                        SELECT COUNT(*) FROM test_results
                        WHERE session_id = ? AND evaluation_status = 'SKIPPED'
                    )
                WHERE session_id = ?
                """,
                (
                    finished_at.isoformat(),
                    session_id,
                    session_id,
                    session_id,
                    session_id,
                    session_id,
                    session_id,
                    session_id,
                    session_id,
                    session_id,
                ),
            )
            if cursor.rowcount == 0:
                missing_session = True
            else:
                await conn.commit()
        except Exception:
            await conn.rollback()
            raise

        if missing_session:
            await conn.rollback()
            raise StorageError(f"Cannot finalize unknown session {session_id}")

    async def upsert_attack_definition(self, attack: AttackDefinition) -> None:
        """
        Inserts or updates one attack definition in the attack_definitions table.
        Changes to prompts.json propagate automatically on every run.
        """
        if self._connection is None:
            raise RuntimeError("Database connection not established. Call connect() first.")

        conn = self._connection
        await conn.execute(
            """
            INSERT INTO attack_definitions (
                attack_id,        category,          attack_name,
                description,      adversarial_prompt, system_prompt,
                expected_behavior, severity,          tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(attack_id) DO UPDATE SET
                category = excluded.category,
                attack_name = excluded.attack_name,
                description = excluded.description,
                adversarial_prompt = excluded.adversarial_prompt,
                system_prompt = excluded.system_prompt,
                expected_behavior = excluded.expected_behavior,
                severity = excluded.severity,
                tags = excluded.tags
            """,
            (
                attack.attack_id,
                attack.category,  # v3.0: was attack_category
                attack.attack_name,
                attack.description,
                attack.adversarial_prompt,
                attack.system_prompt,
                EXPECTED_BEHAVIOR_REFUSAL,
                attack.severity.upper(),
                json.dumps(attack.tags),
            ),
        )
        await conn.commit()

    async def get_test_results_count(self, session_id: str | None = None) -> int:
        """
        Get the count of records in test_results table.
        BULLETPROOF: Used for validation after benchmark completion.

        Args:
            session_id: Optional session ID to filter by. If None, counts all records.

        Returns:
            Total row count in test_results table
        """
        if self._connection is None:
            raise RuntimeError("Database connection not established. Call connect() first.")

        conn = self._connection
        if session_id:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM test_results WHERE session_id = ?",
                (session_id,),
            )
        else:
            cursor = await conn.execute("SELECT COUNT(*) FROM test_results")

        row = await cursor.fetchone()
        return row[0] if row else 0
