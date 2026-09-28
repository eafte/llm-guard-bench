# db/db.py
# DatabaseManager: async SQLite interface using aiosqlite.
# All SQL statements reference the v3.0 column names: model_name, category.
# BULLETPROOF: Validates schema integrity and gracefully handles database repairs.

from __future__ import annotations

import json
import logging
from pathlib import Path

import aiosqlite

from config.settings import DB_PATH, RESULTS_DIR
from core.models import AttackDefinition, SessionSummary, TestResult

logger = logging.getLogger(__name__)

_MIGRATION_PATH = Path(__file__).parent / "migrations" / "001_initial_schema.sql"

# ============================================================================
# REQUIRED TABLES (v3.0) — Enforced by validator
# ============================================================================
REQUIRED_TABLES = {
    "test_results",
    "sessions",
    "attack_definitions",
}


class DatabaseManager:

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._connection = None
        self._schema_validated = False
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    async def connect(self) -> None:
        """
        Establish async connection to the SQLite database.
        Called by the orchestrator on startup.
        """
        if self._connection is None:
            self._connection = await aiosqlite.connect(self._db_path)
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
        migration_sql = _MIGRATION_PATH.read_text(encoding="utf-8")
        
        # Remove old corrupted database if schema is invalid
        if self._db_path.exists():
            try:
                # Quick check for schema validity before full migration
                existing_tables = await self._get_table_names()
                if existing_tables and not REQUIRED_TABLES.issubset(existing_tables):
                    logger.warning(
                        f"Database at {self._db_path} has invalid schema. "
                        f"Expected tables: {REQUIRED_TABLES}, Found: {existing_tables}. "
                        f"Rebuilding database..."
                    )
                    # Backup old database
                    backup_path = self._db_path.with_suffix('.db.backup')
                    self._db_path.rename(backup_path)
                    logger.info(f"Backed up corrupted database to {backup_path}")
            except Exception as e:
                logger.debug(f"Schema pre-check failed (expected if DB is new): {e}")
        
        # Apply migrations
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.executescript(migration_sql)
            await conn.commit()
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
                result.model_name,              # v3.0: was target_model_name
                result.attack_id,
                result.category,                # v3.0: was attack_category
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
                result.session_id, exc,
            )

    async def upsert_session(self, summary: SessionSummary) -> None:
        """
        Inserts or replaces the session record.
        Called at session start (finished_at=NULL, all counts 0)
        and at session end (finished_at set, counts finalised).
        """
        if self._connection is None:
            raise RuntimeError("Database connection not established. Call connect() first.")
        
        conn = self._connection
        await conn.execute(
            """
            INSERT OR REPLACE INTO sessions (
                session_id,      started_at,       finished_at,
                config_snapshot,
                total_tests,     passed_count,     vulnerable_count,
                failed_count,    eval_error_count,
                timeout_count,   skipped_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.session_id,
                summary.started_at.isoformat(),
                summary.finished_at.isoformat() if summary.finished_at else None,
                json.dumps(summary.config_snapshot),
                summary.total_tests,
                summary.passed_count,
                summary.vulnerable_count,
                summary.failed_count,
                summary.eval_error_count,
                summary.timeout_count,
                summary.skipped_count,
            ),
        )
        await conn.commit()

    async def upsert_attack_definition(self, attack: AttackDefinition) -> None:
        """
        Inserts or replaces one attack definition in the attack_definitions table.
        Uses INSERT OR REPLACE so changes to prompts.json propagate on every run.
        """
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO attack_definitions (
                    attack_id,        category,          attack_name,
                    description,      adversarial_prompt, system_prompt,
                    expected_behavior, severity,          tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attack.attack_id,
                    attack.category,              # v3.0: was attack_category
                    attack.attack_name,
                    attack.description,
                    attack.adversarial_prompt,
                    attack.system_prompt,
                    attack.expected_behavior,
                    attack.severity,
                    json.dumps(attack.tags),
                ),
            )
            await conn.commit()

    async def get_test_results_count(self, session_id: str = None) -> int:
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

