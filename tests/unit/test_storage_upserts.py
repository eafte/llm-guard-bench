"""Regression tests for DatabaseManager upsert behavior."""

import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from llm_guard_bench.domain.models import (
    AttackDefinition,
    EvaluationStage,
    EvaluationStatus,
    SessionSummary,
)
from llm_guard_bench.domain.models import (
    TestResult as ResultRecord,
)
from llm_guard_bench.storage.db import DatabaseManager

ATTACK_ID = "test-attack"
SESSION_ID = "test-session"


@pytest.fixture
async def database_manager(
    tmp_path: Path,
) -> AsyncIterator[DatabaseManager]:
    manager = DatabaseManager(tmp_path / "test.db")
    try:
        yield manager
    finally:
        await manager.disconnect()


def _make_attack(
    *,
    description: str = "Test attack",
    severity: Literal[
        "LOW", "MEDIUM", "HIGH", "CRITICAL", "low", "medium", "high", "critical"
    ] = "LOW",
) -> AttackDefinition:
    return AttackDefinition(
        attack_id=ATTACK_ID,
        attack_name="Test",
        description=description,
        category="DAN",
        severity=severity,
        tags=["test"],
        turns=["test prompt"],
    )


def _make_test_result() -> ResultRecord:
    stage: EvaluationStage = "STAGE_1_KEYWORD"
    status: EvaluationStatus = "PASSED"
    return ResultRecord(
        session_id=SESSION_ID,
        timestamp=datetime.now(UTC),
        model_name="test-model",
        attack_id=ATTACK_ID,
        category="DAN",
        adversarial_prompt="test prompt",
        system_prompt=None,
        evaluation_status=status,
        evaluation_stage=stage,
    )


def _seed_attack_definition(database_path: Path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO attack_definitions (
                attack_id, category, attack_name, description,
                adversarial_prompt, system_prompt, expected_behavior,
                severity, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ATTACK_ID,
                "DAN",
                "Test",
                "Test attack",
                "test prompt",
                None,
                "REFUSAL",
                "LOW",
                "[]",
            ),
        )


async def test_attack_definition_upsert_normalizes_severity_and_sets_expected_behavior(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await database_manager.connect()
    await database_manager.initialize()

    await database_manager.upsert_attack_definition(_make_attack(severity="low"))

    with sqlite3.connect(tmp_path / "test.db") as connection:
        row = connection.execute(
            """
            SELECT severity, expected_behavior
            FROM attack_definitions
            WHERE attack_id = ?
            """,
            (ATTACK_ID,),
        ).fetchone()

    assert row == ("LOW", "REFUSAL")


async def test_attack_definition_upsert_updates_existing_description(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await database_manager.connect()
    await database_manager.initialize()
    await database_manager.upsert_attack_definition(_make_attack(description="Old description"))

    await database_manager.upsert_attack_definition(_make_attack(description="New description"))

    with sqlite3.connect(tmp_path / "test.db") as connection:
        rows = connection.execute(
            "SELECT description FROM attack_definitions WHERE attack_id = ?",
            (ATTACK_ID,),
        ).fetchall()

    assert rows == [("New description",)]


async def test_attack_definition_upsert_requires_manager_connection(
    database_manager: DatabaseManager,
) -> None:
    with pytest.raises(RuntimeError):
        await database_manager.upsert_attack_definition(_make_attack())


async def test_session_upsert_does_not_cascade_delete_results(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await database_manager.connect()
    await database_manager.initialize()
    assert database_manager._connection is not None
    await database_manager._connection.execute("PRAGMA foreign_keys=ON")

    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    summary = SessionSummary(
        session_id=SESSION_ID,
        started_at=started_at,
        config_snapshot={},
    )
    await database_manager.upsert_session(summary)
    _seed_attack_definition(tmp_path / "test.db")
    await database_manager.insert_result(_make_test_result())

    finished_at = datetime(2026, 1, 2, tzinfo=UTC)
    summary.finished_at = finished_at
    summary.total_tests = 1
    summary.passed_count = 1
    await database_manager.upsert_session(summary)

    with sqlite3.connect(tmp_path / "test.db") as connection:
        result_count = connection.execute(
            "SELECT COUNT(*) FROM test_results WHERE session_id = ?",
            (SESSION_ID,),
        ).fetchone()[0]
        session_row = connection.execute(
            """
            SELECT finished_at, total_tests, passed_count
            FROM sessions
            WHERE session_id = ?
            """,
            (SESSION_ID,),
        ).fetchone()

    assert result_count == 1
    assert session_row == (finished_at.isoformat(), 1, 1)
