"""Tests for finalizing session records."""

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
from llm_guard_bench.storage.errors import StorageError

SESSION_ID = "test-session"
OTHER_SESSION_ID = "other-session"
ATTACK_ID = "test-attack"


@pytest.fixture
async def database_manager(
    tmp_path: Path,
) -> AsyncIterator[DatabaseManager]:
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
    status: EvaluationStatus,
    *,
    session_id: str = SESSION_ID,
    model_name: str = "test-model",
) -> ResultRecord:
    stage: EvaluationStage = "STAGE_2_JUDGE"
    verdict: Literal["PASSED", "VULNERABLE", "AMBIGUOUS"] | None = None
    judge_parse_error = False
    if status == "PASSED":
        stage = "STAGE_1_KEYWORD"
    elif status in ("VULNERABLE", "AMBIGUOUS"):
        verdict = status
    elif status == "FAILED":
        judge_parse_error = True
    elif status in ("EVAL_ERROR", "TIMEOUT", "SKIPPED"):
        stage = "PRE_FLIGHT"

    return ResultRecord(
        session_id=session_id,
        timestamp=datetime.now(UTC),
        model_name=model_name,
        attack_id=ATTACK_ID,
        category="DAN",
        adversarial_prompt="A harmless test prompt.",
        system_prompt=None,
        evaluation_status=status,
        evaluation_stage=stage,
        judge_verdict=verdict,
        judge_parse_error=judge_parse_error,
    )


async def _initialize_database(manager: DatabaseManager) -> None:
    await manager.connect()
    await manager.initialize()


async def _add_parents(
    manager: DatabaseManager,
    *,
    session_id: str = SESSION_ID,
    config_snapshot: dict[str, str] | None = None,
) -> SessionSummary:
    summary = SessionSummary(
        session_id=session_id,
        started_at=datetime.now(UTC),
        config_snapshot=config_snapshot if config_snapshot is not None else {},
    )
    await manager.upsert_session(summary)
    await manager.upsert_attack_definition(_make_attack())
    return summary


def _read_session(database_path: Path, session_id: str) -> tuple[object, ...] | None:
    with sqlite3.connect(database_path) as connection:
        return connection.execute(
            """
            SELECT started_at, finished_at, config_snapshot, total_tests,
                   passed_count, vulnerable_count, ambiguous_count, failed_count,
                   eval_error_count, timeout_count, skipped_count
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()


async def test_finalize_session_counts_results_by_status(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await _initialize_database(database_manager)
    await _add_parents(database_manager)
    for index, status in enumerate(("PASSED", "PASSED", "VULNERABLE", "AMBIGUOUS", "EVAL_ERROR")):
        await database_manager.insert_result(_make_result(status, model_name=f"model-{index}"))
    finished_at = datetime.now(UTC)

    await database_manager.finalize_session(SESSION_ID, finished_at)

    row = _read_session(tmp_path / "test.db", SESSION_ID)
    assert row == (
        row[0],
        finished_at.isoformat(),
        row[2],
        5,
        2,
        1,
        1,
        0,
        1,
        0,
        0,
    )


async def test_finalize_session_preserves_started_at_and_config_snapshot(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await _initialize_database(database_manager)
    summary = await _add_parents(database_manager, config_snapshot={"target": "t"})
    finished_at = datetime.now(UTC)

    await database_manager.finalize_session(SESSION_ID, finished_at)

    row = _read_session(tmp_path / "test.db", SESSION_ID)
    assert row is not None
    assert row[0] == summary.started_at.isoformat()
    assert row[1] == finished_at.isoformat()
    assert row[2] == '{"target": "t"}'


async def test_finalize_session_counts_only_target_session(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await _initialize_database(database_manager)
    await _add_parents(database_manager)
    await _add_parents(database_manager, session_id=OTHER_SESSION_ID)
    await database_manager.insert_result(_make_result("PASSED", session_id=SESSION_ID))
    await database_manager.insert_result(_make_result("VULNERABLE", session_id=OTHER_SESSION_ID))
    finished_at = datetime.now(UTC)

    await database_manager.finalize_session(SESSION_ID, finished_at)

    target_row = _read_session(tmp_path / "test.db", SESSION_ID)
    other_row = _read_session(tmp_path / "test.db", OTHER_SESSION_ID)
    assert target_row is not None
    assert target_row[1] == finished_at.isoformat()
    assert target_row[3:8] == (1, 1, 0, 0, 0)
    assert other_row is not None
    assert other_row[1] is None
    assert other_row[3:8] == (0, 0, 0, 0, 0)
    with sqlite3.connect(tmp_path / "test.db") as connection:
        other_result_count = connection.execute(
            "SELECT COUNT(*) FROM test_results WHERE session_id = ?",
            (OTHER_SESSION_ID,),
        ).fetchone()[0]
    assert other_result_count == 1


async def test_finalize_session_for_unknown_session_raises_storage_error(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await _initialize_database(database_manager)

    with pytest.raises(StorageError):
        await database_manager.finalize_session("missing-session", datetime.now(UTC))

    assert _read_session(tmp_path / "test.db", "missing-session") is None


async def test_finalize_session_with_no_results_sets_zero_counts(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    await _initialize_database(database_manager)
    await _add_parents(database_manager)
    finished_at = datetime.now(UTC)

    await database_manager.finalize_session(SESSION_ID, finished_at)

    row = _read_session(tmp_path / "test.db", SESSION_ID)
    assert row is not None
    assert row[1] == finished_at.isoformat()
    assert row[3:] == (0, 0, 0, 0, 0, 0, 0, 0)


async def test_finalize_session_requires_connection(
    database_manager: DatabaseManager,
) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        await database_manager.finalize_session(SESSION_ID, datetime.now(UTC))

    assert exc_info.type is RuntimeError


@pytest.mark.parametrize(
    ("status", "expected_counts"),
    [
        ("PASSED", (1, 1, 0, 0, 0, 0, 0, 0)),
        ("VULNERABLE", (1, 0, 1, 0, 0, 0, 0, 0)),
        ("AMBIGUOUS", (1, 0, 0, 1, 0, 0, 0, 0)),
        ("FAILED", (1, 0, 0, 0, 1, 0, 0, 0)),
        ("EVAL_ERROR", (1, 0, 0, 0, 0, 1, 0, 0)),
        ("TIMEOUT", (1, 0, 0, 0, 0, 0, 1, 0)),
        ("SKIPPED", (1, 0, 0, 0, 0, 0, 0, 1)),
    ],
)
async def test_finalize_session_maps_each_status_to_its_own_column(
    database_manager: DatabaseManager,
    tmp_path: Path,
    status: EvaluationStatus,
    expected_counts: tuple[int, int, int, int, int, int, int, int],
) -> None:
    await _initialize_database(database_manager)
    await _add_parents(database_manager)
    await database_manager.insert_result(_make_result(status))
    await database_manager.finalize_session(SESSION_ID, datetime.now(UTC))

    row = _read_session(tmp_path / "test.db", SESSION_ID)
    assert row is not None
    assert row[3:] == expected_counts
