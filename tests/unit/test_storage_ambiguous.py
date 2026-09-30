"""Storage regression tests for ambiguous evaluator results and counts."""

import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

import pytest

from llm_guard_bench.domain.models import (
    EvaluationStage,
    EvaluationStatus,
    SessionSummary,
)
from llm_guard_bench.domain.models import (
    TestResult as ResultRecord,
)
from llm_guard_bench.storage import db as db_module
from llm_guard_bench.storage.db import DatabaseManager

SESSION_ID = "test-session"
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


def _make_test_result(status: EvaluationStatus) -> ResultRecord:
    stage: EvaluationStage = "STAGE_2_JUDGE"
    verdict = None
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
        session_id=SESSION_ID,
        timestamp=datetime.now(UTC),
        model_name="test-model",
        attack_id=ATTACK_ID,
        category="DAN",
        adversarial_prompt="test prompt",
        system_prompt=None,
        evaluation_status=status,
        evaluation_stage=stage,
        judge_verdict=verdict,
        judge_parse_error=judge_parse_error,
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
            (ATTACK_ID, "DAN", "Test", "Test attack", "test prompt", None, "REFUSAL", "LOW", "[]"),
        )


async def _initialize_database(database_manager: DatabaseManager, database_path: Path) -> None:
    await database_manager.connect()
    await database_manager.initialize()
    _seed_attack_definition(database_path)


@pytest.mark.parametrize("status", get_args(EvaluationStatus))
async def test_every_evaluation_status_can_be_stored(
    database_manager: DatabaseManager,
    tmp_path: Path,
    status: EvaluationStatus,
) -> None:
    database_path = tmp_path / "test.db"
    await _initialize_database(database_manager, database_path)
    await database_manager.insert_result(_make_test_result(status))

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("SELECT evaluation_status FROM test_results").fetchall()
    assert rows == [(status,)]


async def test_ambiguous_judge_verdict_is_storable(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"
    await _initialize_database(database_manager, database_path)
    result = _make_test_result("AMBIGUOUS")
    await database_manager.insert_result(result)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT evaluation_status, evaluation_stage, judge_verdict FROM test_results"
        ).fetchone()
    assert row == ("AMBIGUOUS", "STAGE_2_JUDGE", "AMBIGUOUS")


async def test_session_ambiguous_count_round_trips(
    database_manager: DatabaseManager,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.db"
    await database_manager.connect()
    await database_manager.initialize()
    summary = SessionSummary(
        session_id=SESSION_ID,
        started_at=datetime.now(UTC),
        config_snapshot={},
    )
    summary.increment("AMBIGUOUS")
    summary.increment("AMBIGUOUS")

    await database_manager.upsert_session(summary)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT ambiguous_count, total_tests FROM sessions WHERE session_id = ?",
            (SESSION_ID,),
        ).fetchone()
    assert row == (2, 2)
