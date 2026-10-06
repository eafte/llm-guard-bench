"""Tests for orchestrator session identifiers."""

import re

from llm_guard_bench import cli


def _make_orchestrator() -> cli.LLMGuardBenchOrchestrator:
    return cli.LLMGuardBenchOrchestrator(
        target="target-model",
        judge="judge-model",
        concurrency=1,
        auto_flush=False,
    )


def test_session_id_matches_timestamp_and_unique_suffix_pattern() -> None:
    orchestrator = _make_orchestrator()

    assert re.fullmatch(r"^SESS_\d{8}_\d{6}_[0-9a-f]{8}$", orchestrator.session_id)


def test_two_hundred_orchestrators_have_distinct_session_ids() -> None:
    session_ids = [_make_orchestrator().session_id for _ in range(200)]

    assert len(set(session_ids)) == 200


def test_session_id_is_stable_and_identifies_logger() -> None:
    orchestrator = _make_orchestrator()

    first_session_id = orchestrator.session_id
    second_session_id = orchestrator.session_id

    assert first_session_id == second_session_id
    assert orchestrator.logger.name.endswith(first_session_id)
