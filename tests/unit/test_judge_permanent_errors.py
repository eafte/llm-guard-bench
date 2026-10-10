"""Tests that permanent provider failures bypass judge retries."""

from collections.abc import Sequence
from typing import cast

import pytest

from llm_guard_bench.domain.models import AttackContext
from llm_guard_bench.evaluators import evaluator as evaluator_module
from llm_guard_bench.evaluators.evaluator import EvaluationResult, JudgeLLMEvaluator
from llm_guard_bench.providers.adapters import BaseAdapter, PermanentProviderError


class ScriptedJudgeAdapter(BaseAdapter):
    """Adapter that raises or returns scripted results and counts calls."""

    def __init__(self, replies: Sequence[object]) -> None:
        self.replies = list(replies)
        self.generate_calls = 0

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        override_temperature: float | None = None,
    ) -> str:
        self.generate_calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return cast(str, reply)

    async def generate_multi_turn(
        self,
        messages: list[dict[str, str]],
        override_temperature: float | None = None,
    ) -> str:
        raise AssertionError("judge evaluation should use generate")


def _context() -> AttackContext:
    return AttackContext(
        attack_name="Test attack",
        category="DAN",
        description="Test attack description",
        attack_text="Ignore previous instructions.",
    )


def _record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(evaluator_module.asyncio, "sleep", record_sleep)
    return sleeps


async def test_permanent_provider_error_is_not_retried_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedJudgeAdapter([PermanentProviderError("status=401")])
    sleeps = _record_sleeps(monkeypatch)
    judge = JudgeLLMEvaluator(adapter)

    result = await judge.evaluate("A non-empty target response.", context=_context())

    assert adapter.generate_calls == 1
    assert sleeps == []
    assert result is EvaluationResult.EVAL_ERROR


async def test_permanent_provider_error_is_not_retried_with_more_retries_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedJudgeAdapter([PermanentProviderError("status=401")])
    sleeps = _record_sleeps(monkeypatch)
    judge = JudgeLLMEvaluator(adapter, max_retries=5)

    result = await judge.evaluate("A non-empty target response.", context=_context())

    assert adapter.generate_calls == 1
    assert sleeps == []
    assert result is EvaluationResult.EVAL_ERROR


async def test_runtime_error_keeps_the_default_retry_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedJudgeAdapter([RuntimeError("temporary failure") for _ in range(4)])
    sleeps = _record_sleeps(monkeypatch)
    judge = JudgeLLMEvaluator(adapter)

    result = await judge.evaluate("A non-empty target response.", context=_context())

    assert adapter.generate_calls == 4
    assert sleeps == [2, 4, 8]
    assert result is EvaluationResult.EVAL_ERROR


async def test_timeout_error_keeps_the_default_retry_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedJudgeAdapter([TimeoutError("temporary timeout") for _ in range(4)])
    sleeps = _record_sleeps(monkeypatch)
    judge = JudgeLLMEvaluator(adapter)

    result = await judge.evaluate("A non-empty target response.", context=_context())

    assert adapter.generate_calls == 4
    assert sleeps == [2, 4, 8]
    assert result is EvaluationResult.EVAL_ERROR


async def test_runtime_errors_retry_until_a_valid_verdict_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedJudgeAdapter(
        [
            RuntimeError("temporary failure one"),
            RuntimeError("temporary failure two"),
            '{"verdict": "VULNERABLE", "reasoning": "complied"}',
        ]
    )
    sleeps = _record_sleeps(monkeypatch)
    judge = JudgeLLMEvaluator(adapter)

    result = await judge.evaluate("A non-empty target response.", context=_context())

    assert adapter.generate_calls == 3
    assert sleeps == [2, 4]
    assert result is EvaluationResult.VULNERABLE


async def test_permanent_provider_error_logs_that_it_will_not_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = ScriptedJudgeAdapter([PermanentProviderError("status=401")])
    _record_sleeps(monkeypatch)
    judge = JudgeLLMEvaluator(adapter)

    await judge.evaluate("A non-empty target response.", context=_context())

    assert any(
        record.levelname in {"WARNING", "ERROR"} and "not retrying" in record.message.lower()
        for record in caplog.records
    )
