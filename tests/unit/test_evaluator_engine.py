"""Regression tests for missing evidence in the evaluation cascade."""

from collections.abc import Sequence
from typing import cast

import pytest

from llm_guard_bench.evaluators import evaluator as evaluator_module
from llm_guard_bench.evaluators.evaluator import EvaluationEngine, EvaluationResult
from llm_guard_bench.providers.adapters import BaseAdapter


class FakeAdapter(BaseAdapter):
    """Adapter that returns or raises scripted replies and counts judge calls."""

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
        if not self.replies:
            raise AssertionError("No scripted judge reply remains")

        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        # None is deliberately passed through to test malformed adapter output.
        return cast(str, reply)

    async def generate_multi_turn(
        self,
        messages: list[dict[str, str]],
        override_temperature: float | None = None,
    ) -> str:
        return await self.generate(
            system_prompt="",
            user_prompt=str(messages),
            override_temperature=override_temperature,
        )


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def sleep_without_delay(delay: float) -> None:
        return None

    monkeypatch.setattr(evaluator_module.asyncio, "sleep", sleep_without_delay)


@pytest.mark.parametrize("target_response", ["", None])
async def test_missing_target_response_is_an_evaluation_error_without_calling_judge(
    target_response: str | None,
) -> None:
    adapter = FakeAdapter([])
    engine = EvaluationEngine(adapter)

    result = await engine.evaluate(cast(str, target_response))

    assert result is EvaluationResult.EVAL_ERROR
    assert adapter.generate_calls == 0


@pytest.mark.parametrize(
    ("judge_reply", "expected"),
    [
        ('{"verdict":"PASSED","reasoning":"x"}', EvaluationResult.PASSED),
        ('{"verdict":"VULNERABLE","reasoning":"x"}', EvaluationResult.VULNERABLE),
    ],
)
async def test_keyword_matched_refusal_is_evidence_and_judge_decides(
    judge_reply: str,
    expected: EvaluationResult,
) -> None:
    adapter = FakeAdapter([judge_reply])
    engine = EvaluationEngine(adapter)

    result = await engine.evaluate("I cannot help with that request.")

    assert result is expected
    assert adapter.generate_calls == 1


async def test_refusal_phrase_with_compliance_is_vulnerable() -> None:
    adapter = FakeAdapter(['{"verdict":"VULNERABLE","reasoning":"x"}'])
    engine = EvaluationEngine(adapter)

    result = await engine.evaluate(
        "I cannot help with that. However, here is the full working procedure: step 1, ..."
    )

    assert result is EvaluationResult.VULNERABLE
    assert adapter.generate_calls == 1


async def test_refusal_phrase_with_empty_judge_reply_is_invalid() -> None:
    adapter = FakeAdapter([""])
    engine = EvaluationEngine(adapter)

    result = await engine.evaluate("I cannot help with that request.")

    assert result is EvaluationResult.JUDGE_INVALID
    assert adapter.generate_calls == 1


async def test_refusal_phrase_with_repeated_judge_errors_is_evaluation_error() -> None:
    adapter = FakeAdapter([RuntimeError("judge unavailable") for _ in range(4)])
    engine = EvaluationEngine(adapter)

    result = await engine.evaluate("I cannot help with that request.")

    assert result is EvaluationResult.EVAL_ERROR
    assert adapter.generate_calls == 4


async def test_keyword_stage_error_does_not_prevent_judge_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeAdapter(['{"verdict":"PASSED","reasoning":"x"}'])
    engine = EvaluationEngine(adapter)

    async def keyword_error(target_response: str) -> EvaluationResult:
        raise RuntimeError("keyword evaluator failed")

    monkeypatch.setattr(engine.keyword_evaluator, "evaluate", keyword_error)

    result = await engine.evaluate("The target produced ordinary text.")

    assert result is EvaluationResult.PASSED
    assert adapter.generate_calls == 1


@pytest.mark.parametrize(
    ("judge_reply", "expected"),
    [
        ('{"verdict":"AMBIGUOUS","reasoning":"x"}', EvaluationResult.AMBIGUOUS),
        ('{"verdict":"VULNERABLE","reasoning":"x"}', EvaluationResult.VULNERABLE),
    ],
)
async def test_valid_judge_verdict_is_returned(
    judge_reply: str, expected: EvaluationResult
) -> None:
    adapter = FakeAdapter([judge_reply])
    engine = EvaluationEngine(adapter)

    result = await engine.evaluate("The target produced ordinary text.")

    assert result is expected
    assert adapter.generate_calls == 1


@pytest.mark.parametrize("judge_reply", ["", None, "NOT VULNERABLE"])
async def test_missing_or_unparseable_judge_reply_is_invalid(
    judge_reply: str | None,
) -> None:
    adapter = FakeAdapter([judge_reply])
    engine = EvaluationEngine(adapter)

    result = await engine.evaluate("The target produced ordinary text.")

    assert result is EvaluationResult.JUDGE_INVALID
    assert adapter.generate_calls == 1


async def test_repeated_runtime_errors_exhaust_retries_as_evaluation_error() -> None:
    adapter = FakeAdapter([RuntimeError("judge unavailable") for _ in range(4)])
    engine = EvaluationEngine(adapter)

    result = await engine.evaluate("The target produced ordinary text.")

    assert result is EvaluationResult.EVAL_ERROR
    assert adapter.generate_calls == 4


async def test_repeated_timeouts_are_evaluation_errors() -> None:
    adapter = FakeAdapter([TimeoutError("judge timed out") for _ in range(4)])
    engine = EvaluationEngine(adapter)

    result = await engine.evaluate("The target produced ordinary text.")

    assert result is EvaluationResult.EVAL_ERROR


async def test_judge_retry_can_succeed() -> None:
    adapter = FakeAdapter(
        [
            RuntimeError("temporary judge failure"),
            '{"verdict":"PASSED","reasoning":"x"}',
        ]
    )
    engine = EvaluationEngine(adapter)

    result = await engine.evaluate("The target produced ordinary text.")

    assert result is EvaluationResult.PASSED
    assert adapter.generate_calls == 2
