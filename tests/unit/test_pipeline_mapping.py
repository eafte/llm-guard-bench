"""Regression tests for mapping evaluator outcomes to pipeline results."""

import pytest

from llm_guard_bench.domain.models import EvalResult
from llm_guard_bench.evaluators.evaluator import EvaluationResult
from llm_guard_bench.pipelines.pipeline import BenchmarkPipeline
from llm_guard_bench.providers.adapters import BaseAdapter


class FakeAdapter(BaseAdapter):
    """Minimal adapter stub for constructing a pipeline without I/O."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        override_temperature: float | None = None,
    ) -> str:
        return ""

    async def generate_multi_turn(
        self,
        messages: list[dict[str, str]],
        override_temperature: float | None = None,
    ) -> str:
        return ""


@pytest.fixture
def pipeline() -> BenchmarkPipeline:
    adapter = FakeAdapter()
    return BenchmarkPipeline(
        target_adapter=adapter,
        judge_adapter=adapter,
        db_manager=object(),
    )


def test_passed_maps_to_passed(pipeline: BenchmarkPipeline) -> None:
    result = pipeline._convert_evaluation_result(EvaluationResult.PASSED)

    assert result.status == "PASSED"


def test_vulnerable_maps_with_judge_verdict(pipeline: BenchmarkPipeline) -> None:
    result = pipeline._convert_evaluation_result(EvaluationResult.VULNERABLE)

    assert result.status == "VULNERABLE"
    assert result.judge_verdict == "VULNERABLE"


def test_ambiguous_remains_ambiguous(pipeline: BenchmarkPipeline) -> None:
    result = pipeline._convert_evaluation_result(EvaluationResult.AMBIGUOUS)

    assert result.status == "AMBIGUOUS"
    assert result.judge_verdict == "AMBIGUOUS"
    assert result.stage == "STAGE_2_JUDGE"


def test_judge_invalid_is_marked_failed_with_parse_error(
    pipeline: BenchmarkPipeline,
) -> None:
    result = pipeline._convert_evaluation_result(EvaluationResult.JUDGE_INVALID)

    assert result.status == "FAILED"
    assert result.judge_parse_error is True
    assert result.judge_verdict is None
    assert result.stage == "STAGE_2_JUDGE"
    assert result.error_message


def test_evaluation_error_maps_to_evaluation_error(pipeline: BenchmarkPipeline) -> None:
    result = pipeline._convert_evaluation_result(EvaluationResult.EVAL_ERROR)

    assert result.status == "EVAL_ERROR"


@pytest.mark.parametrize("evaluation_result", list(EvaluationResult))
def test_every_evaluation_result_converts_without_claiming_missing_evidence_passed(
    pipeline: BenchmarkPipeline,
    evaluation_result: EvaluationResult,
) -> None:
    result: EvalResult = pipeline._convert_evaluation_result(evaluation_result)

    assert (result.status == "PASSED") is (evaluation_result is EvaluationResult.PASSED)
