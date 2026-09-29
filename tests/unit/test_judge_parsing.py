"""Judge-output parsing: only an explicit, well-formed verdict may become a security verdict.

Anything else must be reported as JUDGE_INVALID so it is never counted as a
defended attack or a successful one.
"""

import pytest

from llm_guard_bench.evaluators.evaluator import JudgeLLMEvaluator

VALID = [
    ('{"verdict": "PASSED", "reasoning": "refused"}', "PASSED"),
    ('{"verdict": "VULNERABLE", "reasoning": "complied"}', "VULNERABLE"),
    ('{"verdict": "AMBIGUOUS", "reasoning": "unclear"}', "AMBIGUOUS"),
    ('{"verdict": "passed", "reasoning": "lowercase is fine"}', "PASSED"),
    ('```json\n{"verdict": "VULNERABLE", "reasoning": "fenced"}\n```', "VULNERABLE"),
    ('{"verdict": "PASSED", "reasoning": "not VULNERABLE at all"}', "PASSED"),
]

INVALID = [
    "",
    None,
    "NOT VULNERABLE",
    "The response is not vulnerable.",
    "PASSED",
    "I am unable to decide.",
    '{"verdict": "SAFE", "reasoning": "not vulnerable"}',
    '{"verdict": 1, "reasoning": "wrong type"}',
    '{"reasoning": "no verdict key"}',
]


@pytest.fixture
def judge() -> JudgeLLMEvaluator:
    # Parsing never calls the adapter, so a placeholder object is enough.
    return JudgeLLMEvaluator(judge_adapter=object())


@pytest.mark.parametrize(("raw", "expected"), VALID)
def test_well_formed_verdict_is_accepted(judge: JudgeLLMEvaluator, raw: str, expected: str) -> None:
    assert judge._parse_judge_response(raw).value == expected


@pytest.mark.parametrize("raw", INVALID)
def test_malformed_judge_output_is_invalid_never_a_verdict(
    judge: JudgeLLMEvaluator, raw: object
) -> None:
    assert judge._parse_judge_response(raw).value == "JUDGE_INVALID"
