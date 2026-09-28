# core/models.py
# All inter-module data structures defined as immutable Pydantic models.
# No raw dicts are passed between pipeline stages — only typed model instances.
#
# v3.0 changes:
#   - TestResult.target_model_name  → TestResult.model_name
#   - TestResult.attack_category    → TestResult.category
#   - AttackDefinition.attack_category → AttackDefinition.category
#   - factory method from_attack_and_eval updated accordingly

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field
from typing import List, Literal
from pydantic import BaseModel, model_validator


# ---------------------------------------------------------------------------
# Shared type aliases
# ---------------------------------------------------------------------------
EvaluationStatus = Literal[
    "PASSED",
    "VULNERABLE",
    "FAILED",      # judge ran but returned unparseable output
    "EVAL_ERROR",  # Python/network exception in pipeline or evaluator
    "TIMEOUT",     # target LLM API timed out (> REQUEST_TIMEOUT_SECONDS)
    "SKIPPED",     # token budget pre-flight blocked execution
]

EvaluationStage = Literal[
    "STAGE_1_KEYWORD",  # keyword matching was decisive
    "STAGE_2_JUDGE",    # judge LLM was invoked
    "PRE_FLIGHT",       # evaluation never started (SKIPPED/TIMEOUT/EVAL_ERROR)
]

AttackCategoryType = Literal[
    "DAN",
    "ROLEPLAY_EXPLOIT",
    "PROMPT_INJECTION",
    "SYSTEM_PROMPT_LEAKAGE",
    "DO_NOT_REFUSE",
]

SeverityType = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ---------------------------------------------------------------------------
# AttackDefinition — loaded from config/prompts.json by core/loader.py
# ---------------------------------------------------------------------------

class AttackDefinition(BaseModel):
    attack_id: str
    attack_name: str
    description: str
    category: str
    # Support both uppercase and lowercase severity values
    severity: Literal['LOW', 'MEDIUM', 'HIGH', 'CRITICAL', 'low', 'medium', 'high', 'critical']
    tags: List[str]
    turns: List[str]  # Sequential conversation turns for multi-turn attack evaluation
    system_prompt: Optional[str] = None  # Optional system prompt for context

    @property
    def adversarial_prompt(self) -> str:
        """
        Construct a single string representation of the multi-turn conversation
        by concatenating all turns with newlines. Used for storage in TestResult.
        """
        return "\n".join(self.turns) if self.turns else ""

    def build_messages(self) -> list[dict[str, str]]:
        """
        Build a multi-turn message history for chat-based adapters.
        Returns a list of dicts with 'role' and 'content' keys.
        """
        messages = []
        # Add system prompt if present
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        # Add user turns alternating with assistant context (for realism in multi-turn)
        for i, turn in enumerate(self.turns):
            if i == 0:
                # First turn is always a user message
                messages.append({"role": "user", "content": turn})
            else:
                # Alternate: odd indices are user, even indices could be assistant for context
                # In this implementation, all turns are user messages (the model's responses aren't in the turns list)
                messages.append({"role": "user", "content": turn})
        return messages


# ---------------------------------------------------------------------------
# EvalResult — internal return type of EvaluationEngine.evaluate()
# Never written directly to storage; merged into TestResult by pipeline.py.
# ---------------------------------------------------------------------------
class EvalResult(BaseModel, frozen=True):
    status:            EvaluationStatus
    stage:             EvaluationStage
    judge_verdict:     Optional[Literal["PASSED", "VULNERABLE"]] = None
    judge_parse_error: bool                                       = False
    error_message:     Optional[str]                             = None

    @model_validator(mode="after")
    def _validate_consistency(self) -> "EvalResult":
        if self.stage == "STAGE_1_KEYWORD" and self.judge_verdict is not None:
            raise ValueError(
                "judge_verdict must be None when stage is STAGE_1_KEYWORD"
            )
        if self.status == "FAILED" and not self.judge_parse_error:
            raise ValueError(
                "judge_parse_error must be True when status is FAILED"
            )
        return self


# ---------------------------------------------------------------------------
# TestResult — the canonical record persisted to SQLite and JSONL
# ---------------------------------------------------------------------------
class TestResult(BaseModel, frozen=True):
    schema_version:    str            = "3.0"
    session_id:        str
    timestamp:         datetime
    model_name:        str            # v3.0: was target_model_name
    attack_id:         str
    category:          AttackCategoryType  # v3.0: was attack_category
    adversarial_prompt: str
    system_prompt:     Optional[str]
    raw_llm_response:  str            = ""
    # Empty string (never None) when no response could be obtained.

    evaluation_status:  EvaluationStatus
    evaluation_stage:   EvaluationStage
    judge_verdict:      Optional[Literal["PASSED", "VULNERABLE"]] = None
    judge_parse_error:  bool          = False
    execution_time_ms:  int           = 0
    total_time_ms:      int           = 0
    prompt_tokens:      Optional[int] = None
    completion_tokens:  Optional[int] = None
    http_status_code:   Optional[int] = None
    error_message:      Optional[str] = None

    @classmethod
    def from_attack_and_eval(
        cls,
        *,
        session_id:        str,
        attack:            AttackDefinition,
        model_name:        str,
        raw_response:      str,
        eval_result:       EvalResult,
        execution_time_ms: int,
        total_time_ms:     int,
        prompt_tokens:     Optional[int] = None,
        completion_tokens: Optional[int] = None,
        http_status_code:  Optional[int] = None,
        adapter_error:     Optional[str] = None,
    ) -> "TestResult":
        """
        Factory method. The only sanctioned way to create a TestResult in pipeline.py.
        Merges attack metadata, adapter response data, and eval verdict into one record.
        """
        return cls(
            session_id         = session_id,
            timestamp          = datetime.now(timezone.utc),
            model_name         = model_name,
            attack_id          = attack.attack_id,
            category           = attack.category,
            adversarial_prompt = attack.adversarial_prompt,
            system_prompt      = attack.system_prompt,
            raw_llm_response   = raw_response,
            evaluation_status  = eval_result.status,
            evaluation_stage   = eval_result.stage,
            judge_verdict      = eval_result.judge_verdict,
            judge_parse_error  = eval_result.judge_parse_error,
            execution_time_ms  = execution_time_ms,
            total_time_ms      = total_time_ms,
            prompt_tokens      = prompt_tokens,
            completion_tokens  = completion_tokens,
            http_status_code   = http_status_code,
            error_message      = eval_result.error_message or adapter_error,
        )


# ---------------------------------------------------------------------------
# SessionSummary — written to the sessions table; counters mutated during run
# ---------------------------------------------------------------------------
class SessionSummary(BaseModel):
    # Not frozen: increment() mutates counters in place.
    session_id:       str
    started_at:       datetime
    finished_at:      Optional[datetime] = None
    config_snapshot:  dict
    total_tests:      int = 0
    passed_count:     int = 0
    vulnerable_count: int = 0
    failed_count:     int = 0
    eval_error_count: int = 0
    timeout_count:    int = 0
    skipped_count:    int = 0

    def increment(self, status: EvaluationStatus) -> None:
        """Mutates the appropriate counter for the given evaluation status."""
        self.total_tests += 1
        match status:
            case "PASSED":     self.passed_count     += 1
            case "VULNERABLE": self.vulnerable_count += 1
            case "FAILED":     self.failed_count     += 1
            case "EVAL_ERROR": self.eval_error_count += 1
            case "TIMEOUT":    self.timeout_count    += 1
            case "SKIPPED":    self.skipped_count    += 1

    @property
    def attack_success_rate(self) -> float:
        """
        ASR = VULNERABLE / (PASSED + VULNERABLE) * 100.
        Returns 0.0 if no decisive results exist.
        FAILED, EVAL_ERROR, TIMEOUT, SKIPPED are excluded from the denominator.
        """
        decisive = self.passed_count + self.vulnerable_count
        if decisive == 0:
            return 0.0
        return (self.vulnerable_count / decisive) * 100.0
