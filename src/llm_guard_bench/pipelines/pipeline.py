"""
Core benchmark pipeline orchestrator.

Executes adversarial attacks against target LLM with concurrent rate limiting,
captures responses, evaluates them through a 2-stage cascading evaluator, and
persists results to both SQLite and JSONL.
"""

import asyncio
import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sized
from dataclasses import dataclass

from llm_guard_bench.domain.models import (
    AttackContext,
    AttackDefinition,
    EvalResult,
    TestResult,
)
from llm_guard_bench.evaluators.evaluator import EvaluationEngine, EvaluationResult
from llm_guard_bench.providers.adapters import BaseAdapter
from llm_guard_bench.storage.errors import StorageError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkSummary:
    """Bounded aggregate counts from a benchmark run."""

    attacks_pulled: int
    results_written: int
    errors: int
    status_counts: Mapping[str, int]


class BenchmarkPipeline:
    """
    Production-ready async benchmark pipeline for executing attacks against target and judge adapters.
    Implements rate limiting, resilience, database persistence, and comprehensive error handling.
    """

    def __init__(
        self,
        target_adapter: BaseAdapter,
        judge_adapter: BaseAdapter,
        db_manager,
        evaluation_delay_seconds: float = 1.0,
    ) -> None:
        """
        Initialize the benchmark pipeline.

        Args:
            target_adapter: BaseAdapter for target model inference
            judge_adapter: BaseAdapter for evaluation model inference
            db_manager: Database manager for persisting results
            evaluation_delay_seconds: Per-attack delay before judge evaluation
        """
        if not math.isfinite(evaluation_delay_seconds) or evaluation_delay_seconds < 0:
            raise ValueError("evaluation_delay_seconds must be finite and non-negative")

        self.target_adapter = target_adapter
        self.judge_adapter = judge_adapter
        self.db_manager = db_manager
        self.evaluation_delay_seconds = evaluation_delay_seconds
        self.evaluation_engine = EvaluationEngine(judge_adapter=judge_adapter)
        self.logger = logging.getLogger(self.__class__.__name__)

    async def run_benchmark(
        self,
        attacks: Iterable[AttackDefinition],
        model_name: str,
        concurrency_limit: int,
        session_id: str,
    ) -> list[TestResult]:
        """
        Execute benchmark across multiple attack vectors with concurrency control.

        Args:
            attacks: List of AttackDefinition objects to execute
            model_name: Target model identifier
            concurrency_limit: Maximum number of concurrent executions
            session_id: Unique session identifier for batch tracking

        Returns:
            List of TestResult objects containing execution and evaluation metrics
        """
        results: list[tuple[int, TestResult]] = []
        pulled_count = await self._run_benchmark_workers(
            attacks=attacks,
            model_name=model_name,
            concurrency_limit=concurrency_limit,
            session_id=session_id,
            on_result=lambda index, result: results.append((index, result)),
            on_error=lambda: None,
        )
        if pulled_count is None:
            return []

        ordered_results = [
            result for _, result in sorted(results, key=lambda indexed_result: indexed_result[0])
        ]

        self.logger.info(
            f"Benchmark completed: {len(ordered_results)} results from {pulled_count} attacks"
        )
        return ordered_results

    async def run_benchmark_summary(
        self,
        attacks: Iterable[AttackDefinition],
        model_name: str,
        concurrency_limit: int,
        session_id: str,
    ) -> BenchmarkSummary:
        """Run a benchmark while retaining only bounded result counts."""
        status_counts: dict[str, int] = {}
        results_written = 0
        errors = 0

        def record_result(_index: int, result: TestResult) -> None:
            nonlocal results_written
            results_written += 1
            status = result.evaluation_status
            status_counts[status] = status_counts.get(status, 0) + 1

        def record_error() -> None:
            nonlocal errors
            errors += 1

        pulled_count = await self._run_benchmark_workers(
            attacks=attacks,
            model_name=model_name,
            concurrency_limit=concurrency_limit,
            session_id=session_id,
            on_result=record_result,
            on_error=record_error,
        )
        if pulled_count is None:
            pulled_count = 0

        self.logger.info(
            "Benchmark summary completed: %s results from %s attacks (%s errors)",
            results_written,
            pulled_count,
            errors,
        )
        return BenchmarkSummary(
            attacks_pulled=pulled_count,
            results_written=results_written,
            errors=errors,
            status_counts=status_counts,
        )

    async def _run_benchmark_workers(
        self,
        attacks: Iterable[AttackDefinition],
        model_name: str,
        concurrency_limit: int,
        session_id: str,
        on_result: Callable[[int, TestResult], None],
        on_error: Callable[[], None],
    ) -> int | None:
        """Execute attacks with bounded workers and report outcomes through callbacks."""
        if concurrency_limit < 1:
            raise ValueError("concurrency_limit must be at least 1")

        if isinstance(attacks, Sized):
            attack_count = len(attacks)
            if attack_count == 0:
                return None
            worker_count = min(concurrency_limit, attack_count)
        else:
            worker_count = concurrency_limit

        queue: asyncio.Queue[tuple[int, AttackDefinition] | None] = asyncio.Queue(
            maxsize=2 * worker_count
        )
        pulled_count = 0

        async def _produce() -> None:
            nonlocal pulled_count
            iterator = iter(attacks)
            try:
                for index, attack in enumerate(iterator):
                    pulled_count += 1
                    await queue.put((index, attack))
                for _ in range(worker_count):
                    await queue.put(None)
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()

        async def _work() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    return

                index, attack = item
                try:
                    result = await self._execute_single_test(
                        attack=attack,
                        model_name=model_name,
                        attack_index=index,
                        session_id=session_id,
                    )
                except StorageError:
                    raise
                except Exception as exc:
                    self.logger.error(
                        "Attack %s (%s) unhandled exception in worker: %s: %s",
                        index,
                        attack.attack_id,
                        type(exc).__name__,
                        exc,
                    )
                    on_error()
                    continue

                if isinstance(result, TestResult):
                    on_result(index, result)

        def _find_storage_error(error: BaseException) -> StorageError | None:
            if isinstance(error, StorageError):
                return error
            if isinstance(error, BaseExceptionGroup):
                for nested_error in error.exceptions:
                    storage_error = _find_storage_error(nested_error)
                    if storage_error is not None:
                        return storage_error
            return None

        try:
            async with asyncio.TaskGroup() as task_group:
                task_group.create_task(_produce())
                for _ in range(worker_count):
                    task_group.create_task(_work())
        except BaseExceptionGroup as error_group:
            storage_error = _find_storage_error(error_group)
            if storage_error is not None:
                self.logger.error("Benchmark encountered persistence failures")
                raise storage_error
            raise

        return pulled_count

    async def _execute_single_test(
        self,
        attack: AttackDefinition,
        model_name: str,
        attack_index: int,
        session_id: str,
        timeout_seconds: float = 180.0,
    ) -> TestResult | None:
        """
        Execute a single benchmark test with full instrumentation and error handling.

        Captures timing metrics, executes target model, evaluates output, and persists to database.
        Uses asyncio.wait_for timeout to prevent indefinite hangs.

        Args:
            attack: AttackDefinition to execute
            model_name: Target model identifier
            attack_index: Index of attack in batch
            session_id: Session identifier for batch tracking
            timeout_seconds: Maximum seconds to wait for target model response

        Returns:
            TestResult object with complete execution and evaluation data, or None on fatal error
        """
        start_time = time.time()
        target_output = ""
        execution_time_ms = 0
        total_time_ms = 0
        eval_result: EvalResult | None = None

        # ===== Pre-flight: Target Provider Health Check =====
        health_start = time.time()
        try:
            target_healthy = await self.target_adapter.health_check()
            if not target_healthy:
                execution_time_ms = int((time.time() - health_start) * 1000)
                eval_result = EvalResult(
                    status="EVAL_ERROR",
                    stage="PRE_FLIGHT",
                    error_message=(
                        "Target model health check failed after connection "
                        "retries; Ollama endpoint remained unreachable"
                    ),
                )
        except Exception as e:
            execution_time_ms = int((time.time() - health_start) * 1000)
            self.logger.error(
                f"Attack {attack.attack_id}: Target health check error: "
                f"{type(e).__name__}: {str(e)}"
            )
            eval_result = EvalResult(
                status="EVAL_ERROR",
                stage="PRE_FLIGHT",
                error_message=(f"Target model health check error: {type(e).__name__}: {str(e)}"),
            )

        # ===== Stage 1: Target Model Execution =====
        if eval_result is None:
            try:
                target_start = time.time()
                try:
                    # Build multi-turn message history from attack definition
                    messages = attack.build_messages()
                    target_output = await asyncio.wait_for(
                        self.target_adapter.generate_multi_turn(messages=messages),
                        timeout=180.0,
                    )
                except TimeoutError:
                    execution_time_ms = int((time.time() - target_start) * 1000)
                    self.logger.warning(
                        f"Attack {attack.attack_id}: Target model timeout after {180.0}s"
                    )
                    eval_result = EvalResult(
                        status="TIMEOUT",
                        stage="PRE_FLIGHT",
                        error_message=f"Target model timeout after {180.0}s",
                    )

                target_end = time.time()
                if eval_result is None:
                    execution_time_ms = int((target_end - target_start) * 1000)
                    if not target_output:
                        target_output = ""
                    self.logger.debug(
                        f"Attack {attack.attack_id}: Target model response ({execution_time_ms}ms)"
                    )

            except Exception as e:
                execution_time_ms = int((time.time() - start_time) * 1000)
                self.logger.error(
                    f"Attack {attack.attack_id}: Target model error: {type(e).__name__}: {str(e)}"
                )
                eval_result = EvalResult(
                    status="EVAL_ERROR",
                    stage="PRE_FLIGHT",
                    error_message=f"Target model error: {type(e).__name__}: {str(e)}",
                )

        # ===== Stage 2: Evaluation Engine (if target succeeded) =====
        if eval_result is None:
            try:
                # Apply per-attack pacing before calling the judge.
                if self.evaluation_delay_seconds > 0:
                    await asyncio.sleep(self.evaluation_delay_seconds)

                eval_start = time.time()
                try:
                    evaluation_result: EvaluationResult = await asyncio.wait_for(
                        self.evaluation_engine.evaluate(
                            target_output, context=AttackContext.from_attack(attack)
                        ),
                        timeout=180.0,
                    )
                    eval_end = time.time()
                    self.logger.debug(
                        f"Attack {attack.attack_id}: Evaluation completed ({int((eval_end - eval_start) * 1000)}ms)"
                    )
                    eval_result = self._convert_evaluation_result(evaluation_result)

                except TimeoutError:
                    self.logger.warning(
                        f"Attack {attack.attack_id}: Evaluation timeout after {180.0}s"
                    )
                    eval_result = EvalResult(
                        status="EVAL_ERROR",
                        stage="STAGE_2_JUDGE",
                        error_message=f"Evaluation timeout after {180.0}s",
                    )

            except Exception as e:
                self.logger.error(
                    f"Attack {attack.attack_id}: Evaluation error: {type(e).__name__}: {str(e)}"
                )
                eval_result = EvalResult(
                    status="EVAL_ERROR",
                    stage="STAGE_2_JUDGE",
                    error_message=f"Evaluation error: {type(e).__name__}: {str(e)}",
                )

        # Ensure eval_result is set (should never be None at this point)
        if eval_result is None:
            eval_result = EvalResult(
                status="EVAL_ERROR",
                stage="PRE_FLIGHT",
                error_message="Unknown evaluation error",
            )

        # ===== Construct Unified TestResult =====
        total_time_ms = int((time.time() - start_time) * 1000)

        test_result = TestResult.from_attack_and_eval(
            session_id=session_id,
            attack=attack,
            model_name=model_name,
            raw_response=target_output,
            eval_result=eval_result,
            execution_time_ms=execution_time_ms,
            total_time_ms=total_time_ms,
        )

        # ===== Persist to Database =====
        try:
            await self.db_manager.insert_result(test_result)
        except Exception as e:
            self.logger.error(f"Failed to persist result for attack {attack.attack_id}: {str(e)}")
            if isinstance(e, StorageError):
                raise
            raise StorageError(
                f"Failed to persist result for attack {attack.attack_id}: {e}"
            ) from e

        return test_result

    def _convert_evaluation_result(self, evaluation_result: EvaluationResult) -> EvalResult:
        """
        Convert EvaluationEngine's EvaluationResult enum to EvalResult model.

        Maps EvaluationResult string values to appropriate EvalResult with status,
        stage, and error context. Ambiguous verdicts remain distinct, while
        unparseable judge output is marked as a parse failure.

        Args:
            evaluation_result: Result from EvaluationEngine.evaluate()

        Returns:
            EvalResult with appropriate status and stage
        """
        result_str = evaluation_result.value.upper()

        if result_str == "PASSED":
            return EvalResult(
                status="PASSED",
                stage="STAGE_2_JUDGE",
                judge_verdict="PASSED",
            )
        elif result_str == "VULNERABLE":
            return EvalResult(
                status="VULNERABLE",
                stage="STAGE_2_JUDGE",
                judge_verdict="VULNERABLE",
            )
        elif result_str == "AMBIGUOUS":
            return EvalResult(
                status="AMBIGUOUS",
                stage="STAGE_2_JUDGE",
                judge_verdict="AMBIGUOUS",
            )
        elif result_str == "JUDGE_INVALID":
            return EvalResult(
                status="FAILED",
                stage="STAGE_2_JUDGE",
                judge_parse_error=True,
                error_message="Judge returned unparseable output",
            )
        elif result_str == "EVAL_ERROR":
            # Handle upstream judge LLM failures (rate limits, network errors, etc.)
            return EvalResult(
                status="EVAL_ERROR",
                stage="STAGE_2_JUDGE",
                error_message="Judge LLM evaluation failed due to an upstream API or network exception",
            )
        else:
            return EvalResult(
                status="EVAL_ERROR",
                stage="STAGE_2_JUDGE",
                error_message=f"Unknown evaluation result string received: {result_str}",
            )
