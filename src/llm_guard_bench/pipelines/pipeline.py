"""
Core benchmark pipeline orchestrator.

Executes adversarial attacks against target LLM with concurrent rate limiting,
captures responses, evaluates them through a 2-stage cascading evaluator, and
persists results to both SQLite and JSONL.
"""

import asyncio
import logging
import time

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
    ) -> None:
        """
        Initialize the benchmark pipeline.

        Args:
            target_adapter: BaseAdapter for target model inference
            judge_adapter: BaseAdapter for evaluation model inference
            db_manager: Database manager for persisting results
        """
        self.target_adapter = target_adapter
        self.judge_adapter = judge_adapter
        self.db_manager = db_manager
        self.evaluation_engine = EvaluationEngine(judge_adapter=judge_adapter)
        self.logger = logging.getLogger(self.__class__.__name__)

    async def run_benchmark(
        self,
        attacks: list[AttackDefinition],
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
        if concurrency_limit < 1:
            raise ValueError("concurrency_limit must be at least 1")

        if not attacks:
            return []

        worker_count = min(concurrency_limit, len(attacks))
        queue: asyncio.Queue[tuple[int, AttackDefinition] | None] = asyncio.Queue(
            maxsize=2 * worker_count
        )
        results: list[tuple[int, TestResult]] = []

        async def _produce() -> None:
            for index, attack in enumerate(attacks):
                await queue.put((index, attack))
            for _ in range(worker_count):
                await queue.put(None)

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
                    continue

                if isinstance(result, TestResult):
                    results.append((index, result))

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

        ordered_results = [
            result for _, result in sorted(results, key=lambda indexed_result: indexed_result[0])
        ]

        self.logger.info(
            f"Benchmark completed: {len(ordered_results)} results from {len(attacks)} attacks"
        )
        return ordered_results

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
                # Defensive pacing: Wait 1.0s before Stage 2 to respect Groq free tier RPM limits
                await asyncio.sleep(1.0)

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
