"""Run a synthetic benchmark using temporary SQLite and JSONL storage."""

from __future__ import annotations

import asyncio
import platform
import shutil
import sys
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from llm_guard_bench.domain.models import SessionSummary
from llm_guard_bench.perf.harness import PeakRssSampler, parse_args
from llm_guard_bench.perf.synthetic import generate_attacks
from llm_guard_bench.pipelines.pipeline import BenchmarkPipeline
from llm_guard_bench.providers.adapters import BaseAdapter
from llm_guard_bench.storage.db import DatabaseManager


class _SyntheticTargetAdapter(BaseAdapter):
    """Return a fixed refusal response without external model calls."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        override_temperature: float | None = None,
    ) -> str:
        return "I cannot help with that request."

    async def generate_multi_turn(
        self,
        messages: list[dict[str, str]],
        override_temperature: float | None = None,
    ) -> str:
        return "I cannot help with that request."


class _SyntheticJudgeAdapter(BaseAdapter):
    """Return a fixed passing judgment without external model calls."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        override_temperature: float | None = None,
    ) -> str:
        return '{"verdict": "PASSED", "reasoning": "synthetic"}'

    async def generate_multi_turn(
        self,
        messages: list[dict[str, str]],
        override_temperature: float | None = None,
    ) -> str:
        return '{"verdict": "PASSED", "reasoning": "synthetic"}'


async def main() -> int:
    """Run the benchmark and report measured environment and resource values."""
    args = parse_args(sys.argv[1:])
    temporary_directory = Path(tempfile.mkdtemp(prefix="llm-guard-bench-synthetic-"))
    db_path = temporary_directory / "bench.db"
    results_dir = temporary_directory
    session_id = str(uuid.uuid4())
    db_manager = DatabaseManager(db_path=db_path, results_dir=results_dir)

    try:
        try:
            await db_manager.initialize()
            await db_manager.connect()
            await db_manager.upsert_session(
                SessionSummary(
                    session_id=session_id,
                    started_at=datetime.now(UTC),
                    config_snapshot={
                        "target": "synthetic",
                        "judge": "synthetic",
                        "concurrency": args.concurrency,
                        "categories": [],
                    },
                )
            )
            for attack in generate_attacks(args.count):
                await db_manager.upsert_attack_definition(attack)

            pipeline = BenchmarkPipeline(
                target_adapter=_SyntheticTargetAdapter(),
                judge_adapter=_SyntheticJudgeAdapter(),
                db_manager=db_manager,
                evaluation_delay_seconds=0,
            )
            async with PeakRssSampler() as sampler:
                started_at = time.perf_counter()
                summary = await pipeline.run_benchmark_summary(
                    attacks=generate_attacks(args.count),
                    model_name="synthetic",
                    concurrency_limit=args.concurrency,
                    session_id=session_id,
                )
                elapsed_seconds = time.perf_counter() - started_at
        finally:
            await db_manager.disconnect()

        jsonl_path = results_dir / f"session_{session_id}.jsonl"
        jsonl_size_mib = jsonl_path.stat().st_size / (1024 * 1024)
        db_size_mib = db_path.stat().st_size / (1024 * 1024)
        results_per_second = summary.results_written / elapsed_seconds

        print(f"Python version: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"Count: {args.count}")
        print(f"Concurrency: {args.concurrency}")
        print(f"Attacks pulled: {summary.attacks_pulled}")
        print(f"Results written: {summary.results_written}")
        print(f"Errors: {summary.errors}")
        print(f"Status counts: {dict(summary.status_counts)}")
        print(f"Elapsed seconds: {elapsed_seconds:.6f}")
        print(f"Results per second: {results_per_second:.6f}")
        print(f"Peak RSS (MiB): {sampler.peak_bytes / (1024 * 1024):.3f}")
        print(f"RSS samples: {sampler.samples}")
        print(f"JSONL size (MiB): {jsonl_size_mib:.6f}")
        print(f"DB size (MiB): {db_size_mib:.6f}")
        print("Single-run numbers from this machine only.")
        return int(summary.errors > 0 or summary.results_written != args.count)
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
