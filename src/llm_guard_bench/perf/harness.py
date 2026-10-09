"""Command-line and RSS sampling utilities for synthetic benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import math
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from types import TracebackType

from llm_guard_bench.perf.synthetic import current_rss_bytes


def _positive_int(value: str) -> int:
    """Parse an integer argument that must be greater than zero."""
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid positive integer: {value}") from exc

    if parsed_value < 1:
        raise argparse.ArgumentTypeError(f"value must be at least 1: {value}")
    return parsed_value


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse synthetic benchmark command-line arguments."""
    parser = argparse.ArgumentParser(prog="bench_synthetic")
    parser.add_argument("--count", type=_positive_int, required=True)
    parser.add_argument("--concurrency", type=_positive_int, default=1)
    return parser.parse_args(argv)


class PeakRssSampler(AbstractAsyncContextManager["PeakRssSampler"]):
    """Sample resident memory and track its peak during an async operation."""

    def __init__(
        self,
        reader: Callable[[], int] = current_rss_bytes,
        interval_seconds: float = 0.05,
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and greater than zero")
        self._reader = reader
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self.samples = 0
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> PeakRssSampler:
        self._sample()
        self._task = asyncio.create_task(self._sample_periodically())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
            self._task = None
        self._sample()

    def _sample(self) -> None:
        rss_bytes = self._reader()
        self.samples += 1
        self.peak_bytes = max(self.peak_bytes, rss_bytes)

    async def _sample_periodically(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            self._sample()
