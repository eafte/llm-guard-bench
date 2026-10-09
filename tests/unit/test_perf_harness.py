"""Tests for synthetic performance-harness utilities."""

import argparse
import asyncio
from itertools import count

import pytest
from llm_guard_bench.perf.harness import PeakRssSampler, parse_args


@pytest.mark.parametrize(
    ("argv", "expected_count", "expected_concurrency"),
    [
        (["--count", "1000"], 1000, 1),
        (["--count", "5", "--concurrency", "3"], 5, 3),
    ],
)
def test_parse_args_accepts_valid_values(
    argv: list[str],
    expected_count: int,
    expected_concurrency: int,
) -> None:
    args = parse_args(argv)

    assert isinstance(args, argparse.Namespace)
    assert args.count == expected_count
    assert args.concurrency == expected_concurrency


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--count", "0"),
        ("--count", "-1"),
        ("--count", "abc"),
        ("--concurrency", "0"),
        ("--concurrency", "-1"),
        ("--concurrency", "abc"),
    ],
)
def test_parse_args_rejects_invalid_positive_integer(
    option: str,
    value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--count", "1", option, value])

    assert exc_info.value.code == 2
    assert f"argument {option}" in capsys.readouterr().err


def test_parse_args_requires_count(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args([])

    assert exc_info.value.code == 2
    assert "--count" in capsys.readouterr().err


async def test_sampler_tracks_peak_and_takes_final_sample() -> None:
    readings = iter([10, 50, 30])
    sampler = PeakRssSampler(
        reader=lambda: next(readings, 30),
        interval_seconds=0.001,
    )

    async with sampler:
        await asyncio.sleep(0.004)

    assert sampler.peak_bytes == 50
    assert sampler.samples >= 2


@pytest.mark.parametrize("interval_seconds", [0, -1])
def test_sampler_rejects_non_positive_interval(interval_seconds: float) -> None:
    with pytest.raises(ValueError):
        PeakRssSampler(interval_seconds=interval_seconds)


async def test_sampler_background_task_is_finished_after_exit() -> None:
    tasks_before = asyncio.all_tasks()
    sampler = PeakRssSampler(reader=lambda: 42, interval_seconds=0.001)

    async with sampler:
        await asyncio.sleep(0.003)

    assert asyncio.all_tasks() == tasks_before


async def test_sampler_takes_final_sample_and_propagates_body_error() -> None:
    reading_count = count()
    sampler = PeakRssSampler(
        reader=lambda: next(reading_count),
        interval_seconds=0.001,
    )

    with pytest.raises(RuntimeError, match="body failed"):
        async with sampler:
            await asyncio.sleep(0.003)
            raise RuntimeError("body failed")

    assert sampler.samples >= 2


async def test_sampler_default_reader_smoke() -> None:
    sampler = PeakRssSampler(interval_seconds=0.001)

    async with sampler:
        await asyncio.sleep(0.002)

    assert sampler.peak_bytes > 0
    assert sampler.samples >= 1
