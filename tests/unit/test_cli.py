"""Regression tests for the console entry point."""

import inspect
import subprocess
import sys

from llm_guard_bench import cli


def test_main_is_a_synchronous_function() -> None:
    """The console script calls main() directly, so it must not be a coroutine function."""
    assert not inspect.iscoroutinefunction(cli.main)


def test_async_main_is_a_coroutine_function() -> None:
    """The async orchestration logic lives in async_main()."""
    assert inspect.iscoroutinefunction(cli.async_main)


def test_help_exits_cleanly() -> None:
    """Running the CLI with --help exits 0 and prints usage, with no coroutine warning."""
    result = subprocess.run(
        [sys.executable, "-m", "llm_guard_bench.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert "--target" in result.stdout
    assert "never awaited" not in result.stderr
