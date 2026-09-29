"""Subprocess tests for settings paths and import side effects."""

import os
import subprocess
import sys
from pathlib import Path

RESULTS_ENV = "LLM_GUARD_BENCH_RESULTS_DIR"
SETTINGS_SCRIPT = (
    "from llm_guard_bench.settings import RESULTS_DIR, DB_PATH\n"
    "print(RESULTS_DIR)\n"
    "print(DB_PATH)\n"
)


def _environment(results_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop(RESULTS_ENV, None)
    if results_dir is not None:
        env[RESULTS_ENV] = str(results_dir)
    return env


def _load_paths(tmp_path: Path, env: dict[str, str]) -> tuple[Path, Path]:
    completed = subprocess.run(
        [sys.executable, "-c", SETTINGS_SCRIPT],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    results_dir, database_path = completed.stdout.splitlines()
    return Path(results_dir).resolve(), Path(database_path).resolve()


def test_default_results_dir_is_working_directory_results(tmp_path: Path) -> None:
    results_dir, database_path = _load_paths(tmp_path, _environment())

    assert results_dir == (tmp_path / "results").resolve()
    assert database_path == (results_dir / "guard_bench.db").resolve()


def test_results_dir_can_be_overridden_by_environment(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom"
    results_dir, database_path = _load_paths(tmp_path, _environment(custom_dir))

    assert results_dir == custom_dir.resolve()
    assert database_path.parent == results_dir


def test_importing_settings_creates_no_directories(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom"
    _load_paths(tmp_path, _environment(custom_dir))

    assert not custom_dir.exists()
    assert not (tmp_path / "results").exists()
