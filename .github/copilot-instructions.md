# Project rules for llm-guard-bench

- Python 3.11+, `src/` layout, package `llm_guard_bench`, virtual env in `.venv`.
- Write or update a failing test BEFORE changing behavior. Never weaken a test to make it pass.
- Never treat an error, empty response, or unparseable judge output as PASSED. Missing evidence is an error, not a defense.
- No new dependencies. No `# type: ignore`. Do not add entries to the ruff ignore list in pyproject.toml.
- The mypy error count must not increase (current baseline: 10). It may decrease.
- Only edit the files named in the task. Do not rename, move, or delete files.
- No bare `except Exception: pass`. Catch specific exceptions or log with exc_info.
- Keep functions small and typed, with short docstrings. Avoid coupling: depend on the `BaseAdapter` interface, not concrete adapters.
- Never run `git commit`, `git push`, or destructive commands. I review and commit myself.
- Verify with: `python -m pytest tests -q`, `ruff check src tests`, `ruff format --check src tests`, `mypy`.
