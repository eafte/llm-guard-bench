"""Tests for CLI attack-file selection and its packaged default."""

import json
import sys
from pathlib import Path

import pytest

from llm_guard_bench import cli
from llm_guard_bench.domain.models import AttackDefinition


def _attack(
    attack_id: str,
    *,
    category: str = "DAN",
) -> dict[str, object]:
    return {
        "attack_id": attack_id,
        "attack_name": f"Attack {attack_id}",
        "description": "A test attack.",
        "category": category,
        "severity": "LOW",
        "tags": ["t"],
        "turns": ["hi"],
    }


def _write_attacks_file(
    tmp_path: Path,
    attacks: list[dict[str, object]],
) -> Path:
    file_path = tmp_path / "attacks.json"
    file_path.write_text(json.dumps({"attacks": attacks}), encoding="utf-8")
    return file_path


def test_parser_attacks_file_defaults_to_none_and_accepts_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["llm-guard-bench", "--target", "t"])
    assert cli.parse_arguments().attacks_file is None

    monkeypatch.setattr(
        sys,
        "argv",
        ["llm-guard-bench", "--target", "t", "--attacks-file", "x.json"],
    )
    assert cli.parse_arguments().attacks_file == "x.json"


async def test_async_main_passes_attacks_file_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    class CapturingOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

        async def orchestrate(self) -> None:
            return None

    monkeypatch.setattr(cli, "validate_configuration", lambda: None)
    monkeypatch.setattr(cli, "LLMGuardBenchOrchestrator", CapturingOrchestrator)
    monkeypatch.setattr(
        sys,
        "argv",
        ["llm-guard-bench", "--target", "t", "--attacks-file", "some.json"],
    )

    await cli.async_main()

    assert captured_kwargs["attacks_file"] == "some.json"


async def test_default_attack_file_loads_shipped_prompts_from_empty_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="t",
        judge="j",
        concurrency=1,
    )

    attacks = await orchestrator.load_attack_definitions()

    assert len(attacks) == 10
    assert all(isinstance(attack, AttackDefinition) for attack in attacks)


async def test_explicit_attack_file_loads_entries_in_order_from_other_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "empty-cwd"
    cwd.mkdir()
    file_path = _write_attacks_file(
        tmp_path,
        [_attack("first"), _attack("second")],
    )
    monkeypatch.chdir(cwd)
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="t",
        judge="j",
        concurrency=1,
        attacks_file=str(file_path),
    )

    attacks = await orchestrator.load_attack_definitions()

    assert [attack.attack_id for attack in attacks] == ["first", "second"]


async def test_missing_explicit_attack_file_raises_file_not_found(
    tmp_path: Path,
) -> None:
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="t",
        judge="j",
        concurrency=1,
        attacks_file=str(tmp_path / "missing.json"),
    )

    with pytest.raises(FileNotFoundError):
        await orchestrator.load_attack_definitions()


async def test_explicit_attack_file_respects_category_filter(tmp_path: Path) -> None:
    file_path = _write_attacks_file(
        tmp_path,
        [
            _attack("dan", category="DAN"),
            _attack("roleplay", category="ROLEPLAY_EXPLOIT"),
        ],
    )
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="t",
        judge="j",
        concurrency=1,
        categories=["DAN"],
        attacks_file=str(file_path),
    )

    attacks = await orchestrator.load_attack_definitions()

    assert [attack.attack_id for attack in attacks] == ["dan"]
