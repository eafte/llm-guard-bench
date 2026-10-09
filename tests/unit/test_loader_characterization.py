"""Characterization tests for loading attack prompt definitions."""

import json
from pathlib import Path

import pytest

from llm_guard_bench.domain.models import AttackDefinition
from llm_guard_bench.streaming_io.loader import AttackLoader


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


def _write_json(tmp_path: Path, data: object) -> Path:
    file_path = tmp_path / "attacks.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return file_path


def test_loads_attacks_key_in_file_order(tmp_path: Path) -> None:
    file_path = _write_json(
        tmp_path,
        {"attacks": [_attack("first"), _attack("second")]},
    )

    attacks = AttackLoader.load_prompts(str(file_path))

    assert all(isinstance(attack, AttackDefinition) for attack in attacks)
    assert [attack.attack_id for attack in attacks] == ["first", "second"]


def test_loads_prompts_key(tmp_path: Path) -> None:
    file_path = _write_json(tmp_path, {"prompts": [_attack("prompt-1")]})

    attacks = AttackLoader.load_prompts(str(file_path))

    assert [attack.attack_id for attack in attacks] == ["prompt-1"]


def test_filters_by_category(tmp_path: Path) -> None:
    file_path = _write_json(
        tmp_path,
        {
            "attacks": [
                _attack("dan", category="DAN"),
                _attack("roleplay", category="ROLEPLAY_EXPLOIT"),
            ]
        },
    )

    dan_attacks = AttackLoader.load_prompts(str(file_path), categories=["DAN"])
    unmatched_attacks = AttackLoader.load_prompts(str(file_path), categories=["NOPE"])

    assert [attack.attack_id for attack in dan_attacks] == ["dan"]
    assert unmatched_attacks == []


def test_skips_invalid_entry_and_keeps_valid_neighbours(tmp_path: Path) -> None:
    invalid_attack = _attack("invalid")
    del invalid_attack["turns"]
    file_path = _write_json(
        tmp_path,
        {"attacks": [_attack("before"), invalid_attack, _attack("after")]},
    )

    attacks = AttackLoader.load_prompts(str(file_path))

    assert [attack.attack_id for attack in attacks] == ["before", "after"]


def test_missing_file_raises_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AttackLoader.load_prompts(str(tmp_path / "missing.json"))


def test_zero_byte_file_raises_value_error(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.json"
    file_path.write_bytes(b"")

    with pytest.raises(ValueError):
        AttackLoader.load_prompts(str(file_path))


def test_invalid_json_raises_json_decode_error(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.json"
    file_path.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        AttackLoader.load_prompts(str(file_path))


def test_json_list_top_level_raises_value_error(tmp_path: Path) -> None:
    file_path = _write_json(tmp_path, [_attack("top-level-list")])

    with pytest.raises(ValueError, match="JSON object"):
        AttackLoader.load_prompts(str(file_path))


def test_missing_attacks_and_prompts_keys_raises_value_error(tmp_path: Path) -> None:
    file_path = _write_json(tmp_path, {"other": [_attack("unused")]})

    with pytest.raises(ValueError, match="attacks.*prompts"):
        AttackLoader.load_prompts(str(file_path))


def test_shipped_prompts_resource_contains_ten_attacks() -> None:
    resource_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "llm_guard_bench"
        / "resources"
        / "prompts.json"
    )

    attacks = AttackLoader.load_prompts(str(resource_path))

    assert len(attacks) == 10
