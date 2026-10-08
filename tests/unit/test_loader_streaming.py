"""Tests for bounded JSONL attack loading and legacy file-size limits."""

import json
from pathlib import Path

import pytest

from llm_guard_bench.domain.models import AttackDefinition
from llm_guard_bench.streaming_io import loader as loader_module
from llm_guard_bench.streaming_io.loader import AttackLoader, LoadStats


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


def _json_line(attack: dict[str, object]) -> bytes:
    return json.dumps(attack).encode("utf-8") + b"\n"


def test_iter_attacks_yields_jsonl_in_file_order_and_counts_accepted(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "attacks.jsonl"
    file_path.write_bytes(_json_line(_attack("first")) + _json_line(_attack("second")))
    stats = LoadStats()

    attacks = list(AttackLoader.iter_attacks(str(file_path), stats=stats))

    assert all(isinstance(attack, AttackDefinition) for attack in attacks)
    assert [attack.attack_id for attack in attacks] == ["first", "second"]
    assert stats.accepted == 2


def test_iter_attacks_parses_lazily(tmp_path: Path) -> None:
    file_path = tmp_path / "lazy.jsonl"
    file_path.write_bytes(_json_line(_attack("first")) + b"{broken json\n")
    stats = LoadStats()
    attacks = AttackLoader.iter_attacks(str(file_path), stats=stats)

    first_attack = next(attacks)

    assert first_attack.attack_id == "first"
    assert stats.accepted == 1
    assert stats.rejected_invalid == 0


def test_iter_attacks_filters_categories_and_counts_filtered_out(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "categories.jsonl"
    file_path.write_bytes(
        _json_line(_attack("dan", category="DAN"))
        + _json_line(_attack("roleplay", category="ROLEPLAY_EXPLOIT"))
    )
    stats = LoadStats()

    attacks = list(AttackLoader.iter_attacks(str(file_path), categories=["DAN"], stats=stats))

    assert [attack.attack_id for attack in attacks] == ["dan"]
    assert stats.accepted == 1
    assert stats.filtered_out == 1


def test_iter_attacks_skips_invalid_json_between_valid_lines(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid-json.jsonl"
    file_path.write_bytes(
        _json_line(_attack("before")) + b"{broken json\n" + _json_line(_attack("after"))
    )
    stats = LoadStats()

    attacks = list(AttackLoader.iter_attacks(str(file_path), stats=stats))

    assert [attack.attack_id for attack in attacks] == ["before", "after"]
    assert stats.rejected_invalid == 1


def test_iter_attacks_counts_model_validation_failure_as_invalid(tmp_path: Path) -> None:
    invalid_attack = _attack("invalid")
    del invalid_attack["turns"]
    file_path = tmp_path / "invalid-model.jsonl"
    file_path.write_bytes(_json_line(invalid_attack) + _json_line(_attack("valid")))
    stats = LoadStats()

    attacks = list(AttackLoader.iter_attacks(str(file_path), stats=stats))

    assert [attack.attack_id for attack in attacks] == ["valid"]
    assert stats.rejected_invalid == 1


def test_iter_attacks_counts_invalid_utf8_and_keeps_neighbours(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid-utf8.jsonl"
    file_path.write_bytes(
        _json_line(_attack("before")) + b"\xff\xfe\n" + _json_line(_attack("after"))
    )
    stats = LoadStats()

    attacks = list(AttackLoader.iter_attacks(str(file_path), stats=stats))

    assert [attack.attack_id for attack in attacks] == ["before", "after"]
    assert stats.rejected_invalid == 1


def test_iter_attacks_skips_oversized_lines_and_handles_unterminated_final_line(
    tmp_path: Path,
) -> None:
    oversized_line = b'{"attack_id":"' + (b"x" * (1024 * 1024)) + b'"}'
    file_path = tmp_path / "oversized.jsonl"
    file_path.write_bytes(oversized_line + b"\n" + _json_line(_attack("after")))
    stats = LoadStats()

    attacks = list(
        AttackLoader.iter_attacks(
            str(file_path),
            max_line_bytes=1024,
            stats=stats,
        )
    )

    assert [attack.attack_id for attack in attacks] == ["after"]
    assert stats.rejected_oversized == 1

    final_path = tmp_path / "oversized-final.jsonl"
    final_path.write_bytes(oversized_line)
    final_stats = LoadStats()

    final_attacks = list(
        AttackLoader.iter_attacks(
            str(final_path),
            max_line_bytes=1024,
            stats=final_stats,
        )
    )

    assert final_attacks == []
    assert final_stats.rejected_oversized == 1


def test_iter_attacks_skips_blank_lines_without_counting_them(tmp_path: Path) -> None:
    file_path = tmp_path / "blank-lines.jsonl"
    file_path.write_bytes(b"\n \t\r\n" + _json_line(_attack("only")) + b"\n")
    stats = LoadStats()

    attacks = list(AttackLoader.iter_attacks(str(file_path), stats=stats))

    assert [attack.attack_id for attack in attacks] == ["only"]
    assert stats.accepted == 1
    assert stats.filtered_out == 0
    assert stats.rejected_invalid == 0
    assert stats.rejected_oversized == 0


def test_iter_attacks_bounds_sanitized_error_samples(tmp_path: Path) -> None:
    file_path = tmp_path / "many-bad-lines.jsonl"
    file_path.write_text(
        "".join(f"private raw line content {index}\n" for index in range(1000)),
        encoding="utf-8",
    )
    stats = LoadStats()

    attacks = list(AttackLoader.iter_attacks(str(file_path), stats=stats))

    assert attacks == []
    assert stats.rejected_invalid == 1000
    assert len(stats.samples) <= 10
    for sample in stats.samples:
        assert set(sample) == {"line", "reason"}
        assert isinstance(sample["line"], int)
        assert sample["line"] >= 1
        assert isinstance(sample["reason"], str)
        assert 0 < len(sample["reason"]) <= 100
        assert "private raw line content" not in sample["reason"]


def test_iter_attacks_missing_file_raises_when_iterated(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jsonl"

    with pytest.raises(FileNotFoundError):
        list(AttackLoader.iter_attacks(str(missing_path)))


def test_iter_attacks_rejects_zero_line_limit_when_iterated(tmp_path: Path) -> None:
    file_path = tmp_path / "one.jsonl"
    file_path.write_bytes(_json_line(_attack("one")))

    with pytest.raises(ValueError):
        list(AttackLoader.iter_attacks(str(file_path), max_line_bytes=0))


def test_load_prompts_rejects_file_over_size_limit_before_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "oversized.json"
    file_path.write_text(
        json.dumps({"attacks": [_attack("valid")]}) + (" " * 200),
        encoding="utf-8",
    )

    def reject_parse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("oversized input must be rejected before parsing")

    monkeypatch.setattr(loader_module.json, "load", reject_parse)
    monkeypatch.setattr(loader_module.json, "loads", reject_parse)

    with pytest.raises(ValueError, match="too large"):
        AttackLoader.load_prompts(str(file_path), max_file_bytes=100)
