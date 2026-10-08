"""Tests for repeatable JSONL attack sources."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from llm_guard_bench.domain.models import AttackDefinition
from llm_guard_bench.streaming_io.loader import (
    AttackLoader,
    AttackSourceChangedError,
    JsonlAttackSource,
    LoadStats,
)


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


def test_source_is_lazy_and_missing_path_errors_on_first_next(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.jsonl"
    source = JsonlAttackSource(str(missing_path))

    assert source.count is None
    assert source.last_stats is None

    iterator = iter(source)
    with pytest.raises(FileNotFoundError):
        next(iterator)


def test_source_records_stats_after_complete_pass(tmp_path: Path) -> None:
    file_path = tmp_path / "attacks.jsonl"
    file_path.write_bytes(
        _json_line(_attack("first"))
        + b"{invalid json\n"
        + _json_line(_attack("second"))
        + _json_line(_attack("third"))
    )
    source = JsonlAttackSource(str(file_path))

    attacks = list(source)

    assert [attack.attack_id for attack in attacks] == ["first", "second", "third"]
    assert source.count == 3
    assert source.last_stats is not None
    assert isinstance(source.last_stats, LoadStats)
    assert source.last_stats.accepted == 3
    assert source.last_stats.rejected_invalid == 1


def test_source_can_be_iterated_again_without_changing_count(tmp_path: Path) -> None:
    file_path = tmp_path / "repeat.jsonl"
    file_path.write_bytes(_json_line(_attack("first")) + _json_line(_attack("second")))
    source = JsonlAttackSource(str(file_path))

    first_pass = [attack.attack_id for attack in source]
    first_count = source.count
    second_pass = [attack.attack_id for attack in source]

    assert first_pass == ["first", "second"]
    assert second_pass == first_pass
    assert source.count == first_count == 2


def test_source_applies_categories_and_counts_filtered_entries(tmp_path: Path) -> None:
    file_path = tmp_path / "categories.jsonl"
    file_path.write_bytes(
        _json_line(_attack("dan", category="DAN"))
        + _json_line(_attack("roleplay", category="ROLEPLAY_EXPLOIT"))
    )
    source = JsonlAttackSource(str(file_path), categories=["DAN"])

    attacks = list(source)

    assert [attack.attack_id for attack in attacks] == ["dan"]
    assert source.count == 1
    assert source.last_stats is not None
    assert source.last_stats.filtered_out == 1


def test_source_detects_file_size_change_before_second_pass_yields(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "changed-size.jsonl"
    file_path.write_bytes(_json_line(_attack("first")))
    source = JsonlAttackSource(str(file_path))
    assert [attack.attack_id for attack in source] == ["first"]

    file_path.write_bytes(_json_line(_attack("first")) + _json_line(_attack("second")))

    second_pass = iter(source)
    with pytest.raises(AttackSourceChangedError):
        next(second_pass)


def test_source_detects_same_metadata_content_change_at_pass_end(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "same-metadata-change.jsonl"
    original_lines = [
        _json_line(_attack("first")),
        _json_line(_attack("second")),
        _json_line(_attack("third")),
    ]
    file_path.write_bytes(b"".join(original_lines))
    original_stat = file_path.stat()
    source = JsonlAttackSource(str(file_path))

    assert [attack.attack_id for attack in source] == ["first", "second", "third"]

    replacement_line = b"!" * (len(original_lines[1]) - 1) + b"\n"
    assert len(replacement_line) == len(original_lines[1])
    file_path.write_bytes(original_lines[0] + replacement_line + original_lines[2])
    os.utime(file_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    pass_two_attack_ids: list[str] = []
    with pytest.raises(AttackSourceChangedError):
        for attack in source:
            pass_two_attack_ids.append(attack.attack_id)

    assert pass_two_attack_ids == ["first", "third"]


def test_source_closes_underlying_iterator_when_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file_path = tmp_path / "close.jsonl"
    file_path.write_bytes(_json_line(_attack("only")))
    attack = AttackDefinition(**_attack("only"))

    class CloseTrackingIterator:
        def __init__(self) -> None:
            self.items = iter([attack])
            self.closed = False

        def __iter__(self) -> CloseTrackingIterator:
            return self

        def __next__(self) -> AttackDefinition:
            return next(self.items)

        def close(self) -> None:
            self.closed = True

    underlying_iterator = CloseTrackingIterator()
    monkeypatch.setattr(
        AttackLoader,
        "iter_attacks",
        lambda *_args, **_kwargs: underlying_iterator,
    )
    source = JsonlAttackSource(str(file_path))
    source_iterator = iter(source)

    assert next(source_iterator).attack_id == "only"
    source_iterator.close()

    assert underlying_iterator.closed
