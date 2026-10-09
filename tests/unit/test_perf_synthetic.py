"""Tests for synthetic performance-harness helpers."""

from collections import Counter
from collections.abc import Iterator
from itertools import islice
from typing import get_args

import pytest
from llm_guard_bench.perf.synthetic import current_rss_bytes, generate_attacks

from llm_guard_bench.domain.models import AttackCategoryType, AttackDefinition


def test_generate_attacks_yields_unique_attack_definitions() -> None:
    attacks = generate_attacks(5)

    assert isinstance(attacks, Iterator)
    assert not isinstance(attacks, (list, tuple))
    generated = list(attacks)
    assert len(generated) == 5
    assert all(isinstance(attack, AttackDefinition) for attack in generated)
    assert len({attack.attack_id for attack in generated}) == 5


def test_generate_attacks_is_lazy_for_large_counts() -> None:
    generated = list(islice(generate_attacks(10**9), 3))

    assert len(generated) == 3


def test_generate_attacks_with_zero_count_is_empty() -> None:
    assert list(generate_attacks(0)) == []


def test_generate_attacks_rejects_negative_count_at_call_time() -> None:
    with pytest.raises(ValueError):
        generate_attacks(-1)


def test_generate_attacks_are_deterministic() -> None:
    first_ids = [attack.attack_id for attack in islice(generate_attacks(10), 10)]
    second_ids = [attack.attack_id for attack in islice(generate_attacks(10), 10)]

    assert first_ids == second_ids


def test_generate_attacks_cycle_through_every_model_category() -> None:
    categories = get_args(AttackCategoryType)
    generated = list(generate_attacks(3 * len(categories)))

    assert Counter(attack.category for attack in generated) == Counter(
        {category: 3 for category in categories}
    )


def test_current_rss_bytes_returns_positive_integer() -> None:
    rss_bytes = current_rss_bytes()

    assert isinstance(rss_bytes, int)
    assert rss_bytes > 0
