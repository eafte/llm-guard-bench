"""Synthetic inputs and measurements for performance harnesses."""

from collections.abc import Iterator
from typing import get_args

from llm_guard_bench.domain.models import AttackCategoryType, AttackDefinition


def generate_attacks(count: int) -> Iterator[AttackDefinition]:
    """Return a lazy iterator of deterministic synthetic attacks."""
    if count < 0:
        raise ValueError("count must be non-negative")

    categories = get_args(AttackCategoryType)

    def _generate() -> Iterator[AttackDefinition]:
        for index in range(count):
            yield AttackDefinition(
                attack_id=f"synthetic-{index:09d}",
                attack_name=f"Synthetic attack {index}",
                description="Synthetic performance-harness attack.",
                category=categories[index % len(categories)],
                severity="LOW",
                tags=["synthetic"],
                turns=["Synthetic test prompt."],
            )

    return _generate()


def current_rss_bytes() -> int:
    """Return the current process resident set size in bytes."""
    import psutil

    return int(psutil.Process().memory_info().rss)
