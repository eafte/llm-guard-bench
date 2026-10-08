import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from llm_guard_bench.domain.models import AttackDefinition

logger = logging.getLogger(__name__)

_RejectionReason = Literal[
    "invalid_json",
    "invalid_utf8",
    "invalid_attack",
    "not_an_object",
    "line_too_long",
]


@dataclass
class LoadStats:
    """Counters and sanitized samples for attack loading."""

    accepted: int = 0
    filtered_out: int = 0
    rejected_invalid: int = 0
    rejected_oversized: int = 0
    samples: list[dict[str, int | str]] = field(default_factory=list)

    def record_rejection(self, line: int, reason: _RejectionReason) -> None:
        """Count a rejected line and retain a bounded, sanitized sample."""
        if reason == "line_too_long":
            self.rejected_oversized += 1
        else:
            self.rejected_invalid += 1

        if len(self.samples) < 10:
            self.samples.append({"line": line, "reason": reason})


class AttackLoader:
    """Loads and manages attack definitions from configuration files."""

    @staticmethod
    def iter_attacks(
        file_path: str,
        categories: list[str] | None = None,
        *,
        max_line_bytes: int = 1_048_576,
        stats: LoadStats | None = None,
    ) -> Iterator[AttackDefinition]:
        """Yield valid attack definitions from a bounded-memory JSONL file."""
        if max_line_bytes < 1:
            raise ValueError("max_line_bytes must be at least 1")

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Attacks file not found: {file_path}")

        load_stats = stats if stats is not None else LoadStats()
        category_set = set(categories) if categories is not None else None
        read_size = max_line_bytes + 2

        with file_path_obj.open("rb") as source:
            line_number = 0
            while True:
                raw_line = source.readline(read_size)
                if not raw_line:
                    break
                line_number += 1

                has_newline = raw_line.endswith(b"\n")
                content_length = len(raw_line)
                if has_newline:
                    content_length -= 1
                    if raw_line.endswith(b"\r\n"):
                        content_length -= 1

                if content_length > max_line_bytes or (
                    not has_newline and len(raw_line) == read_size
                ):
                    load_stats.record_rejection(line_number, "line_too_long")
                    if not has_newline:
                        del raw_line
                        while True:
                            discarded_chunk = source.readline(read_size)
                            if not discarded_chunk:
                                break
                            chunk_ends_line = discarded_chunk.endswith(b"\n")
                            del discarded_chunk
                            if chunk_ends_line:
                                break
                    continue

                content = (
                    raw_line[:-2]
                    if raw_line.endswith(b"\r\n")
                    else (raw_line[:-1] if has_newline else raw_line)
                )
                if line_number == 1 and content.startswith(b"\xef\xbb\xbf"):
                    content = content[3:]
                if not content.strip():
                    continue

                try:
                    decoded_line = content.decode("utf-8")
                except UnicodeDecodeError:
                    load_stats.record_rejection(line_number, "invalid_utf8")
                    continue

                try:
                    attack_data: Any = json.loads(decoded_line)
                except json.JSONDecodeError:
                    load_stats.record_rejection(line_number, "invalid_json")
                    continue

                if not isinstance(attack_data, dict):
                    load_stats.record_rejection(line_number, "not_an_object")
                    continue

                if category_set is not None:
                    try:
                        category_matches = attack_data.get("category") in category_set
                    except TypeError:
                        load_stats.record_rejection(line_number, "invalid_attack")
                        continue
                    if not category_matches:
                        load_stats.filtered_out += 1
                        continue

                try:
                    attack = AttackDefinition(**attack_data)
                except (TypeError, ValueError, ValidationError):
                    load_stats.record_rejection(line_number, "invalid_attack")
                    continue

                load_stats.accepted += 1
                yield attack

    @staticmethod
    def load_prompts(
        file_path: str,
        categories: list[str] | None = None,
        *,
        max_file_bytes: int = 52_428_800,
    ) -> list[AttackDefinition]:
        """
        Load and parse attack definitions from a JSON file with optional category filtering.

        Args:
            file_path: Path to the attacks JSON file.
            categories: Optional list of attack categories to filter by.
                       If None, all attacks are returned.
            max_file_bytes: Maximum file size accepted before JSON parsing.

        Returns:
            List of AttackDefinition objects matching the specified criteria.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            ValueError: If the file is empty or does not contain a valid attacks array.
        """
        file_path_obj: Path = Path(file_path)

        # Validate file existence
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Attacks file not found: {file_path}")

        # Check if file is empty
        if file_path_obj.stat().st_size == 0:
            raise ValueError(f"Attacks file is empty: {file_path}")

        if max_file_bytes < 1:
            raise ValueError("max_file_bytes must be at least 1")
        if file_path_obj.stat().st_size > max_file_bytes:
            raise ValueError(f"Attacks file is too large: {file_path}")

        try:
            with open(file_path_obj, encoding="utf-8") as f:
                data: Any = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in attacks file: {file_path}",
                e.doc,
                e.pos,
            ) from e

        # Validate data structure - support both "attacks" and "prompts" keys for compatibility
        if not isinstance(data, dict):
            raise ValueError("Attacks file must contain a JSON object")

        attacks_list: Any = data.get("attacks") or data.get("prompts")

        if attacks_list is None:
            raise ValueError(
                "Attacks file must contain an 'attacks' or 'prompts' key with a list value"
            )

        if not isinstance(attacks_list, list):
            raise ValueError("'attacks'/'prompts' field must be a list")

        if len(attacks_list) == 0:
            logger.warning(f"Attacks file contains an empty list: {file_path}")
            return []

        # Convert raw dicts to AttackDefinition objects and filter by categories if provided
        attack_definitions: list[AttackDefinition] = []

        if categories is None:
            # Return all attacks if no categories specified
            for attack_dict in attacks_list:
                try:
                    attack_def = AttackDefinition(**attack_dict)
                    attack_definitions.append(attack_def)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Skipping invalid attack entry: {e}")
        else:
            # Filter attacks by category
            category_set: set = set(categories)
            for attack_dict in attacks_list:
                try:
                    if isinstance(attack_dict, dict):
                        attack_category: str | None = attack_dict.get("category")
                        if attack_category in category_set:
                            attack_def = AttackDefinition(**attack_dict)
                            attack_definitions.append(attack_def)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Skipping invalid attack entry: {e}")

        logger.info(
            f"Loaded {len(attack_definitions)} attack definitions from {file_path} "
            f"(categories: {categories if categories else 'all'})"
        )

        return attack_definitions
