import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from core.models import AttackDefinition

logger = logging.getLogger(__name__)


class AttackLoader:
    """Loads and manages attack definitions from configuration files."""

    @staticmethod
    def load_prompts(
        file_path: str, categories: Optional[List[str]] = None
    ) -> List[AttackDefinition]:
        """
        Load and parse attack definitions from a JSON file with optional category filtering.

        Args:
            file_path: Path to the attacks JSON file.
            categories: Optional list of attack categories to filter by.
                       If None, all attacks are returned.

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

        try:
            with open(file_path_obj, "r", encoding="utf-8") as f:
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
        attack_definitions: List[AttackDefinition] = []

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
                        attack_category: Optional[str] = attack_dict.get("category")
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
