"""RethinkSkill benchmarks livemath."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rethinkskill.benchmarks.core import case_id, optional_bool, required, required_text
from rethinkskill.benchmarks.scoring import Verification, verify_livemath
from rethinkskill.errors import ResultValidationError


def evaluate(row: Mapping[str, Any], _: Path) -> Verification:
    choices = required(row, "choices")
    correct = required(row, "correct_choice")
    if not isinstance(choices, list) or not all(isinstance(choice, dict) for choice in choices):
        raise ResultValidationError(f"case {case_id(row)}: choices must be a list of objects")
    if not isinstance(correct, dict):
        raise ResultValidationError(f"case {case_id(row)}: correct_choice must be an object")
    return verify_livemath(
        required_text(row, "frozen_response"),
        choices,
        correct,
        strict_json=optional_bool(row, "strict_json"),
    )
