"""RethinkSkill benchmarks officeqa."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rethinkskill.benchmarks.core import optional_bool, required_text
from rethinkskill.benchmarks.scoring import Verification, verify_officeqa


def evaluate(row: Mapping[str, Any], _: Path) -> Verification:
    return verify_officeqa(
        required_text(row, "frozen_response"),
        required_text(row, "gold_answer"),
        strict_json=optional_bool(row, "strict_json"),
    )
