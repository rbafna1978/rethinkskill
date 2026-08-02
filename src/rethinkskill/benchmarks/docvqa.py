"""RethinkSkill benchmarks docvqa."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rethinkskill.benchmarks.core import optional_bool, required, required_text
from rethinkskill.benchmarks.scoring import Verification, verify_docvqa


def evaluate(row: Mapping[str, Any], _: Path) -> Verification:
    return verify_docvqa(
        required_text(row, "frozen_response"),
        required(row, "gold_aliases"),
        strict_json=optional_bool(row, "strict_json"),
    )
