"""RethinkSkill benchmarks searchqa."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rethinkskill.benchmarks.core import required, required_text
from rethinkskill.benchmarks.scoring import Verification, verify_searchqa


def evaluate(row: Mapping[str, Any], _: Path) -> Verification:
    return verify_searchqa(required_text(row, "frozen_response"), required(row, "gold_aliases"))
