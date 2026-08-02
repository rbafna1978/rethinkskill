"""RethinkSkill benchmarks alfworld."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rethinkskill.benchmarks.scoring import Verification, verify_alfworld


def evaluate(_: Mapping[str, Any], __: Path) -> Verification:
    return verify_alfworld()
