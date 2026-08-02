"""RethinkSkill benchmarks spreadsheetbench."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rethinkskill.benchmarks.core import artifact_path
from rethinkskill.benchmarks.scoring import Verdict, Verification, verify_spreadsheet
from rethinkskill.errors import ResultValidationError


def evaluate(row: Mapping[str, Any], base: Path) -> Verification:
    forced = row.get("forced_verdict")
    if forced:
        try:
            verdict = Verdict(str(forced))
        except ValueError as exc:
            raise ResultValidationError(f"invalid forced_verdict: {forced!r}") from exc
        return Verification(verdict, str(row.get("forced_reason") or "forced_by_frozen_input"))
    prediction = row.get("prediction_artifact")
    reference = row.get("reference_artifact")
    if not isinstance(prediction, dict) or not isinstance(reference, dict):
        return Verification(Verdict.ABSTAIN, "frozen_artifact_record_unavailable")
    return verify_spreadsheet(
        artifact_path(base, prediction, "prediction"),
        artifact_path(base, reference, "reference"),
        str(row.get("answer_position") or ""),
    )
