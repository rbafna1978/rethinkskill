"""RethinkSkill integrations spreadsheetbench."""

from __future__ import annotations

from rethinkskill.benchmarks.core import BenchmarkSpec
from rethinkskill.benchmarks.spreadsheetbench import evaluate as evaluate_spreadsheetbench
from rethinkskill.integrations.shared import case_spec


def benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    return (
        case_spec(
            "spreadsheetbench",
            "artifact_verification",
            "spreadsheets",
            evaluate_spreadsheetbench,
            artifact_fields=("prediction_artifact", "reference_artifact"),
        ),
    )
