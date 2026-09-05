"""Entry-point registration for the optional SpreadsheetBench harness."""

from rethinkskill.benchmarks.harness_catalog import (
    EvaluationAuthority,
    NativeHarnessSpec,
)
from rethinkskill_spreadsheetbench.harness import SpreadsheetBenchHarness


def spec() -> NativeHarnessSpec:
    return NativeHarnessSpec(
        name="spreadsheetbench",
        harness=SpreadsheetBenchHarness(),
        execution_scope=(
            "spreadsheetbench_single_codegen; frozen input workbooks; "
            "bounded Python artifact execution; "
            "official answer-cell value evaluation"
        ),
        source="rethinkskill_spreadsheetbench_schema",
        evaluation_authority=EvaluationAuthority.HARNESS,
    )


__all__ = ["spec"]
