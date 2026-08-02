"""Optional Meta Context Engineering integration."""

from rethinkskill_mce.plugin import BENCHMARKS, benchmark_specs, harness_specs
from rethinkskill_mce.scoring import (
    evaluate_aegis2,
    evaluate_label,
    evaluate_lawbench_charge,
    evaluate_uspto50k,
)

__all__ = [
    "BENCHMARKS",
    "benchmark_specs",
    "evaluate_aegis2",
    "evaluate_label",
    "evaluate_lawbench_charge",
    "evaluate_uspto50k",
    "harness_specs",
]
