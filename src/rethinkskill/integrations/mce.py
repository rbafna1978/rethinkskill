"""RethinkSkill integrations mce."""

from __future__ import annotations

from rethinkskill.benchmarks.core import BenchmarkSpec
from rethinkskill.integrations.shared import MCE, declared_spec

BENCHMARKS = ("finer", "uspto50k", "symptom2disease", "lawbench-charge", "aegis2")

NOTES = "Optional file-backed scorer and harness; install rethinkskill-mce. Official dataset preprocessing remains provenance-bound to the supplied dataset."


def benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    return (
        declared_spec(
            "finer",
            "single_label_classification",
            "finance",
            metrics=("pass_rate", "accuracy"),
            sources=(MCE,),
            notes=NOTES,
        ),
        declared_spec(
            "uspto50k",
            "structured_exact_match",
            "chemistry",
            metrics=("pass_rate", "exact_match"),
            sources=(MCE,),
            notes=NOTES,
        ),
        declared_spec(
            "symptom2disease",
            "single_label_classification",
            "medicine",
            metrics=("pass_rate", "accuracy"),
            sources=(MCE,),
            notes=NOTES,
        ),
        declared_spec(
            "lawbench-charge",
            "multi_label_classification",
            "law",
            metrics=("pass_rate", "micro_f1"),
            sources=(MCE,),
            notes=NOTES,
        ),
        declared_spec(
            "aegis2",
            "binary_classification",
            "ai_safety",
            metrics=("pass_rate", "f1"),
            sources=(MCE,),
            notes=NOTES,
        ),
    )
