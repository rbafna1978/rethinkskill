"""RethinkSkill integrations skillrl."""

from __future__ import annotations

from rethinkskill.benchmarks.core import BenchmarkSpec
from rethinkskill.integrations.shared import SKILLRL, declared_spec

BENCHMARKS = ("nq", "triviaqa", "popqa", "hotpotqa", "2wiki", "musique", "bamboogle")


def benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    return tuple(
        declared_spec(
            name,
            "search_augmented_qa",
            "open_domain_qa",
            metrics=("pass_rate", "exact_match", "f1", "substring"),
            sources=(SKILLRL,),
            notes="Optional offline response scorer and context-grounded harness; install rethinkskill-skillrl. This is not the paper's live retrieval environment.",
        )
        for name in BENCHMARKS
    )
