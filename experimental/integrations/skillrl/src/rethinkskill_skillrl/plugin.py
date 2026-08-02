"""Entry-point registrations for the optional SkillRL offline suite."""

from rethinkskill.benchmarks.core import (
    BenchmarkSource,
    BenchmarkSpec,
    EvaluationMode,
    reduce_mean_metrics,
)
from rethinkskill.benchmarks.harness_catalog import NativeHarnessSpec
from rethinkskill_skillrl.harness import SkillRLOfflineHarness
from rethinkskill_skillrl.scoring import evaluate

BENCHMARKS = (
    "nq",
    "triviaqa",
    "popqa",
    "hotpotqa",
    "2wiki",
    "musique",
    "bamboogle",
)
SOURCE = BenchmarkSource(
    relation="evaluated_by",
    title=("SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning"),
    url="https://arxiv.org/abs/2602.08234",
)
NOTES = (
    "Optional offline response scorer and context-grounded harness; "
    "install rethinkskill-skillrl. This is not the paper's live "
    "retrieval environment."
)


def benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    """Materialize all seven declarations with one deterministic scorer."""

    reducer = reduce_mean_metrics(("exact_match", "f1", "substring"))
    return tuple(
        BenchmarkSpec(
            name=name,
            family="search_augmented_qa",
            domain="open_domain_qa",
            mode=EvaluationMode.CASES,
            metrics=("pass_rate", "exact_match", "f1", "substring"),
            adapter=evaluate,
            reducer=reducer,
            sources=(SOURCE,),
            notes=NOTES,
        )
        for name in BENCHMARKS
    )


def harness_specs() -> tuple[NativeHarnessSpec, ...]:
    """Register all seven names against the shared offline QA harness."""

    harness = SkillRLOfflineHarness()
    return tuple(
        NativeHarnessSpec(
            name=name,
            harness=harness,
            execution_scope=(
                "context_grounded_offline_qa; not equivalent to the "
                "paper's live retrieval environment"
            ),
            source="SkillRL offline proxy package",
        )
        for name in BENCHMARKS
    )


__all__ = ["BENCHMARKS", "benchmark_specs", "harness_specs"]
