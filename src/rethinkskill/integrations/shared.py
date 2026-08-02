"""RethinkSkill integrations shared."""

from __future__ import annotations

from rethinkskill.benchmarks.core import (
    BenchmarkSource,
    BenchmarkSpec,
    CaseAdapter,
    EvaluationMode,
    MetricReducer,
)


def paper(title: str, url: str) -> BenchmarkSource:
    return BenchmarkSource("evaluated_by", title, url)


def repository(title: str, url: str) -> BenchmarkSource:
    return BenchmarkSource("official_repository", title, url)


SKILLRL = paper(
    "SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning",
    "https://arxiv.org/abs/2602.08234",
)

MCE = paper(
    "Meta Context Engineering via Agentic Skill Evolution", "https://arxiv.org/abs/2601.21557"
)

A_EVOLVE_REPOSITORY = repository("A-Evolve", "https://github.com/A-EVO-Lab/a-evolve")


def case_spec(
    name: str,
    family: str,
    domain: str,
    adapter: CaseAdapter,
    *,
    metrics: tuple[str, ...] = ("pass_rate",),
    reducer: MetricReducer | None = None,
    artifact_fields: tuple[str, ...] = (),
    sources: tuple[BenchmarkSource, ...] = (),
    notes: str = "",
) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        family=family,
        domain=domain,
        mode=EvaluationMode.CASES,
        metrics=metrics,
        adapter=adapter,
        reducer=reducer,
        artifact_fields=artifact_fields,
        sources=sources,
        notes=notes,
    )


def external_spec(
    name: str,
    family: str,
    domain: str,
    *,
    metrics: tuple[str, ...],
    sources: tuple[BenchmarkSource, ...],
    notes: str,
) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        family=family,
        domain=domain,
        mode=EvaluationMode.EXTERNAL_HARNESS,
        metrics=metrics,
        sources=sources,
        notes=notes,
    )


def declared_spec(
    name: str,
    family: str,
    domain: str,
    *,
    metrics: tuple[str, ...],
    sources: tuple[BenchmarkSource, ...],
    notes: str,
    artifact_fields: tuple[str, ...] = (),
) -> BenchmarkSpec:
    """Declare a capability whose executable scorer is separately packaged."""
    return BenchmarkSpec(
        name=name,
        family=family,
        domain=domain,
        mode=EvaluationMode.DECLARED_ONLY,
        metrics=metrics,
        artifact_fields=artifact_fields,
        sources=sources,
        notes=notes,
    )
