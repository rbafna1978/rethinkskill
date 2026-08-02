"""RethinkSkill integrations alfworld."""

from __future__ import annotations

from rethinkskill.benchmarks.core import BenchmarkSpec
from rethinkskill.integrations.shared import external_spec


def benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    return (
        external_spec(
            "alfworld",
            "interactive_environment",
            "embodied_planning",
            metrics=("success_rate",),
            sources=(),
            notes="Success is determined by the installed environment's terminal won state; a frozen row without environment state is not an independent deterministic scoring case.",
        ),
    )
