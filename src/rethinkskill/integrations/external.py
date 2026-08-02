"""RethinkSkill integrations external."""

from __future__ import annotations

from rethinkskill.benchmarks.core import BenchmarkSpec
from rethinkskill.integrations.shared import A_EVOLVE_REPOSITORY, SKILLRL, external_spec


def benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    return (
        external_spec(
            "webshop",
            "interactive_environment",
            "web_navigation",
            metrics=("score", "success_rate"),
            sources=(SKILLRL,),
            notes="Requires the WebShop environment and its native reward function.",
        ),
        external_spec(
            "mcp-atlas",
            "tool_environment",
            "tool_use",
            metrics=("success_rate",),
            sources=(A_EVOLVE_REPOSITORY,),
            notes="Requires MCP servers and the benchmark's native evaluator.",
        ),
    )
