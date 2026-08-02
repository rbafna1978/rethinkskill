"""RethinkSkill integrations builtin."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rethinkskill.benchmarks.core import BenchmarkSpec
from rethinkskill.integrations.alfworld import benchmark_specs as alfworld_benchmark_specs
from rethinkskill.integrations.external import benchmark_specs as external_benchmark_specs
from rethinkskill.integrations.mce import benchmark_specs as mce_benchmark_specs
from rethinkskill.integrations.original_qa import benchmark_specs as original_qa_benchmark_specs
from rethinkskill.integrations.original_qa import harness_specs as original_qa_harness_specs
from rethinkskill.integrations.skillrl import benchmark_specs as skillrl_benchmark_specs
from rethinkskill.integrations.spreadsheetbench import (
    benchmark_specs as spreadsheetbench_benchmark_specs,
)

if TYPE_CHECKING:
    from rethinkskill.benchmarks.harness_catalog import NativeHarnessSpec


def builtin_benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    """Return all maintained scorer declarations in stable catalog order."""
    return (
        *original_qa_benchmark_specs(),
        *spreadsheetbench_benchmark_specs(),
        *alfworld_benchmark_specs(),
        *skillrl_benchmark_specs(),
        *mce_benchmark_specs(),
        *external_benchmark_specs(),
    )


def builtin_native_harness_specs() -> tuple[NativeHarnessSpec, ...]:
    """Return all maintained native adapters in stable catalog order."""
    return (*original_qa_harness_specs(),)
