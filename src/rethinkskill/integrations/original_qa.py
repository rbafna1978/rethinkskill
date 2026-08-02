"""RethinkSkill integrations original qa."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rethinkskill.benchmarks.core import BenchmarkSpec, reduce_mean_metrics
from rethinkskill.benchmarks.docvqa import evaluate as evaluate_docvqa
from rethinkskill.benchmarks.livemath import evaluate as evaluate_livemath
from rethinkskill.benchmarks.officeqa import evaluate as evaluate_officeqa
from rethinkskill.benchmarks.searchqa import evaluate as evaluate_searchqa
from rethinkskill.integrations.shared import case_spec

if TYPE_CHECKING:
    from rethinkskill.benchmarks.harness_catalog import NativeHarnessSpec


def benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    return (
        case_spec(
            "searchqa",
            "search_qa",
            "open_domain_qa",
            evaluate_searchqa,
            metrics=("pass_rate", "exact_match", "f1", "substring"),
            reducer=reduce_mean_metrics(("exact_match", "f1", "substring")),
        ),
        case_spec("officeqa", "text_qa", "office", evaluate_officeqa),
        case_spec("docvqa", "visual_qa", "documents", evaluate_docvqa),
        case_spec("livemath", "multiple_choice", "mathematics", evaluate_livemath),
    )


def harness_specs() -> tuple[NativeHarnessSpec, ...]:
    from rethinkskill.benchmarks.harness_catalog import NativeHarnessSpec
    from rethinkskill.benchmarks.qa import (
        DocVQAHarness,
        LiveMathHarness,
        OfficeQAHarness,
        SearchQAHarness,
    )

    return (
        NativeHarnessSpec(
            name="searchqa",
            harness=SearchQAHarness(),
            execution_scope="context_grounded_qa",
            source="rethinkskill_searchqa_schema",
        ),
        NativeHarnessSpec(
            name="livemath",
            harness=LiveMathHarness(),
            execution_scope="live_mathematician_bench_mcq; seeded choice shuffling; theorem and sketch excluded from target prompt",
            source="rethinkskill_livemathematicianbench_schema",
        ),
        NativeHarnessSpec(
            name="officeqa",
            harness=OfficeQAHarness(),
            execution_scope="officeqa_local_document_bundle; deterministic provider-neutral lexical evidence retrieval",
            source="rethinkskill_officeqa_schema",
        ),
        NativeHarnessSpec(
            name="docvqa",
            harness=DocVQAHarness(),
            execution_scope="docvqa_document_image; task-local hash-bound multimodal attachment",
            source="rethinkskill_docvqa_schema",
        ),
    )
