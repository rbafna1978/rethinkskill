"""Entry-point registrations for the optional MCE suite."""

from dataclasses import dataclass

from rethinkskill.benchmarks.core import (
    BenchmarkSource,
    BenchmarkSpec,
    CaseAdapter,
    EvaluationMode,
    MetricReducer,
    reduce_binary_f1,
    reduce_mean_metrics,
    reduce_micro_f1,
)
from rethinkskill.benchmarks.harness_catalog import NativeHarnessSpec
from rethinkskill.runtime.harnesses import DeterministicCaseHarness
from rethinkskill_mce.scoring import (
    evaluate_aegis2,
    evaluate_label,
    evaluate_lawbench_charge,
    evaluate_uspto50k,
)

BENCHMARKS = (
    "finer",
    "uspto50k",
    "symptom2disease",
    "lawbench-charge",
    "aegis2",
)
SOURCE = BenchmarkSource(
    relation="evaluated_by",
    title="Meta Context Engineering via Agentic Skill Evolution",
    url="https://arxiv.org/abs/2601.21557",
)
NOTES = (
    "Optional file-backed scorer and harness; install rethinkskill-mce. "
    "Official dataset preprocessing remains provenance-bound to the supplied "
    "dataset."
)


@dataclass(frozen=True, slots=True)
class _TaskDefinition:
    name: str
    family: str
    domain: str
    metrics: tuple[str, ...]
    adapter: CaseAdapter
    reducer: MetricReducer
    task_kind: str
    input_fields: tuple[str, ...]
    gold_field: str
    gold_aliases: tuple[str, ...]
    scope: str


_TASKS = (
    _TaskDefinition(
        "finer",
        "single_label_classification",
        "finance",
        ("pass_rate", "accuracy"),
        evaluate_label,
        reduce_mean_metrics(("accuracy",)),
        "single_label",
        ("input", "text", "sentence", "prompt"),
        "gold_label",
        ("label",),
        "finance classification",
    ),
    _TaskDefinition(
        "uspto50k",
        "structured_exact_match",
        "chemistry",
        ("pass_rate", "exact_match"),
        evaluate_uspto50k,
        reduce_mean_metrics(("exact_match",)),
        "structured_exact_match",
        ("input", "product", "product_smiles", "prompt"),
        "gold_reactants",
        ("reactants",),
        "reaction-center reactant prediction",
    ),
    _TaskDefinition(
        "symptom2disease",
        "single_label_classification",
        "medicine",
        ("pass_rate", "accuracy"),
        evaluate_label,
        reduce_mean_metrics(("accuracy",)),
        "single_label",
        ("input", "text", "symptoms", "prompt"),
        "gold_label",
        ("label", "disease"),
        "symptom-to-disease classification",
    ),
    _TaskDefinition(
        "lawbench-charge",
        "multi_label_classification",
        "law",
        ("pass_rate", "micro_f1"),
        evaluate_lawbench_charge,
        reduce_micro_f1,
        "multi_label",
        ("input", "text", "fact", "prompt"),
        "gold_labels",
        ("labels", "charges"),
        "criminal-charge multi-label classification",
    ),
    _TaskDefinition(
        "aegis2",
        "binary_classification",
        "ai_safety",
        ("pass_rate", "f1"),
        evaluate_aegis2,
        reduce_binary_f1,
        "single_label",
        ("input", "text", "prompt"),
        "gold_label",
        ("label",),
        "binary safety classification",
    ),
)


def benchmark_specs() -> tuple[BenchmarkSpec, ...]:
    """Materialize all five declarations with deterministic scorers."""

    return tuple(
        BenchmarkSpec(
            name=task.name,
            family=task.family,
            domain=task.domain,
            mode=EvaluationMode.CASES,
            metrics=task.metrics,
            adapter=task.adapter,
            reducer=task.reducer,
            sources=(SOURCE,),
            notes=NOTES,
        )
        for task in _TASKS
    )


def harness_specs() -> tuple[NativeHarnessSpec, ...]:
    """Register all five names against their file-backed task schemas."""

    scorers = {spec.name: spec.adapter for spec in benchmark_specs()}
    return tuple(
        NativeHarnessSpec(
            name=task.name,
            harness=DeterministicCaseHarness(
                benchmark=task.name,
                task_kind=task.task_kind,
                input_fields=task.input_fields,
                gold_field=task.gold_field,
                gold_aliases=task.gold_aliases,
                adapter=scorers[task.name],
            ),
            execution_scope=(
                f"file_backed_{task.scope}; official dataset preprocessing "
                "remains provenance-bound to the supplied dataset"
            ),
            source="Meta Context Engineering optional integration package",
        )
        for task in _TASKS
    )


__all__ = ["BENCHMARKS", "benchmark_specs", "harness_specs"]
