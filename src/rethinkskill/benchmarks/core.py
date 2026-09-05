"""RethinkSkill benchmarks core."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from rethinkskill.benchmarks.scoring import Verdict, Verification, freeze_verification
from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.integrations.catalog import AdapterDelivery, IntegrationSpec, integration_catalog
from rethinkskill.utils.fs import is_within
from rethinkskill.utils.plugins import (
    ExtensionRegistration,
    LoadedPluginSpec,
    SealableCatalog,
    load_plugin_spec_records,
)
from rethinkskill.utils.release import canonical_distribution_name


def case_id(row: Mapping[str, Any]) -> str:
    return str(row.get("case_id") or "<unknown>")


def required(row: Mapping[str, Any], key: str) -> Any:
    if key not in row:
        raise ResultValidationError(f"case {case_id(row)}: missing verifier field {key!r}")
    return row[key]


def required_text(row: Mapping[str, Any], key: str) -> str:
    value = required(row, key)
    if not isinstance(value, str):
        raise ResultValidationError(f"case {case_id(row)}: verifier field {key!r} must be text")
    return value


def optional_bool(row: Mapping[str, Any], key: str, *, default: bool = False) -> bool:
    value = row.get(key, default)
    if type(value) is not bool:
        raise ResultValidationError(f"case {case_id(row)}: verifier field {key!r} must be boolean")
    return value


def artifact_path(base: Path, record: Mapping[str, Any], label: str) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise ResultValidationError(f"{label} artifact requires a relative path")
    relative = PurePosixPath(raw)
    if ".." in relative.parts:
        raise ResultValidationError(f"{label} artifact escapes case directory: {raw}")
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", "."} for part in relative.parts)
        or ("\\" in raw)
    ):
        raise ResultValidationError(f"{label} artifact path must be portable relative text: {raw}")
    base_path = base.expanduser().absolute()
    path = base_path.joinpath(*relative.parts)
    if base_path.is_symlink():
        raise ResultValidationError(f"{label} artifact base must not be a symlink")
    current = base_path
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ResultValidationError(f"{label} artifact must not use symlinks: {raw}")
    if not is_within(path.resolve(), base_path.resolve()):
        raise ResultValidationError(f"{label} artifact escapes case directory: {raw}")
    return path


class CaseAdapter(Protocol):
    """Evaluate one frozen case without model calls."""

    def __call__(self, row: Mapping[str, Any], base: Path) -> Verification: ...


class MetricReducer(Protocol):
    """Reduce case verdicts into benchmark-level metrics."""

    def __call__(self, rows: Sequence[Verification]) -> Mapping[str, float]: ...


class EvaluationMode(str, Enum):
    CASES = "deterministic_cases"
    EXTERNAL_HARNESS = "external_harness"
    DECLARED_ONLY = "declared_only"


@dataclass(frozen=True, slots=True)
class BenchmarkSource:
    relation: str
    title: str
    url: str

    def validate(self) -> None:
        if not all(
            type(value) is str and value.strip() for value in (self.relation, self.title, self.url)
        ):
            raise ConfigurationError("benchmark source relation, title, and URL are required")

    def public(self) -> dict[str, str]:
        return {"relation": self.relation, "title": self.title, "url": self.url}


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """One benchmark capability, independent of the experiment runner."""

    name: str
    family: str
    domain: str
    mode: EvaluationMode
    metrics: tuple[str, ...]
    adapter: CaseAdapter | None = None
    reducer: MetricReducer | None = None
    sources: tuple[BenchmarkSource, ...] = ()
    notes: str = ""
    artifact_fields: tuple[str, ...] = ()

    def validate(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name != self.name.lower()
            or (self.name != self.name.strip())
        ):
            raise ConfigurationError(
                f"benchmark name must be non-empty lowercase text: {self.name!r}"
            )
        if (
            type(self.family) is not str
            or not self.family.strip()
            or type(self.domain) is not str
            or (not self.domain.strip())
        ):
            raise ConfigurationError(f"benchmark family and domain are required: {self.name}")
        if type(self.mode) is not EvaluationMode:
            raise ConfigurationError(f"benchmark mode is invalid: {self.name}: {self.mode!r}")
        if (
            type(self.metrics) is not tuple
            or not self.metrics
            or (not all(type(metric) is str and metric.strip() for metric in self.metrics))
            or (len(set(self.metrics)) != len(self.metrics))
        ):
            raise ConfigurationError(
                f"benchmark metrics must be a non-empty tuple of unique strings: {self.name}"
            )
        if self.adapter is not None and (not callable(self.adapter)):
            raise ConfigurationError(f"benchmark adapter must be callable: {self.name}")
        if self.reducer is not None and (not callable(self.reducer)):
            raise ConfigurationError(f"benchmark reducer must be callable: {self.name}")
        if (
            type(self.artifact_fields) is not tuple
            or not all(type(field) is str and field.strip() for field in self.artifact_fields)
            or len(set(self.artifact_fields)) != len(self.artifact_fields)
        ):
            raise ConfigurationError(f"benchmark artifact fields are invalid: {self.name}")
        if type(self.sources) is not tuple or not all(
            type(source) is BenchmarkSource for source in self.sources
        ):
            raise ConfigurationError(
                f"benchmark sources must be exact BenchmarkSource values: {self.name}"
            )
        for source in self.sources:
            source.validate()
        if type(self.notes) is not str:
            raise ConfigurationError(f"benchmark notes must be text: {self.name}")
        if self.mode is EvaluationMode.CASES and self.adapter is None:
            raise ConfigurationError(
                f"deterministic benchmark requires a case adapter: {self.name}"
            )
        if self.mode is EvaluationMode.EXTERNAL_HARNESS and self.adapter is not None:
            raise ConfigurationError(
                f"external harness cannot masquerade as a case adapter: {self.name}"
            )
        if self.mode is EvaluationMode.DECLARED_ONLY and (
            self.adapter is not None or self.reducer is not None
        ):
            raise ConfigurationError(
                f"declared-only benchmark cannot contain executable scoring code: {self.name}"
            )

    @property
    def locally_evaluable(self) -> bool:
        return self.mode is EvaluationMode.CASES

    def summarize(self, rows: Sequence[Verification]) -> dict[str, float]:
        if not self.locally_evaluable:
            raise ConfigurationError(
                f"benchmark has no installed deterministic scorer: {self.name}"
            )
        try:
            frozen_rows = tuple(freeze_verification(row) for row in rows)
        except Exception as exc:
            raise ConfigurationError(
                f"benchmark reducer requires valid exact Verification rows: {self.name}"
            ) from exc
        if self.reducer is not None:
            observed = self.reducer(frozen_rows)
        else:
            observed = reduce_verdicts(frozen_rows)
        if not isinstance(observed, Mapping):
            raise ConfigurationError(f"benchmark reducer returned a non-mapping: {self.name}")
        summary = dict(observed)
        for metric, value in summary.items():
            if type(metric) is not str or not metric:
                raise ConfigurationError(
                    f"benchmark reducer returned an invalid metric name: {self.name}"
                )
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ConfigurationError(
                    f"benchmark reducer returned a non-finite numeric metric: {self.name}.{metric}"
                )
        return summary

    def public(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "domain": self.domain,
            "evaluation_mode": self.mode.value,
            "locally_evaluable": self.locally_evaluable,
            "metrics": list(self.metrics),
            "sources": [source.public() for source in self.sources],
            "notes": self.notes,
        }


def reduce_verdicts(rows: Sequence[Verification]) -> dict[str, float]:
    decided = sum(row.verdict is not Verdict.ABSTAIN for row in rows)
    passed = sum(row.verdict is Verdict.PASS for row in rows)
    return {
        "pass_rate": passed / decided if decided else 0.0,
        "coverage": decided / len(rows) if rows else 0.0,
    }


def reduce_mean_metrics(
    metric_names: Sequence[str],
) -> Callable[[Sequence[Verification]], Mapping[str, float]]:
    names = tuple(metric_names)

    def reduce(rows: Sequence[Verification]) -> Mapping[str, float]:
        output = reduce_verdicts(rows)
        for name in names:
            values = [
                float(row.metrics[name])
                for row in rows
                if row.metrics is not None and name in row.metrics
            ]
            output[name] = sum(values) / len(values) if values else 0.0
        return output

    return reduce


def reduce_binary_f1(rows: Sequence[Verification]) -> Mapping[str, float]:
    totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
    for row in rows:
        for name in totals:
            if row.metrics is not None:
                totals[name] += float(row.metrics.get(name, 0.0))
    denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    return {**reduce_verdicts(rows), "f1": 2 * totals["tp"] / denominator if denominator else 0.0}


def reduce_micro_f1(rows: Sequence[Verification]) -> Mapping[str, float]:
    output = dict(reduce_binary_f1(rows))
    output["micro_f1"] = output.pop("f1")
    return output


class BenchmarkCatalog(SealableCatalog):
    """Fail-closed registry for built-in and explicitly loaded plugins."""

    _catalog_label = "benchmark"

    def __init__(self, specs: Iterable[BenchmarkSpec] = (), *, source: str = "direct"):
        self._initialize_catalog_storage()
        for spec in specs:
            self.register(spec, source=source)

    def register(
        self,
        spec: BenchmarkSpec,
        *,
        source: str = "direct",
        entry_point_group: str | None = None,
        entry_point_name: str | None = None,
        distribution_name: str | None = None,
        distribution_version: str | None = None,
    ) -> None:
        self._ensure_mutable()
        if type(spec) is not BenchmarkSpec:
            raise ConfigurationError("benchmark registration requires an exact BenchmarkSpec")
        spec.validate()
        registration = ExtensionRegistration(
            name=spec.name,
            component_kind="benchmark",
            source=source,
            entry_point_group=entry_point_group,
            entry_point_name=entry_point_name,
            distribution_name=distribution_name,
            distribution_version=distribution_version,
        )
        registration.validate()
        self._commit_validated_entries(
            ((spec.name, spec, registration),), duplicate_label="benchmark"
        )

    def _commit_plugin_entries(
        self,
        entries: tuple[tuple[str, BenchmarkSpec, ExtensionRegistration], ...],
        *,
        materialized_names: frozenset[str],
    ) -> None:
        """Atomically add plugins and replace approved declarations."""
        self._ensure_mutable()
        if type(entries) is not tuple or type(materialized_names) is not frozenset:
            raise ConfigurationError("benchmark plugin commit requires exact immutable batches")
        names = tuple((name for name, _, _ in entries))
        if len(names) != len(set(names)):
            raise ConfigurationError("duplicate benchmark registration in plugin batch")
        if not materialized_names.issubset(names):
            raise ConfigurationError(
                "benchmark materialization set is not contained in plugin batch"
            )
        next_specs = dict(self._specs)
        next_registrations = dict(self._registrations)
        for name, spec, registration in entries:
            if (
                type(name) is not str
                or type(spec) is not BenchmarkSpec
                or type(registration) is not ExtensionRegistration
                or (spec.name != name)
                or (registration.name != name)
            ):
                raise ConfigurationError("benchmark plugin registration identity mismatch")
            if name in materialized_names:
                current = next_specs.get(name)
                if (
                    type(current) is not BenchmarkSpec
                    or current.mode is not EvaluationMode.DECLARED_ONLY
                    or spec.mode is not EvaluationMode.CASES
                ):
                    raise ConfigurationError(f"benchmark is not materializable: {name}")
            elif name in next_specs:
                raise ConfigurationError(f"duplicate benchmark registration: {name}")
            next_specs[name] = spec
            next_registrations[name] = registration
        self._specs = next_specs
        self._registrations = next_registrations

    def resolve(self, name: str) -> BenchmarkSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            expected = ", ".join(self.names())
            raise ConfigurationError(
                f"unknown benchmark {name!r}; expected one of: {expected}"
            ) from exc

    def resolve_scorer(self, name: str) -> BenchmarkSpec:
        """Resolve a scorer through the unified selection protocol."""
        return self.resolve(name)

    def registration(self, name: str) -> ExtensionRegistration:
        self.resolve(name)
        return self._registrations[name]

    def names(self, *, locally_evaluable: bool | None = None) -> tuple[str, ...]:
        return tuple(
            (
                name
                for name, spec in self._specs.items()
                if locally_evaluable is None or spec.locally_evaluable is locally_evaluable
            )
        )

    def manifest(self) -> dict[str, object]:
        values = [
            {**spec.public(), "registration": self._registrations[name].public()}
            for name, spec in self._specs.items()
        ]
        return {
            "schema_version": 4,
            "status": "RETHINKSKILL_BENCHMARK_CATALOG",
            "sealed": self._sealed,
            "benchmarks": values,
            "counts": {
                "total": len(values),
                "locally_evaluable": sum(bool(item["locally_evaluable"]) for item in values),
                "external_harness": sum(
                    item["evaluation_mode"] == EvaluationMode.EXTERNAL_HARNESS.value
                    for item in values
                ),
                "declared_only": sum(
                    item["evaluation_mode"] == EvaluationMode.DECLARED_ONLY.value for item in values
                ),
            },
            "model_calls": 0,
        }

    def load_entry_points(self, *, group: str = "rethinkskill.benchmarks") -> tuple[str, ...]:
        """Import installed plugins only after an explicit caller opt-in."""
        self._ensure_mutable()
        return load_benchmark_entry_points(self, group=group)


def builtin_specs() -> tuple[BenchmarkSpec, ...]:
    """Return maintained scorer declarations through the integration layer."""
    from rethinkskill.integrations.builtin import builtin_benchmark_specs

    return builtin_benchmark_specs()


def benchmark_catalog(*, load_plugins: bool = False) -> BenchmarkCatalog:
    """Assemble maintained scorers and optionally explicit entry points."""
    catalog = BenchmarkCatalog(builtin_specs(), source="builtin")
    if load_plugins:
        catalog.load_entry_points()
    return catalog.seal()


DEFAULT_CATALOG = benchmark_catalog()

DETERMINISTIC_BENCHMARKS = DEFAULT_CATALOG.names(locally_evaluable=True)


def case_adapter(benchmark: str, *, catalog: BenchmarkCatalog | None = None) -> CaseAdapter:
    """Resolve one maintained deterministic case adapter."""
    spec = (catalog or DEFAULT_CATALOG).resolve(benchmark)
    if not spec.locally_evaluable or spec.adapter is None:
        raise ConfigurationError(f"benchmark {benchmark!r} requires an external harness")
    return spec.adapter


def _prepare_benchmark_registration(
    record: LoadedPluginSpec[BenchmarkSpec], *, group: str
) -> tuple[str, ExtensionRegistration]:
    spec = record.spec
    spec.validate()
    registration = ExtensionRegistration(
        name=spec.name,
        component_kind="benchmark",
        source="entry_point",
        entry_point_group=group,
        entry_point_name=record.entry_point_name,
        distribution_name=record.distribution_name,
        distribution_version=record.distribution_version,
    )
    registration.validate()
    return (spec.name, registration)


def load_benchmark_entry_points(
    catalog: BenchmarkCatalog, *, group: str = "rethinkskill.benchmarks"
) -> tuple[str, ...]:
    """Load validated benchmark specs only after an explicit request."""
    records = load_plugin_spec_records(group=group, expected_type=BenchmarkSpec, label="benchmark")
    existing = set(catalog.names())
    pending: set[str] = set()
    materialized: set[str] = set()
    materialization_batches: dict[str, IntegrationSpec] = {}
    prepared: list[tuple[str, BenchmarkSpec, ExtensionRegistration]] = []
    ownership = integration_catalog()
    for record in records:
        name, registration = _prepare_benchmark_registration(record, group=group)
        if name in pending:
            raise ConfigurationError(f"duplicate benchmark registration: {name}")
        if name in existing:
            declared = catalog.resolve(name)
            declared_registration = catalog.registration(name)
            try:
                integration = ownership.for_benchmark(name)
            except ConfigurationError as exc:
                raise ConfigurationError(f"duplicate benchmark registration: {name}") from exc
            if (
                declared.mode is not EvaluationMode.DECLARED_ONLY
                or declared_registration.source != "builtin"
                or integration.delivery is not AdapterDelivery.OPTIONAL_PACKAGE
                or (
                    canonical_distribution_name(record.distribution_name)
                    != canonical_distribution_name(integration.adapter_distribution)
                )
                or (record.spec.mode is not EvaluationMode.CASES)
                or (_declaration_metadata(record.spec) != _declaration_metadata(declared))
            ):
                raise ConfigurationError(f"benchmark declaration materialization rejected: {name}")
            materialized.add(name)
            materialization_batches[integration.name] = integration
        pending.add(name)
        prepared.append((name, record.spec, registration))
    for integration in materialization_batches.values():
        expected = {
            name
            for name in integration.benchmarks
            if name in existing and catalog.resolve(name).mode is EvaluationMode.DECLARED_ONLY
        }
        observed = materialized.intersection(integration.benchmarks)
        if observed != expected:
            raise ConfigurationError(
                f"benchmark declaration suite materialization must be atomic: {integration.name}"
            )
    frozen = tuple(prepared)
    catalog._commit_plugin_entries(frozen, materialized_names=frozenset(materialized))
    return tuple((name for name, _, _ in prepared))


def _declaration_metadata(spec: BenchmarkSpec) -> tuple[object, ...]:
    return (
        spec.name,
        spec.family,
        spec.domain,
        spec.metrics,
        spec.sources,
        spec.notes,
        spec.artifact_fields,
    )
