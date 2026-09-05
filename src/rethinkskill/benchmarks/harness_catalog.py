"""RethinkSkill benchmarks harness catalog."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from rethinkskill.errors import ConfigurationError
from rethinkskill.integrations.catalog import AdapterDelivery, integration_catalog
from rethinkskill.runtime.types import BenchmarkHarness
from rethinkskill.utils.plugins import (
    ExtensionRegistration,
    LoadedPluginSpec,
    SealableCatalog,
    load_plugin_spec_records,
    prepare_plugin_spec_batch,
)
from rethinkskill.utils.release import canonical_distribution_name, freeze_mapping_manifest


class EvaluationAuthority(str, Enum):
    """Component responsible for turning execution state into a verdict."""

    CASE_SCORER = "case_scorer"
    HARNESS = "harness"


@dataclass(frozen=True, slots=True)
class NativeHarnessSpec:
    name: str
    harness: BenchmarkHarness
    execution_scope: str
    source: str
    evaluation_authority: EvaluationAuthority = EvaluationAuthority.CASE_SCORER

    def validate(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name != self.name.lower()
            or (self.name != self.name.strip())
        ):
            raise ConfigurationError(f"native harness name must be lowercase: {self.name!r}")
        if (
            type(self.execution_scope) is not str
            or not self.execution_scope.strip()
            or type(self.source) is not str
            or (not self.source.strip())
        ):
            raise ConfigurationError(f"native harness scope and source are required: {self.name}")
        if type(self.evaluation_authority) is not EvaluationAuthority:
            raise ConfigurationError(
                f"native harness evaluation authority is invalid: {self.name}: {self.evaluation_authority!r}"
            )
        missing = tuple(
            method
            for method in ("load_tasks", "render", "evaluate")
            if not callable(getattr(self.harness, method, None))
        )
        if missing:
            raise ConfigurationError(
                f"native harness contract is incomplete for {self.name}: {', '.join(missing)}"
            )

    def dependency_manifest(self) -> dict[str, object]:
        probe = getattr(self.harness, "dependency_manifest", None)
        if not callable(probe):
            return {}
        return freeze_mapping_manifest(
            self.harness, method_name="dependency_manifest", label=f"{self.name} runtime dependency"
        )

    @staticmethod
    def _runtime_ready(dependency: dict[str, object]) -> bool:
        if "ready" in dependency:
            return dependency["ready"] is True
        availability = [value for key, value in dependency.items() if key.endswith("_available")]
        return all(value is True for value in availability)

    @property
    def runtime_ready(self) -> bool:
        return self._runtime_ready(self.dependency_manifest())

    def public(self) -> dict[str, object]:
        dependency = self.dependency_manifest()
        runtime_ready = self._runtime_ready(dependency)
        value: dict[str, object] = {
            "name": self.name,
            "execution_scope": self.execution_scope,
            "source": self.source,
            "native": True,
            "adapter_available": True,
            "runtime_ready": runtime_ready,
            "evaluation_authority": self.evaluation_authority.value,
            "task_execution": "harness_managed"
            if callable(getattr(self.harness, "execute_task", None))
            else "single_model_call",
        }
        if dependency:
            value["runtime_dependency"] = dependency
        return value


class HarnessCatalog(SealableCatalog):
    _catalog_label = "native harness"

    def __init__(self, specs: Iterable[NativeHarnessSpec] = (), *, source: str = "direct"):
        self._initialize_catalog_storage()
        for spec in specs:
            self.register(spec, source=source)

    def register(
        self,
        spec: NativeHarnessSpec,
        *,
        source: str = "direct",
        entry_point_group: str | None = None,
        entry_point_name: str | None = None,
        distribution_name: str | None = None,
        distribution_version: str | None = None,
    ) -> None:
        self._ensure_mutable()
        if type(spec) is not NativeHarnessSpec:
            raise ConfigurationError(
                "native harness registration requires an exact NativeHarnessSpec"
            )
        spec.validate()
        registration = ExtensionRegistration(
            name=spec.name,
            component_kind="native_harness",
            source=source,
            entry_point_group=entry_point_group,
            entry_point_name=entry_point_name,
            distribution_name=distribution_name,
            distribution_version=distribution_version,
        )
        registration.validate()
        self._commit_validated_entries(
            ((spec.name, spec, registration),), duplicate_label="native harness"
        )

    def resolve(self, name: str) -> NativeHarnessSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ConfigurationError(
                f"benchmark has no native harness {name!r}; expected one of: {', '.join(self._specs)}"
            ) from exc

    def resolve_native(self, name: str) -> NativeHarnessSpec:
        """Resolve a harness through the unified capability catalog."""
        return self.resolve(name)

    def registration(self, name: str) -> ExtensionRegistration:
        self.resolve(name)
        return self._registrations[name]

    def native_registration(self, name: str) -> ExtensionRegistration:
        """Expose catalog-owned provenance to native planning."""
        return self.registration(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def manifest(self) -> dict[str, object]:
        specs = [
            {**spec.public(), "registration": self._registrations[name].public()}
            for name, spec in self._specs.items()
        ]
        return {
            "schema_version": 3,
            "status": "RETHINKSKILL_NATIVE_HARNESS_CATALOG",
            "sealed": self._sealed,
            "harnesses": specs,
            "count": len(specs),
            "counts": {
                "adapter_available": len(specs),
                "runtime_ready": sum(bool(spec["runtime_ready"]) for spec in specs),
            },
            "model_calls": 0,
        }

    def load_entry_points(self, *, group: str = "rethinkskill.harnesses") -> tuple[str, ...]:
        """Explicitly load third-party harness plugins."""
        self._ensure_mutable()
        return load_harness_entry_points(self, group=group)


def builtin_native_harnesses() -> tuple[NativeHarnessSpec, ...]:
    """Return maintained native adapters in deterministic order."""
    from rethinkskill.integrations.builtin import builtin_native_harness_specs

    return builtin_native_harness_specs()


def native_harness_catalog(*, load_plugins: bool = False) -> HarnessCatalog:
    """Assemble maintained harnesses and optionally explicit entry points."""
    catalog = HarnessCatalog(builtin_native_harnesses(), source="builtin")
    if load_plugins:
        catalog.load_entry_points()
    return catalog.seal()


def _prepare_harness_registration(
    record: LoadedPluginSpec[NativeHarnessSpec], *, group: str
) -> tuple[str, ExtensionRegistration]:
    spec = record.spec
    spec.validate()
    registration = ExtensionRegistration(
        name=spec.name,
        component_kind="native_harness",
        source="entry_point",
        entry_point_group=group,
        entry_point_name=record.entry_point_name,
        distribution_name=record.distribution_name,
        distribution_version=record.distribution_version,
    )
    registration.validate()
    return (spec.name, registration)


def load_harness_entry_points(
    catalog: HarnessCatalog, *, group: str = "rethinkskill.harnesses"
) -> tuple[str, ...]:
    """Load validated harness specs only after an explicit request."""
    records = load_plugin_spec_records(
        group=group, expected_type=NativeHarnessSpec, label="native harness"
    )
    ownership = integration_catalog()
    touched: dict[str, tuple[str, ...]] = {}
    observed_names = {record.spec.name for record in records}
    for record in records:
        try:
            integration = ownership.for_benchmark(record.spec.name)
        except ConfigurationError:
            continue
        if integration.delivery is not AdapterDelivery.OPTIONAL_PACKAGE:
            continue
        if canonical_distribution_name(record.distribution_name) != canonical_distribution_name(
            integration.adapter_distribution
        ):
            raise ConfigurationError(
                f"optional benchmark harness distribution mismatch: {record.spec.name}"
            )
        touched[integration.name] = integration.benchmarks
    for integration_name, benchmark_names in touched.items():
        if observed_names.intersection(benchmark_names) != set(benchmark_names):
            raise ConfigurationError(
                f"optional benchmark harness suite registration must be atomic: {integration_name}"
            )
    prepared = prepare_plugin_spec_batch(
        records,
        existing_names=catalog.names(),
        duplicate_label="native harness",
        prepare=lambda record: _prepare_harness_registration(record, group=group),
    )
    catalog._commit_validated_entries(prepared, duplicate_label="native harness")
    return tuple((name for name, _, _ in prepared))
