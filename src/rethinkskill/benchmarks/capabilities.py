"""RethinkSkill benchmarks capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rethinkskill.benchmarks.core import (
    BenchmarkCatalog,
    BenchmarkSpec,
    EvaluationMode,
    benchmark_catalog,
)
from rethinkskill.benchmarks.harness_catalog import (
    EvaluationAuthority,
    HarnessCatalog,
    NativeHarnessSpec,
    native_harness_catalog,
)
from rethinkskill.errors import ConfigurationError
from rethinkskill.integrations.catalog import (
    AdapterDelivery,
    ExecutionFidelity,
    IntegrationCatalog,
    IntegrationSpec,
    IntegrationTier,
    integration_catalog,
)
from rethinkskill.utils.plugins import ExtensionRegistration
from rethinkskill.utils.release import canonical_distribution_name

if TYPE_CHECKING:
    from rethinkskill.benchmarks.core import BenchmarkSpec
    from rethinkskill.benchmarks.harness_catalog import NativeHarnessSpec
    from rethinkskill.integrations.catalog import IntegrationSpec
    from rethinkskill.utils.plugins import ExtensionRegistration


@dataclass(frozen=True, slots=True)
class BenchmarkCapability:
    """One benchmark name with its scorer and optional native harness."""

    scoring: BenchmarkSpec
    native: NativeHarnessSpec | None
    integration: IntegrationSpec
    scoring_registration: ExtensionRegistration | None = None
    native_registration: ExtensionRegistration | None = None

    @property
    def name(self) -> str:
        return self.scoring.name

    @property
    def scorer_available(self) -> bool:
        return self.scoring.locally_evaluable

    @property
    def adapter_available(self) -> bool:
        return self.native is not None

    @property
    def runtime_ready(self) -> bool:
        return self.native is not None and self.native.runtime_ready

    @property
    def evaluation_ready(self) -> bool:
        return self.scorer_available or (
            self.native is not None
            and self.native.evaluation_authority is EvaluationAuthority.HARNESS
        )

    @property
    def native_runnable(self) -> bool:
        return self.adapter_available and self.runtime_ready and self.evaluation_ready

    def public(self) -> dict[str, object]:
        native = self.native.public() if self.native is not None else None
        adapter_available = native is not None
        runtime_ready = bool(native and native["runtime_ready"] is True)
        evaluation_ready = self.scorer_available or (
            self.native is not None
            and self.native.evaluation_authority is EvaluationAuthority.HARNESS
        )
        return {
            "name": self.name,
            "declared": True,
            "integration": self.integration.public(),
            "scoring": self.scoring.public(),
            "scoring_registration": self.scoring_registration.public()
            if self.scoring_registration is not None
            else None,
            "scorer_available": self.scorer_available,
            "native_execution": native,
            "native_registration": self.native_registration.public()
            if self.native_registration is not None
            else None,
            "adapter_available": adapter_available,
            "runtime_ready": runtime_ready,
            "evaluation_ready": evaluation_ready,
            "native_runnable": adapter_available and runtime_ready and evaluation_ready,
        }


@dataclass(frozen=True, slots=True, init=False)
class CapabilityCatalog:
    """Fail-closed joint view of benchmark scoring and execution support."""

    scorers: BenchmarkCatalog
    harnesses: HarnessCatalog
    integrations: IntegrationCatalog

    def __init__(
        self,
        scorers: BenchmarkCatalog,
        harnesses: HarnessCatalog,
        integrations: IntegrationCatalog | None = None,
    ):
        if type(scorers) is not BenchmarkCatalog:
            raise ConfigurationError("capability selection requires an exact BenchmarkCatalog")
        if type(harnesses) is not HarnessCatalog:
            raise ConfigurationError("capability selection requires an exact HarnessCatalog")
        selected_integrations = integrations or integration_catalog()
        if type(selected_integrations) is not IntegrationCatalog:
            raise ConfigurationError("capability selection requires an exact IntegrationCatalog")
        for name in harnesses.names():
            scoring = scorers.resolve(name)
            harness = harnesses.resolve(name)
            if (
                scoring.mode is EvaluationMode.EXTERNAL_HARNESS
                and harness.evaluation_authority is not EvaluationAuthority.HARNESS
            ):
                raise ConfigurationError(
                    f"external benchmark harness must own evaluation: {name}; set evaluation_authority='harness'"
                )
            registration = harnesses.registration(name)
            try:
                integration = selected_integrations.for_benchmark(name)
            except ConfigurationError:
                integration = None
            if integration is not None:
                if (
                    registration.source == "entry_point"
                    and integration.delivery is AdapterDelivery.OPTIONAL_PACKAGE
                    and (
                        canonical_distribution_name(registration.distribution_name)
                        != canonical_distribution_name(integration.adapter_distribution)
                    )
                ):
                    raise ConfigurationError(
                        f"optional benchmark harness distribution mismatch: {name}"
                    )
        scorers.seal()
        harnesses.seal()
        object.__setattr__(self, "scorers", scorers)
        object.__setattr__(self, "harnesses", harnesses)
        object.__setattr__(self, "integrations", selected_integrations)

    def _integration_for(self, name: str) -> IntegrationSpec:
        try:
            return self.integrations.for_benchmark(name)
        except ConfigurationError:
            return IntegrationSpec(
                name="third-party",
                tier=IntegrationTier.THIRD_PARTY,
                fidelity=ExecutionFidelity.PLUGIN_DECLARED,
                benchmarks=(name,),
                description="Explicitly loaded third-party benchmark integration.",
            )

    def resolve(self, name: str) -> BenchmarkCapability:
        scoring = self.scorers.resolve(name)
        native = self.harnesses.resolve(name) if name in self.harnesses.names() else None
        return BenchmarkCapability(
            scoring=scoring,
            native=native,
            integration=self._integration_for(name),
            scoring_registration=self.scorers.registration(name),
            native_registration=self.harnesses.registration(name) if native is not None else None,
        )

    def resolve_capability(self, name: str) -> BenchmarkCapability:
        """Explicit selection interface consumed by native planning."""
        return self.resolve(name)

    def resolve_scorer(self, name: str) -> BenchmarkSpec:
        return self.resolve(name).scoring

    def resolve_native(self, name: str) -> NativeHarnessSpec:
        capability = self.resolve(name)
        if capability.native is None:
            expected = ", ".join(self.harnesses.names())
            raise ConfigurationError(
                f"benchmark has no native harness {name!r}; expected one of: {expected}"
            )
        return capability.native

    def native_registration(self, name: str) -> ExtensionRegistration:
        """Return the harness provenance selected by the joined catalog."""
        self.resolve_native(name)
        return self.harnesses.registration(name)

    def resolve_runnable(self, name: str) -> NativeHarnessSpec:
        """Resolve a harness only when its runtime dependencies are ready."""
        capability = self.resolve(name)
        if capability.native is None:
            raise ConfigurationError(f"benchmark is not runtime-ready {name!r}: {{}}")
        native = capability.native.public()
        evaluation_ready = (
            capability.scorer_available
            or capability.native.evaluation_authority is EvaluationAuthority.HARNESS
        )
        if native["runtime_ready"] is not True or not evaluation_ready:
            dependency = native.get("runtime_dependency", {})
            raise ConfigurationError(f"benchmark is not runtime-ready {name!r}: {dependency}")
        return capability.native

    def names(self, *, native_runnable: bool | None = None) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.scorers.names()
            if native_runnable is None or self.resolve(name).native_runnable is native_runnable
        )

    def manifest(self) -> dict[str, object]:
        capabilities = [self.resolve(name).public() for name in self.scorers.names()]
        native_count = sum(bool(item["native_runnable"]) for item in capabilities)
        adapter_count = sum(bool(item["adapter_available"]) for item in capabilities)
        runtime_count = sum(bool(item["runtime_ready"]) for item in capabilities)
        deterministic_count = sum(bool(item["scorer_available"]) for item in capabilities)
        scoring_only = sum(
            bool(item["scorer_available"]) and (not bool(item["adapter_available"]))
            for item in capabilities
        )
        external_only = sum(
            item["scoring"]["evaluation_mode"] == EvaluationMode.EXTERNAL_HARNESS.value
            and (not bool(item["scorer_available"]))
            and (not bool(item["adapter_available"]))
            for item in capabilities
        )
        declared_only = sum(
            item["scoring"]["evaluation_mode"] == EvaluationMode.DECLARED_ONLY.value
            and (not bool(item["scorer_available"]))
            for item in capabilities
        )
        incomplete_adapter = sum(
            item["scoring"]["evaluation_mode"] == EvaluationMode.DECLARED_ONLY.value
            and (not bool(item["scorer_available"]))
            and bool(item["adapter_available"])
            for item in capabilities
        )
        unavailable = sum(
            not bool(item["scorer_available"]) and (not bool(item["adapter_available"]))
            for item in capabilities
        )
        external_adapter = sum(
            item["scoring"]["evaluation_mode"] == EvaluationMode.EXTERNAL_HARNESS.value
            and bool(item["adapter_available"])
            for item in capabilities
        )
        return {
            "schema_version": 5,
            "status": "RETHINKSKILL_CAPABILITY_CATALOG",
            "sealed": self.scorers.sealed and self.harnesses.sealed,
            "capabilities": capabilities,
            "counts": {
                "total": len(capabilities),
                "deterministic_scorer": deterministic_count,
                "adapter_available": adapter_count,
                "runtime_ready": runtime_count,
                "native_runnable": native_count,
                "scoring_only": scoring_only,
                "deterministic_only": scoring_only,
                "external_adapter": external_adapter,
                "external_only": external_only,
                "declared_only": declared_only,
                "incomplete_adapter": incomplete_adapter,
                "unavailable": unavailable,
            },
            "model_calls": 0,
        }


def capability_catalog(
    *, load_benchmark_plugins: bool = False, load_harness_plugins: bool = False
) -> CapabilityCatalog:
    """Construct one validated benchmark selection boundary."""
    return CapabilityCatalog(
        benchmark_catalog(load_plugins=load_benchmark_plugins),
        native_harness_catalog(load_plugins=load_harness_plugins),
        integration_catalog(),
    )
