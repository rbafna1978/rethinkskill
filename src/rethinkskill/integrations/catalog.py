"""RethinkSkill integrations catalog."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rethinkskill.errors import ConfigurationError


class IntegrationTier(str, Enum):
    """Maintenance and installation boundary for one integration."""

    FIRST_PARTY = "first_party"
    FIRST_PARTY_OPTIONAL = "first_party_optional"
    SUITE_ADAPTER = "suite_adapter"
    EXTERNAL = "external"
    THIRD_PARTY = "third_party"


class ExecutionFidelity(str, Enum):
    """How closely an adapter reproduces its named benchmark environment."""

    PAPER_NATIVE = "paper_native"
    OFFLINE_PROXY = "offline_proxy"
    DATASET_BOUND = "dataset_bound"
    EXTERNAL_REQUIRED = "external_required"
    PLUGIN_DECLARED = "plugin_declared"


class AdapterDelivery(str, Enum):
    """How executable adapter code is delivered to users."""

    BUNDLED = "bundled"
    OPTIONAL_PACKAGE = "optional_package"
    DECLARED_ONLY = "declared_only"


@dataclass(frozen=True, slots=True)
class IntegrationSpec:
    """Static integration ownership independent of runtime availability."""

    name: str
    tier: IntegrationTier
    fidelity: ExecutionFidelity
    benchmarks: tuple[str, ...]
    description: str
    delivery: AdapterDelivery = AdapterDelivery.BUNDLED
    adapter_distribution: str | None = "rethinkskill"
    optional_dependencies: tuple[str, ...] = ()
    external_requirements: tuple[str, ...] = ()

    def validate(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name != self.name.lower()
            or (self.name != self.name.strip())
        ):
            raise ConfigurationError(
                f"integration name must be non-empty lowercase text: {self.name!r}"
            )
        if type(self.tier) is not IntegrationTier:
            raise ConfigurationError(f"integration tier is invalid: {self.name}: {self.tier!r}")
        if type(self.fidelity) is not ExecutionFidelity:
            raise ConfigurationError(
                f"integration execution fidelity is invalid: {self.name}: {self.fidelity!r}"
            )
        if type(self.delivery) is not AdapterDelivery:
            raise ConfigurationError(
                f"integration adapter delivery is invalid: {self.name}: {self.delivery!r}"
            )
        if (
            type(self.benchmarks) is not tuple
            or not self.benchmarks
            or len(set(self.benchmarks)) != len(self.benchmarks)
        ):
            raise ConfigurationError(
                f"integration benchmark names must be non-empty and unique: {self.name}"
            )
        if any(
            type(name) is not str or not name or name != name.lower() or (name != name.strip())
            for name in self.benchmarks
        ):
            raise ConfigurationError(f"integration benchmark names must be lowercase: {self.name}")
        if type(self.description) is not str or not self.description.strip():
            raise ConfigurationError(f"integration description is required: {self.name}")
        if self.adapter_distribution is not None and (
            type(self.adapter_distribution) is not str or not self.adapter_distribution.strip()
        ):
            raise ConfigurationError(
                f"integration adapter distribution must be non-empty text or None: {self.name}"
            )
        for label, values in (
            ("optional dependencies", self.optional_dependencies),
            ("external requirements", self.external_requirements),
        ):
            if (
                type(values) is not tuple
                or len(set(values)) != len(values)
                or (not all(type(value) is str and value.strip() for value in values))
            ):
                raise ConfigurationError(f"integration {label} must be unique strings: {self.name}")
        if self.delivery is AdapterDelivery.DECLARED_ONLY:
            if self.adapter_distribution is not None:
                raise ConfigurationError(
                    f"declared-only integration cannot name an adapter distribution: {self.name}"
                )
        elif not self.adapter_distribution:
            raise ConfigurationError(f"delivered integration requires a distribution: {self.name}")

    def public(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tier": self.tier.value,
            "execution_fidelity": self.fidelity.value,
            "benchmarks": list(self.benchmarks),
            "description": self.description,
            "adapter_delivery": self.delivery.value,
            "adapter_distribution": self.adapter_distribution,
            "optional_dependencies": list(self.optional_dependencies),
            "external_requirements": list(self.external_requirements),
        }


class IntegrationCatalog:
    """Fail-closed mapping from benchmark names to integration ownership."""

    def __init__(self, specs: tuple[IntegrationSpec, ...]):
        if type(specs) is not tuple:
            raise ConfigurationError(
                "integration catalog requires a tuple of IntegrationSpec values"
            )
        self._specs: dict[str, IntegrationSpec] = {}
        self._by_benchmark: dict[str, IntegrationSpec] = {}
        for spec in specs:
            if type(spec) is not IntegrationSpec:
                raise ConfigurationError(
                    "integration catalog requires exact IntegrationSpec values"
                )
            spec.validate()
            if spec.name in self._specs:
                raise ConfigurationError(f"duplicate integration registration: {spec.name}")
            self._specs[spec.name] = spec
            for benchmark in spec.benchmarks:
                if benchmark in self._by_benchmark:
                    owner = self._by_benchmark[benchmark].name
                    raise ConfigurationError(
                        f"benchmark belongs to multiple integrations: {benchmark}: {owner}, {spec.name}"
                    )
                self._by_benchmark[benchmark] = spec

    def resolve(self, name: str) -> IntegrationSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown integration {name!r}; expected one of: {', '.join(self._specs)}"
            ) from exc

    def for_benchmark(self, benchmark: str) -> IntegrationSpec:
        try:
            return self._by_benchmark[benchmark]
        except KeyError as exc:
            raise ConfigurationError(
                f"benchmark has no integration ownership metadata: {benchmark}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "status": "RETHINKSKILL_INTEGRATION_CATALOG",
            "integrations": [spec.public() for spec in self._specs.values()],
            "count": len(self._specs),
            "counts": {
                delivery.value: sum(spec.delivery is delivery for spec in self._specs.values())
                for delivery in AdapterDelivery
            },
            "model_calls": 0,
        }


def builtin_integration_specs() -> tuple[IntegrationSpec, ...]:
    """Load maintained benchmark ownership definitions."""
    from rethinkskill.integrations.definitions import builtin_integration_specs as definitions

    return definitions()


def integration_catalog() -> IntegrationCatalog:
    return IntegrationCatalog(builtin_integration_specs())
