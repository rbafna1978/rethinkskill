"""RethinkSkill providers catalog."""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from rethinkskill.errors import AuthorizationRequiredError, ConfigurationError
from rethinkskill.providers.transport import TransportConfig, TransportKind, validate_provider_name
from rethinkskill.runtime.tasks import RenderedTask
from rethinkskill.runtime.types import ModelExecutor, ModelOutcome, freeze_model_outcome
from rethinkskill.utils.plugins import (
    LoadedPluginSpec,
    SealableCatalog,
    load_plugin_spec_records,
    prepare_plugin_spec_batch,
)
from rethinkskill.utils.release import freeze_component_manifest


def _validate_optional_identity_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ConfigurationError(f"invalid {label}: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class ProviderRegistration:
    """Catalog-owned provider origin and installed version metadata."""

    name: str
    transport_kind: TransportKind
    source: str
    entry_point_group: str | None = None
    entry_point_name: str | None = None
    distribution_name: str | None = None
    distribution_version: str | None = None

    @classmethod
    def from_public(cls, value: object) -> ProviderRegistration:
        """Reconstruct serialized provider provenance."""
        keys = {
            "name",
            "transport_kind",
            "source",
            "entry_point_group",
            "entry_point_name",
            "distribution_name",
            "distribution_version",
        }
        if type(value) is not dict or set(value) != keys:
            raise ConfigurationError("provider registration has an invalid public schema")
        try:
            transport_kind = TransportKind(value["transport_kind"])
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("provider registration transport kind is invalid") from exc
        registration = cls(
            name=cast(str, value["name"]),
            transport_kind=transport_kind,
            source=cast(str, value["source"]),
            entry_point_group=cast(str | None, value["entry_point_group"]),
            entry_point_name=cast(str | None, value["entry_point_name"]),
            distribution_name=cast(str | None, value["distribution_name"]),
            distribution_version=cast(str | None, value["distribution_version"]),
        )
        registration.validate()
        if registration.public() != value:
            raise ConfigurationError("provider registration public fields are inconsistent")
        return registration

    def validate(self) -> None:
        validate_provider_name(self.name)
        if type(self.transport_kind) is not TransportKind:
            raise ConfigurationError(f"invalid provider transport kind: {self.transport_kind!r}")
        if type(self.source) is not str or self.source not in {"builtin", "direct", "entry_point"}:
            raise ConfigurationError(f"invalid provider registration source: {self.source!r}")
        for label, value in (
            ("provider entry-point group", self.entry_point_group),
            ("provider entry-point name", self.entry_point_name),
            ("provider distribution name", self.distribution_name),
            ("provider distribution version", self.distribution_version),
        ):
            _validate_optional_identity_text(value, label=label)
        entry_point_values = (self.entry_point_group, self.entry_point_name)
        if self.source == "entry_point":
            if any(value is None for value in entry_point_values):
                raise ConfigurationError("entry-point provider registration is missing identity")
        elif any(value is not None for value in entry_point_values):
            raise ConfigurationError("non-plugin provider cannot declare entry-point identity")
        if self.source != "entry_point" and (
            self.distribution_name is not None
            or self.distribution_version is not None
        ):
            raise ConfigurationError("non-plugin provider cannot declare distribution metadata")
        if (self.distribution_name is None) != (self.distribution_version is None):
            raise ConfigurationError("provider distribution name and version must be paired")

    def public(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "transport_kind": self.transport_kind.value,
            "source": self.source,
            "entry_point_group": self.entry_point_group,
            "entry_point_name": self.entry_point_name,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
        }


if TYPE_CHECKING:
    from rethinkskill.runtime.types import ModelExecutor


class ProviderInterface(str, Enum):
    """Process boundary used by one provider implementation."""

    CLI = "cli"
    HTTP = "http"


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Selection metadata plus the executor implementation for one provider."""

    kind: TransportKind
    executor_type: type
    interface: ProviderInterface
    credential_mode: str
    default_launcher: str | None = None
    supported_roles: tuple[str, ...] = ("target", "optimizer")
    name: str | None = None

    @property
    def identifier(self) -> str:
        """Return the catalog key, defaulting to the built-in transport name."""
        return self.name or self.kind.value

    def validate(self) -> None:
        if type(self.kind) is not TransportKind:
            raise ConfigurationError(f"provider kind is invalid: {self.kind!r}")
        if type(self.interface) is not ProviderInterface:
            raise ConfigurationError(f"provider interface is invalid: {self.interface!r}")
        validate_provider_name(self.kind if self.name is None else self.name)
        if not isinstance(self.executor_type, type):
            raise ConfigurationError(f"provider executor must be a type: {self.identifier}")
        prepare = getattr(self.executor_type, "prepare", None)
        if not callable(prepare):
            raise ConfigurationError(f"provider executor has no prepare method: {self.identifier}")
        if type(self.credential_mode) is not str or not self.credential_mode.strip():
            raise ConfigurationError(f"provider credential mode is empty: {self.identifier}")
        if self.default_launcher is not None and (
            type(self.default_launcher) is not str
            or not self.default_launcher.strip()
            or "\x00" in self.default_launcher
        ):
            raise ConfigurationError(f"provider default launcher is invalid: {self.identifier}")
        if self.interface is ProviderInterface.CLI and (not self.default_launcher):
            raise ConfigurationError(f"CLI provider default launcher is empty: {self.identifier}")
        if self.interface is ProviderInterface.HTTP and self.default_launcher:
            raise ConfigurationError(f"HTTP provider cannot declare a launcher: {self.identifier}")
        if (
            self.interface is ProviderInterface.HTTP
            and self.kind is not TransportKind.OPENAI_COMPATIBLE
        ):
            raise ConfigurationError(
                f"HTTP provider must use the openai-compatible transport contract: {self.identifier}"
            )
        if self.interface is ProviderInterface.CLI and self.kind is TransportKind.OPENAI_COMPATIBLE:
            raise ConfigurationError(
                f"CLI provider cannot use the openai-compatible transport contract: {self.identifier}"
            )
        if (
            type(self.supported_roles) is not tuple
            or not self.supported_roles
            or len(set(self.supported_roles)) != len(self.supported_roles)
            or any(
                type(role) is not str or role not in {"target", "optimizer"}
                for role in self.supported_roles
            )
        ):
            raise ConfigurationError(
                f"invalid provider roles for {self.identifier}: {self.supported_roles!r}"
            )

    @property
    def default_runtime_detected(self) -> bool:
        """Report zero-call local discovery, not configured execution readiness."""
        if self.interface is ProviderInterface.HTTP:
            return True
        assert self.default_launcher is not None
        return shutil.which(self.default_launcher) is not None

    def prepare(
        self, config: TransportConfig, *, authorized: bool, role: str | None = None
    ) -> ModelExecutor:
        self.validate_config(config, role=role)
        if self.name is not None and role is None:
            raise ConfigurationError(f"provider {self.identifier!r} requires an explicit role")
        if type(authorized) is not bool:
            raise ConfigurationError("authorized must be boolean")
        if not authorized:
            raise AuthorizationRequiredError(
                "provider preparation requires explicit model-call authorization"
            )
        prepared_config = (
            replace(config, launcher=self.default_launcher)
            if self.interface is ProviderInterface.CLI and config.launcher is None
            else config
        )
        return self.executor_type.prepare(prepared_config, authorized=authorized)

    def validate_config(self, config: TransportConfig, *, role: str | None = None) -> None:
        """Validate selection and static transport fields without live calls."""
        if type(config) is not TransportConfig:
            raise ConfigurationError("transport config must be an exact TransportConfig")
        if config.kind is not self.kind or config.provider_name != self.identifier:
            raise ConfigurationError(
                f"provider selection/config mismatch: {self.identifier} != {config.provider_name}"
            )
        if role is not None:
            if type(role) is not str or role not in {"target", "optimizer"}:
                raise ConfigurationError(f"invalid provider role: {role!r}")
            if role not in self.supported_roles:
                raise ConfigurationError(
                    f"provider {self.identifier!r} does not support the {role} role"
                )
        config.validate_static()

    def public(self) -> dict[str, object]:
        return {
            "kind": self.identifier,
            "transport_kind": self.kind.value,
            "interface": self.interface.value,
            "credential_mode": self.credential_mode,
            "adapter_available": True,
            "default_launcher": self.default_launcher,
            "default_runtime_detected": self.default_runtime_detected,
            "configured_readiness": "requires TransportConfig and authorization",
            "supported_roles": list(self.supported_roles),
        }


def validate_prepared_executor(value: object) -> ModelExecutor:
    """Reject plugin preparation results that lack the executor contract."""
    try:
        public_manifest = value.public_manifest
        execute = value.execute
        if not callable(public_manifest) or not callable(execute):
            raise TypeError("executor methods are not callable")
    except Exception as exc:
        raise ConfigurationError(
            f"provider prepared executor contract is invalid: {type(exc).__name__}"
        ) from exc
    return cast(ModelExecutor, value)


@dataclass(frozen=True, slots=True)
class RegisteredProviderExecutor:
    """Delegate execution while adding catalog-owned origin evidence."""

    executor: ModelExecutor
    registration: ProviderRegistration

    def public_manifest(self) -> dict[str, object]:
        underlying = freeze_component_manifest(self.executor, label="provider executor")
        return {**underlying, "provider_registration": self.registration.public()}

    def execute(
        self, rendered: RenderedTask, *, workspace: Path, timeout_seconds: int
    ) -> ModelOutcome:
        return freeze_model_outcome(
            self.executor.execute(rendered, workspace=workspace, timeout_seconds=timeout_seconds)
        )


if TYPE_CHECKING:
    from rethinkskill.runtime.types import ModelExecutor


class ProviderCatalog(SealableCatalog):
    """Fail-closed registry used by every native provider selection."""

    _catalog_label = "provider"

    def __init__(self, specs: Iterable[ProviderSpec] = (), *, source: str = "direct"):
        self._initialize_catalog_storage()
        for spec in specs:
            self.register(spec, source=source)

    def register(
        self,
        spec: ProviderSpec,
        *,
        source: str = "direct",
        entry_point_group: str | None = None,
        entry_point_name: str | None = None,
        distribution_name: str | None = None,
        distribution_version: str | None = None,
    ) -> None:
        self._ensure_mutable()
        if type(spec) is not ProviderSpec:
            raise ConfigurationError("provider registration requires an exact ProviderSpec")
        spec.validate()
        name = spec.identifier
        registration = ProviderRegistration(
            name=name,
            transport_kind=spec.kind,
            source=source,
            entry_point_group=entry_point_group,
            entry_point_name=entry_point_name,
            distribution_name=distribution_name,
            distribution_version=distribution_version,
        )
        registration.validate()
        self._commit_validated_entries(((name, spec, registration),), duplicate_label="provider")

    def resolve(self, name: TransportKind | str) -> ProviderSpec:
        normalized = validate_provider_name(name)
        try:
            return self._specs[normalized]
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown provider {normalized!r}; expected one of: {', '.join(self.names())}"
            ) from exc

    def validate_config(self, config: TransportConfig, *, role: str | None = None) -> None:
        """Validate a selected provider and config without preparing an executor."""
        if type(config) is not TransportConfig:
            raise ConfigurationError("transport config must be an exact TransportConfig")
        self.resolve(config.provider_name).validate_config(config, role=role)

    def prepare(
        self, config: TransportConfig, *, authorized: bool, role: str | None = None
    ) -> ModelExecutor:
        spec = self.resolve(config.provider_name)
        registration = self._registrations[spec.identifier]
        spec.validate_config(config, role=role)
        if type(authorized) is not bool:
            raise ConfigurationError("authorized must be boolean")
        if not authorized:
            raise AuthorizationRequiredError(
                "provider preparation requires explicit model-call authorization"
            )
        try:
            candidate = spec.prepare(config, authorized=True, role=role)
        except Exception as exc:
            if registration.source == "entry_point":
                raise ConfigurationError(
                    f"provider plugin {spec.identifier!r} preparation failed: {type(exc).__name__}"
                ) from exc
            raise
        executor = validate_prepared_executor(candidate)
        return RegisteredProviderExecutor(executor=executor, registration=registration)

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def load_entry_points(self, *, group: str = "rethinkskill.providers") -> tuple[str, ...]:
        """Explicitly load third-party provider plugins."""
        self._ensure_mutable()
        return load_provider_entry_points(self, group=group)

    def manifest(self) -> dict[str, object]:
        providers = [
            {**spec.public(), "registration": self._registrations[name].public()}
            for name, spec in self._specs.items()
        ]
        return {
            "schema_version": 5,
            "status": "RETHINKSKILL_PROVIDER_CATALOG",
            "sealed": self._sealed,
            "providers": providers,
            "count": len(providers),
            "counts": {
                "total": len(providers),
                "adapter_available": len(providers),
                "default_runtime_detected": sum(
                    bool(spec["default_runtime_detected"]) for spec in providers
                ),
                "cli": sum(spec["interface"] == ProviderInterface.CLI.value for spec in providers),
                "http": sum(
                    spec["interface"] == ProviderInterface.HTTP.value for spec in providers
                ),
            },
            "model_calls": 0,
        }


def builtin_provider_specs() -> tuple[ProviderSpec, ...]:
    """Return maintained provider registrations in stable CLI order."""
    from rethinkskill.providers.builtin import (
        claude_code_provider_spec,
        codex_provider_spec,
        gemini_cli_provider_spec,
        openai_compatible_provider_spec,
    )

    return (
        codex_provider_spec(),
        claude_code_provider_spec(),
        gemini_cli_provider_spec(),
        openai_compatible_provider_spec(),
    )


def provider_catalog(*, load_plugins: bool = False) -> ProviderCatalog:
    """Assemble maintained providers and optionally explicit entry points."""
    catalog = ProviderCatalog(builtin_provider_specs(), source="builtin")
    if load_plugins:
        catalog.load_entry_points()
    return catalog.seal()


def _prepare_provider_registration(
    record: LoadedPluginSpec[ProviderSpec], *, group: str
) -> tuple[str, ProviderRegistration]:
    spec = record.spec
    try:
        spec.validate()
        registration = ProviderRegistration(
            name=spec.identifier,
            transport_kind=spec.kind,
            source="entry_point",
            entry_point_group=group,
            entry_point_name=record.entry_point_name,
            distribution_name=record.distribution_name,
            distribution_version=record.distribution_version,
        )
        registration.validate()
    except Exception as exc:
        raise ConfigurationError(
            f"provider plugin validation failed: {type(exc).__name__}"
        ) from exc
    return (spec.identifier, registration)


def load_provider_entry_points(
    catalog: ProviderCatalog, *, group: str = "rethinkskill.providers"
) -> tuple[str, ...]:
    """Load validated provider specs only after an explicit request."""
    records = load_plugin_spec_records(group=group, expected_type=ProviderSpec, label="provider")
    prepared = prepare_plugin_spec_batch(
        records,
        existing_names=catalog.names(),
        duplicate_label="provider",
        prepare=lambda record: _prepare_provider_registration(record, group=group),
    )
    catalog._commit_validated_entries(prepared, duplicate_label="provider")
    return tuple((name for name, _, _ in prepared))


def prepare_executor(
    config: TransportConfig,
    *,
    authorized: bool,
    catalog: ProviderCatalog | None = None,
    role: str | None = None,
):
    """Prepare one selected provider executor."""
    return (catalog or provider_catalog()).prepare(config, authorized=authorized, role=role)
