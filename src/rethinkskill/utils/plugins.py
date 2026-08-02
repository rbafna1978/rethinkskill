"""RethinkSkill utils plugins."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from types import MappingProxyType
from typing import Generic, TypeVar, cast

from rethinkskill.errors import ConfigurationError
from rethinkskill.utils.release import (
    distribution_file_identity_from_metadata,
    distribution_inventory_uses_indirection,
    observe_distribution_file_identity,
    validate_distribution_file_identity,
)
from rethinkskill.utils.serde import freeze_json_mapping, thaw_json_mapping

CatalogT = TypeVar("CatalogT", bound="SealableCatalog")


class SealableCatalog:
    """Internal mutable-build/read-only-selection catalog base."""

    _catalog_label = "catalog"
    _sealed: bool
    _specs: Mapping[str, object]
    _registrations: Mapping[str, object]

    def _initialize_catalog_storage(self) -> None:
        self._sealed = False
        self._specs = {}
        self._registrations = {}

    @property
    def sealed(self) -> bool:
        return self._sealed

    def _ensure_mutable(self) -> None:
        if self._sealed:
            raise ConfigurationError(f"{self._catalog_label} catalog is sealed")

    def _commit_validated_entries(
        self, entries: tuple[tuple[str, object, object], ...], *, duplicate_label: str
    ) -> None:
        """Commit a fully validated batch through copy-on-write indexes."""
        self._ensure_mutable()
        if type(entries) is not tuple:
            raise ConfigurationError("catalog registration batch must be an exact tuple")
        if type(duplicate_label) is not str or not duplicate_label.strip():
            raise ConfigurationError("catalog duplicate label must be non-empty text")
        if not entries:
            return
        next_specs = dict(self._specs)
        next_registrations = dict(self._registrations)
        for entry in entries:
            if type(entry) is not tuple or len(entry) != 3:
                raise ConfigurationError("catalog registration entries must be exact triples")
            name, spec, registration = entry
            if type(name) is not str or not name:
                raise ConfigurationError("catalog registration name must be non-empty text")
            if getattr(registration, "name", None) != name:
                raise ConfigurationError(
                    f"{duplicate_label} registration identity mismatch: {name}"
                )
            if name in next_specs:
                raise ConfigurationError(f"duplicate {duplicate_label} registration: {name}")
            next_specs[name] = spec
            next_registrations[name] = registration
        self._specs = next_specs
        self._registrations = next_registrations

    def seal(self: CatalogT) -> CatalogT:
        if not self._sealed:
            self._specs = MappingProxyType(dict(self._specs))
            self._registrations = MappingProxyType(dict(self._registrations))
            self._sealed = True
        return self


_COMPONENT_KINDS = {"benchmark", "native_harness", "optimizer_strategy"}

_SOURCES = {"builtin", "direct", "entry_point"}


def _validate_identity_text(value: object, *, label: str, optional: bool) -> str | None:
    if value is None and optional:
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
class ExtensionRegistration:
    """Origin metadata supplied by the catalog rather than extension code."""

    name: str
    component_kind: str
    source: str
    entry_point_group: str | None = None
    entry_point_name: str | None = None
    distribution_name: str | None = None
    distribution_version: str | None = None
    distribution_identity: Mapping[str, object] | None = None
    _distribution_metadata: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.distribution_identity is not None:
            object.__setattr__(
                self,
                "distribution_identity",
                validate_distribution_file_identity(self.distribution_identity),
            )

    @classmethod
    def from_public(cls, value: object) -> ExtensionRegistration:
        """Reconstruct and validate one exact serialized registration."""
        keys = {
            "name",
            "component_kind",
            "source",
            "entry_point_group",
            "entry_point_name",
            "distribution_name",
            "distribution_version",
            "external_source_freeze_required",
        }
        accepted = (keys, keys | {"distribution_identity"})
        if type(value) is not dict or set(value) not in accepted:
            raise ConfigurationError("extension registration has an invalid public schema")
        if type(value["external_source_freeze_required"]) is not bool:
            raise ConfigurationError("extension registration freeze requirement must be boolean")
        registration = cls(
            name=cast(str, value["name"]),
            component_kind=cast(str, value["component_kind"]),
            source=cast(str, value["source"]),
            entry_point_group=cast(str | None, value["entry_point_group"]),
            entry_point_name=cast(str | None, value["entry_point_name"]),
            distribution_name=cast(str | None, value["distribution_name"]),
            distribution_version=cast(str | None, value["distribution_version"]),
            distribution_identity=cast(
                Mapping[str, object] | None, value.get("distribution_identity")
            ),
        )
        registration.validate()
        if registration.public() != value:
            raise ConfigurationError("extension registration public fields are inconsistent")
        return registration

    def validate(self) -> None:
        name = _validate_identity_text(self.name, label="extension name", optional=False)
        assert name is not None
        if name != name.lower() or name != name.strip():
            raise ConfigurationError(f"extension name must be lowercase: {name!r}")
        if type(self.component_kind) is not str or self.component_kind not in _COMPONENT_KINDS:
            raise ConfigurationError(f"invalid extension component kind: {self.component_kind!r}")
        if type(self.source) is not str or self.source not in _SOURCES:
            raise ConfigurationError(f"invalid extension registration source: {self.source!r}")
        for label, value in (
            ("extension entry-point group", self.entry_point_group),
            ("extension entry-point name", self.entry_point_name),
            ("extension distribution name", self.distribution_name),
            ("extension distribution version", self.distribution_version),
        ):
            _validate_identity_text(value, label=label, optional=True)
        entry_point_identity = (self.entry_point_group, self.entry_point_name)
        if self.source == "entry_point":
            if any(value is None for value in entry_point_identity):
                raise ConfigurationError("entry-point extension registration is missing identity")
        elif any(value is not None for value in entry_point_identity):
            raise ConfigurationError("non-plugin extension cannot declare entry-point identity")
        if self.source != "entry_point" and (
            self.distribution_name is not None
            or self.distribution_version is not None
            or self.distribution_identity is not None
        ):
            raise ConfigurationError("non-plugin extension cannot declare distribution identity")
        if (self.distribution_name is None) != (self.distribution_version is None):
            raise ConfigurationError("extension distribution name and version must be paired")
        if self.distribution_identity is not None:
            if self.distribution_name is None or self.distribution_version is None:
                raise ConfigurationError("extension distribution content identity is incomplete")
            validate_distribution_file_identity(
                self.distribution_identity,
                name=self.distribution_name,
                version=self.distribution_version,
            )

    def observe_distribution_identity(self) -> dict[str, object]:
        """Observe current plugin package content without raising raw failures."""
        return observe_distribution_file_identity(
            self._distribution_metadata,
            name=self.distribution_name,
            version=self.distribution_version,
        )

    def require_current_distribution_identity(self) -> dict[str, object] | None:
        """Fail closed when a selected entry-point plugin is unfrozen or drifts."""
        self.validate()
        if self.source != "entry_point":
            return None
        if self.distribution_identity is None:
            raise ConfigurationError(f"entry-point extension source is not frozen: {self.name}")
        observed = self.observe_distribution_identity()
        expected = thaw_json_mapping(self.distribution_identity)
        if observed != expected:
            raise ConfigurationError(f"entry-point extension source identity drifted: {self.name}")
        return observed

    def public(self) -> dict[str, object]:
        self.validate()
        value = {
            "name": self.name,
            "component_kind": self.component_kind,
            "source": self.source,
            "entry_point_group": self.entry_point_group,
            "entry_point_name": self.entry_point_name,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "external_source_freeze_required": self.source != "builtin"
            and self.distribution_identity is None,
        }
        if self.distribution_identity is not None:
            value["distribution_identity"] = thaw_json_mapping(self.distribution_identity)
        return value


_Spec = TypeVar("_Spec")

_Registration = TypeVar("_Registration")


@dataclass(frozen=True, slots=True)
class LoadedPluginSpec(Generic[_Spec]):
    """One validated entry-point load plus package-level identity metadata."""

    spec: _Spec
    entry_point_name: str
    distribution_name: str | None
    distribution_version: str | None
    distribution_identity: Mapping[str, object] | None = None
    _distribution_metadata: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.distribution_identity is not None:
            object.__setattr__(
                self, "distribution_identity", freeze_json_mapping(self.distribution_identity)
            )


def _optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    text = str(value)
    if (
        not text
        or len(text) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise TypeError(f"invalid {label}")
    return text


def _distribution_identity(
    entry_point: object,
) -> tuple[str | None, str | None, Mapping[str, object] | None, object | None]:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return (None, None, None, None)
    package_name = None
    package_metadata = getattr(distribution, "metadata", None)
    if package_metadata is not None:
        package_name = package_metadata.get("Name")
    name = _optional_text(package_name, label="plugin distribution name")
    version = _optional_text(
        getattr(distribution, "version", None), label="plugin distribution version"
    )
    identity = None
    if (
        name is not None
        and version is not None
        and (getattr(distribution, "files", None) is not None)
        and callable(getattr(distribution, "locate_file", None))
        and (not distribution_inventory_uses_indirection(distribution))
    ):
        identity = validate_distribution_file_identity(
            distribution_file_identity_from_metadata(distribution, name=name, version=version),
            name=name,
            version=version,
        )
    return (name, version, identity, distribution)


def load_plugin_spec_records(
    *, group: str, expected_type: type[_Spec], label: str
) -> tuple[LoadedPluginSpec[_Spec], ...]:
    """Load exact spec values with package identity and redacted failures.

    Subclasses are rejected because they can override validation, manifest, or
    preparation methods that belong to the core trust boundary.
    """
    try:
        discovered = metadata.entry_points()
        selected = (
            discovered.select(group=group)
            if hasattr(discovered, "select")
            else discovered.get(group, ())
        )
        entry_points = tuple(
            sorted(selected, key=lambda item: (str(item.name), str(getattr(item, "value", ""))))
        )
    except Exception as exc:
        raise ConfigurationError(f"{label} plugin discovery failed: {type(exc).__name__}") from exc
    records: list[LoadedPluginSpec[_Spec]] = []
    for entry_point in entry_points:
        name = str(entry_point.name)
        try:
            (
                distribution_name,
                distribution_version,
                distribution_identity,
                distribution_metadata,
            ) = _distribution_identity(entry_point)
            value = entry_point.load()
            candidate = (
                value if type(value) is expected_type else value() if callable(value) else value
            )
            if type(candidate) is expected_type:
                values = (candidate,)
            elif isinstance(candidate, Iterable) and (
                not isinstance(candidate, (str, bytes, bytearray, Mapping))
            ):
                values = tuple(candidate)
            else:
                values = ()
            if not values or not all(type(spec) is expected_type for spec in values):
                raise TypeError(f"expected one or more {expected_type.__name__} values")
            observed_name, observed_version, observed_identity, observed_metadata = (
                _distribution_identity(entry_point)
            )
            if (
                observed_name != distribution_name
                or observed_version != distribution_version
                or observed_identity != distribution_identity
            ):
                raise ConfigurationError("plugin distribution changed while loading")
            distribution_metadata = observed_metadata
        except Exception as exc:
            raise ConfigurationError(
                f"{label} plugin {name!r} loading failed: {type(exc).__name__}"
            ) from exc
        records.extend(
            LoadedPluginSpec(
                spec=spec,
                entry_point_name=name,
                distribution_name=distribution_name,
                distribution_version=distribution_version,
                distribution_identity=distribution_identity,
                _distribution_metadata=distribution_metadata,
            )
            for spec in values
        )
    return tuple(records)


def prepare_plugin_spec_batch(
    records: tuple[LoadedPluginSpec[_Spec], ...],
    *,
    existing_names: Iterable[str],
    duplicate_label: str,
    prepare: Callable[[LoadedPluginSpec[_Spec]], tuple[str, _Registration]],
) -> tuple[tuple[str, _Spec, _Registration], ...]:
    """Validate an entire plugin batch before any catalog mutation."""
    if type(records) is not tuple:
        raise ConfigurationError("plugin registration records must be an exact tuple")
    if type(duplicate_label) is not str or not duplicate_label.strip():
        raise ConfigurationError("plugin duplicate label must be non-empty text")
    existing = set(existing_names)
    pending: set[str] = set()
    prepared: list[tuple[str, _Spec, _Registration]] = []
    for record in records:
        name, registration = prepare(record)
        if type(name) is not str or not name:
            raise ConfigurationError(f"invalid {duplicate_label} plugin registration name")
        if name in existing or name in pending:
            raise ConfigurationError(f"duplicate {duplicate_label} registration: {name}")
        pending.add(name)
        prepared.append((name, record.spec, registration))
    return tuple(prepared)


def load_plugin_specs(*, group: str, expected_type: type[_Spec], label: str) -> tuple[_Spec, ...]:
    """Load and materialize plugin specs without exposing plugin exceptions."""
    return tuple(
        record.spec
        for record in load_plugin_spec_records(
            group=group, expected_type=expected_type, label=label
        )
    )
