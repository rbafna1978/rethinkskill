"""Shared plugin catalog helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from types import MappingProxyType
from typing import Generic, TypeVar, cast

from rethinkskill.errors import ConfigurationError

CatalogT = TypeVar("CatalogT", bound="SealableCatalog")


class SealableCatalog:
    """Mutable during assembly and read-only after selection begins."""

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
        self._ensure_mutable()
        if type(entries) is not tuple:
            raise ConfigurationError("catalog registration batch must be an exact tuple")
        if type(duplicate_label) is not str or not duplicate_label.strip():
            raise ConfigurationError("catalog duplicate label must be non-empty text")
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


def _optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ConfigurationError(f"invalid {label}")
    return value


@dataclass(frozen=True, slots=True)
class ExtensionRegistration:
    """Origin metadata for a benchmark, harness, or optimizer plugin."""

    name: str
    component_kind: str
    source: str
    entry_point_group: str | None = None
    entry_point_name: str | None = None
    distribution_name: str | None = None
    distribution_version: str | None = None

    @classmethod
    def from_public(cls, value: object) -> ExtensionRegistration:
        keys = {
            "name",
            "component_kind",
            "source",
            "entry_point_group",
            "entry_point_name",
            "distribution_name",
            "distribution_version",
        }
        if type(value) is not dict or set(value) != keys:
            raise ConfigurationError("extension registration has an invalid public schema")
        registration = cls(
            name=cast(str, value["name"]),
            component_kind=cast(str, value["component_kind"]),
            source=cast(str, value["source"]),
            entry_point_group=cast(str | None, value["entry_point_group"]),
            entry_point_name=cast(str | None, value["entry_point_name"]),
            distribution_name=cast(str | None, value["distribution_name"]),
            distribution_version=cast(str | None, value["distribution_version"]),
        )
        registration.validate()
        return registration

    def validate(self) -> None:
        name = _optional_text(self.name, label="extension name")
        if name is None or name != name.lower() or name != name.strip():
            raise ConfigurationError(f"extension name must be lowercase: {self.name!r}")
        if self.component_kind not in _COMPONENT_KINDS:
            raise ConfigurationError(f"invalid extension component kind: {self.component_kind!r}")
        if self.source not in _SOURCES:
            raise ConfigurationError(f"invalid extension registration source: {self.source!r}")
        for label, value in (
            ("extension entry-point group", self.entry_point_group),
            ("extension entry-point name", self.entry_point_name),
            ("extension distribution name", self.distribution_name),
            ("extension distribution version", self.distribution_version),
        ):
            _optional_text(value, label=label)
        entry_point_identity = (self.entry_point_group, self.entry_point_name)
        distribution = (self.distribution_name, self.distribution_version)
        if self.source == "entry_point":
            if any(value is None for value in entry_point_identity):
                raise ConfigurationError("entry-point extension registration is missing identity")
        elif any(value is not None for value in (*entry_point_identity, *distribution)):
            raise ConfigurationError("non-plugin extension cannot declare plugin metadata")
        if (self.distribution_name is None) != (self.distribution_version is None):
            raise ConfigurationError("extension distribution name and version must be paired")

    def public(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "component_kind": self.component_kind,
            "source": self.source,
            "entry_point_group": self.entry_point_group,
            "entry_point_name": self.entry_point_name,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
        }


_Spec = TypeVar("_Spec")
_Registration = TypeVar("_Registration")


@dataclass(frozen=True, slots=True)
class LoadedPluginSpec(Generic[_Spec]):
    """One validated entry-point load plus lightweight package metadata."""

    spec: _Spec
    entry_point_name: str
    distribution_name: str | None
    distribution_version: str | None


def _distribution_metadata(entry_point: object) -> tuple[str | None, str | None]:
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        return (None, None)
    package_metadata = getattr(distribution, "metadata", None)
    name = _optional_text(
        package_metadata.get("Name") if package_metadata is not None else None,
        label="plugin distribution name",
    )
    version = _optional_text(
        getattr(distribution, "version", None), label="plugin distribution version"
    )
    if (name is None) != (version is None):
        return (None, None)
    return (name, version)


def load_plugin_spec_records(
    *, group: str, expected_type: type[_Spec], label: str
) -> tuple[LoadedPluginSpec[_Spec], ...]:
    """Load exact plugin specs after explicit opt-in."""
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
            distribution_name, distribution_version = _distribution_metadata(entry_point)
            value = entry_point.load()
            candidate = (
                value if type(value) is expected_type else value() if callable(value) else value
            )
            if type(candidate) is expected_type:
                values = (candidate,)
            elif isinstance(candidate, Iterable) and not isinstance(
                candidate, (str, bytes, bytearray, Mapping)
            ):
                values = tuple(candidate)
            else:
                values = ()
            if not values or not all(type(spec) is expected_type for spec in values):
                raise TypeError(f"expected one or more {expected_type.__name__} values")
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
    """Validate an entire plugin batch before catalog mutation."""
    if type(records) is not tuple:
        raise ConfigurationError("plugin registration records must be an exact tuple")
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
    return tuple(
        record.spec
        for record in load_plugin_spec_records(
            group=group, expected_type=expected_type, label=label
        )
    )
