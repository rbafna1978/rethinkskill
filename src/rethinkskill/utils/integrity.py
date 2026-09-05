"""RethinkSkill utils integrity."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from dataclasses import dataclass

from rethinkskill.errors import ConfigurationError
from rethinkskill.utils.serde import (
    freeze_json_mapping,
    thaw_json_mapping,
)

_BUILTIN_EXCEPTION_NAMES = {
    value: value.__name__
    for value in vars(builtins).values()
    if isinstance(value, type) and issubclass(value, BaseException)
}


def safe_exception_type(exc: BaseException) -> str:
    """Return only a stable identifier, never an exception message."""
    if not isinstance(exc, BaseException):
        raise TypeError("boundary exception must derive from BaseException")
    for exception_type in type(exc).__mro__:
        name = _BUILTIN_EXCEPTION_NAMES.get(exception_type)
        if name is not None:
            return name
    return "Exception"


def boundary_failure(label: str, exc: BaseException) -> str:
    """Build one redacted failure sentence for a component boundary."""
    if type(label) is not str or not label.strip():
        raise TypeError("boundary label must be exact non-empty text")
    return f"{label} raised an unhandled {safe_exception_type(exc)}"


@dataclass(frozen=True, slots=True)
class CallCounts:
    attempted: int
    completed: int


def exact_call_counts(attempted: object, completed: object, *, label: str) -> CallCounts:
    """Reject coercion and enforce ``0 <= completed <= attempted``."""
    if type(label) is not str or not label.strip():
        raise ValueError("call accounting label must be exact non-empty text")
    if (
        type(attempted) is not int
        or type(completed) is not int
        or attempted < 0
        or (completed < 0)
        or (completed > attempted)
    ):
        raise ValueError(f"{label} call counts must satisfy 0 <= completed <= attempted")
    return CallCounts(attempted=attempted, completed=completed)


def freeze_manifest(
    value: Mapping[str, object], *, label: str = "manifest"
) -> Mapping[str, object]:
    """Detach a JSON manifest without adding a redundant self-hash."""
    if type(label) is not str or not label.strip():
        raise ConfigurationError("manifest label must be non-empty text")
    try:
        return freeze_json_mapping(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConfigurationError(f"{label} must be finite canonical JSON") from exc


def freeze_receipt(value: Mapping[str, object]) -> Mapping[str, object]:
    """Detach a JSON receipt; external artifact references carry content hashes."""
    return freeze_manifest(value, label="receipt")


def validate_plan_manifest(value: Mapping[str, object], *, label: str) -> dict[str, object]:
    """Validate one plan manifest and detach a mutable snapshot."""
    frozen = freeze_manifest(value, label=label)
    return thaw_json_mapping(frozen)


def validate_frozen_plan_fields(
    manifest: Mapping[str, object], expected: Mapping[str, object], *, label: str
) -> None:
    """Reject the first semantic field that drifted after plan creation."""
    if not isinstance(manifest, Mapping):
        raise ConfigurationError("frozen plan manifest must be a mapping")
    if not isinstance(expected, Mapping):
        raise ConfigurationError("expected frozen plan fields must be a mapping")
    if type(label) is not str or not label.strip():
        raise ConfigurationError("frozen plan label must be non-empty text")
    for field, value in expected.items():
        if type(field) is not str or not field:
            raise ConfigurationError("expected frozen plan field names must be exact text")
        if manifest.get(field) != value:
            raise ConfigurationError(f"{label} frozen plan drifted: {field}")


def validate_zero_call_preflight(
    manifest: Mapping[str, object],
    *,
    label: str,
    expected_status: str,
    call_fields: tuple[str, ...],
) -> None:
    """Require an exact preflight state and exact integer-zero call counts."""
    if not isinstance(manifest, Mapping):
        raise ConfigurationError("preflight manifest must be a mapping")
    if type(label) is not str or not label.strip():
        raise ConfigurationError("preflight label must be non-empty text")
    if type(expected_status) is not str or not expected_status.strip():
        raise ConfigurationError("preflight status must be non-empty text")
    if (
        type(call_fields) is not tuple
        or not call_fields
        or (not all(type(field) is str and field for field in call_fields))
        or (len(call_fields) != len(set(call_fields)))
    ):
        raise ConfigurationError("preflight call fields must be unique exact names")
    invalid = manifest.get("status") != expected_status
    for field in call_fields:
        value = manifest.get(field)
        if type(value) is not int or value != 0:
            invalid = True
    if invalid:
        raise ConfigurationError(f"{label} status or call accounting is invalid")


EVOLUTION_RECEIPT_SCHEMA_VERSION = 6

NATIVE_RECEIPT_SCHEMA_VERSION = 10
