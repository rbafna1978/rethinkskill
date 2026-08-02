"""RethinkSkill utils integrity."""

from __future__ import annotations

import builtins
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from rethinkskill.errors import ConfigurationError
from rethinkskill.utils.serde import (
    canonical_json_bytes,
    freeze_json_mapping,
    sha256_bytes,
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


_HASH_FIELD = re.compile("^[a-z][a-z0-9_]*_sha256$")

_SHA256 = re.compile("^[0-9a-f]{64}$")


def _contract(hash_field: object, label: object) -> tuple[str, str]:
    if type(hash_field) is not str or _HASH_FIELD.fullmatch(hash_field) is None:
        raise ConfigurationError("evidence hash field must be an exact name")
    if type(label) is not str or not label.strip():
        raise ConfigurationError("evidence label must be exact non-empty text")
    return (hash_field, label)


def bind_evidence_hash(
    value: Mapping[str, object], *, hash_field: str, label: str
) -> Mapping[str, object]:
    """Return one deeply immutable canonical evidence self-hash envelope."""
    hash_field, label = _contract(hash_field, label)
    try:
        payload = thaw_json_mapping(freeze_json_mapping(value))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConfigurationError(f"{label} must be finite canonical JSON") from exc
    payload.pop(hash_field, None)
    try:
        encoded = canonical_json_bytes(payload)
        detached = json.loads(encoded)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConfigurationError(f"{label} must be finite canonical JSON") from exc
    if not isinstance(detached, dict):
        raise ConfigurationError(f"{label} must be a JSON object")
    detached[hash_field] = sha256_bytes(encoded)
    return freeze_json_mapping(detached)


def validate_evidence_hash(value: Mapping[str, object], *, hash_field: str, label: str) -> str:
    """Validate and return one canonical evidence self-hash."""
    hash_field, label = _contract(hash_field, label)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a mapping")
    try:
        payload = thaw_json_mapping(freeze_json_mapping(value))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConfigurationError(f"{label} must be finite canonical JSON") from exc
    expected = payload.pop(hash_field, None)
    if type(expected) is not str or _SHA256.fullmatch(expected) is None:
        raise ConfigurationError(f"{label} has no valid {hash_field}")
    try:
        observed = sha256_bytes(canonical_json_bytes(payload))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConfigurationError(f"{label} must be finite canonical JSON") from exc
    if observed != expected:
        raise ConfigurationError(f"{label} integrity check failed")
    return observed


def bind_manifest_hash(value: Mapping[str, object]) -> Mapping[str, object]:
    """Return a deeply immutable manifest with a canonical self-hash."""
    return bind_evidence_hash(value, hash_field="manifest_sha256", label="plan manifest")


def validate_manifest_hash(value: Mapping[str, object], *, label: str) -> str:
    """Validate and return the canonical self-hash of one manifest."""
    return validate_evidence_hash(value, hash_field="manifest_sha256", label=label)


def validate_plan_manifest(value: Mapping[str, object], *, label: str) -> dict[str, object]:
    """Validate one self-hashed plan manifest and detach a mutable snapshot."""
    try:
        frozen = freeze_json_mapping(value)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConfigurationError(f"{label} must be finite canonical JSON") from exc
    validate_manifest_hash(frozen, label=label)
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


RECEIPT_HASH_FIELD = "receipt_sha256"

EVOLUTION_RECEIPT_SCHEMA_VERSION = 5

NATIVE_RECEIPT_SCHEMA_VERSION = 9


def bind_receipt_hash(value: Mapping[str, object]) -> Mapping[str, object]:
    """Return one detached immutable receipt with a canonical self-hash."""
    return bind_evidence_hash(value, hash_field=RECEIPT_HASH_FIELD, label="receipt")


def validate_receipt_hash(value: Mapping[str, object], *, label: str = "receipt") -> str:
    """Validate and return one receipt's canonical self-hash."""
    return validate_evidence_hash(value, hash_field=RECEIPT_HASH_FIELD, label=label)
