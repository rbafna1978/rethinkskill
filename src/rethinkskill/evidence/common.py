"""Strict filesystem and JSON primitives for evidence replay."""

from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path

from rethinkskill.errors import ResultValidationError
from rethinkskill.utils.serde import RegularFileSnapshot, snapshot_regular_file, strict_json_loads

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def invalid(message: str) -> None:
    raise ResultValidationError(message)


def regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        invalid(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        invalid(f"{label} is absent: {path}")
    return path


def directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        invalid(f"{label} is absent or unsafe: {path}")
    return path


def json_object(path: Path, *, label: str) -> dict[str, object]:
    value, _ = json_object_snapshot(path, label=label)
    return value


def json_object_snapshot(
    path: Path,
    *,
    label: str,
) -> tuple[dict[str, object], RegularFileSnapshot]:
    """Parse one strict JSON object from the exact bytes returned."""

    regular_file(path, label=label)
    try:
        snapshot = snapshot_regular_file(path)
        value = strict_json_loads(snapshot.payload)
    except (
        OSError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ResultValidationError(f"{label} is not strict JSON: {type(exc).__name__}") from exc
    if type(value) is not dict:
        invalid(f"{label} must contain one JSON object")
    return value, snapshot


def sha256_digest(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        invalid(f"{label} must be a lowercase SHA-256 digest")
    return value


def integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        invalid(f"{label} must be an integer >= {minimum}")
    return value


def mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        invalid(f"{label} must be a JSON object")
    return value


def artifact_snapshot(
    value: object,
    *,
    path: Path,
    label: str,
    keys: frozenset[str] = frozenset({"path", "sha256"}),
) -> tuple[dict[str, object], RegularFileSnapshot]:
    """Validate one artifact reference against the exact bytes returned."""

    regular_file(path, label=label)
    try:
        snapshot = snapshot_regular_file(path)
    except (OSError, ValueError) as exc:
        raise ResultValidationError(
            f"{label} could not be snapshotted: {type(exc).__name__}"
        ) from exc
    reference = _validate_artifact_snapshot(
        value,
        snapshot=snapshot,
        label=label,
        keys=keys,
    )
    return reference, snapshot


def _validate_artifact_snapshot(
    value: object,
    *,
    snapshot: RegularFileSnapshot,
    label: str,
    keys: frozenset[str] = frozenset({"path", "sha256"}),
) -> dict[str, object]:
    """Validate a reference against an already captured exact snapshot."""

    if type(snapshot) is not RegularFileSnapshot:
        invalid(f"{label} requires an exact regular-file snapshot")
    reference = mapping(value, label=f"{label} reference")
    if set(reference) != keys or type(reference.get("path")) is not str:
        invalid(f"{label} reference has an invalid schema")
    expected = sha256_digest(
        reference.get("sha256"),
        label=f"{label} SHA-256",
    )
    if snapshot.sha256 != expected:
        invalid(f"{label} SHA-256 does not match its receipt")
    return reference


def manifest_receipt_pair(
    *,
    manifest_path: Path,
    receipt_path: Path,
    manifest_label: str,
    receipt_label: str,
    declaration_field: str,
    schema_version: int,
) -> tuple[
    dict[str, object],
    dict[str, object],
    RegularFileSnapshot,
    RegularFileSnapshot,
]:
    manifest, manifest_snapshot = json_object_snapshot(
        manifest_path,
        label=manifest_label,
    )
    receipt, receipt_snapshot = json_object_snapshot(
        receipt_path,
        label=receipt_label,
    )
    if type(schema_version) is not int:
        invalid("receipt schema support declaration is invalid")
    declared = manifest.get(declaration_field)
    if declared != schema_version:
        invalid(f"{manifest_label} does not declare {receipt_label} schema {schema_version}")
    if receipt.get("schema_version") != declared:
        invalid(f"{receipt_label} schema does not match its manifest")
    _validate_artifact_snapshot(
        receipt.get("run_manifest"),
        snapshot=manifest_snapshot,
        label=manifest_label,
    )
    return manifest, receipt, manifest_snapshot, receipt_snapshot


def ledger_rows(
    snapshot: RegularFileSnapshot,
    *,
    label: str,
) -> tuple[dict[str, object], ...]:
    if type(snapshot) is not RegularFileSnapshot:
        invalid(f"{label} requires an exact regular-file snapshot")
    rows: list[dict[str, object]] = []
    with BytesIO(snapshot.payload) as handle:
        for line_number, raw in enumerate(handle, start=1):
            try:
                value = strict_json_loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ResultValidationError(
                    f"{label} line {line_number} is invalid: {type(exc).__name__}"
                ) from exc
            if type(value) is not dict:
                invalid(f"{label} line {line_number} must be a JSON object")
            rows.append(value)
    return tuple(rows)


def report_without_path(value: object, *, label: str) -> dict[str, object]:
    report = dict(mapping(value, label=label))
    path = report.pop("path", None)
    if type(path) is not str or not path:
        invalid(f"{label} must record its original path")
    return report


__all__ = [
    "artifact_snapshot",
    "directory",
    "integer",
    "invalid",
    "json_object",
    "json_object_snapshot",
    "ledger_rows",
    "manifest_receipt_pair",
    "mapping",
    "regular_file",
    "report_without_path",
    "sha256_digest",
]
