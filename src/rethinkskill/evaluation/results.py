"""RethinkSkill evaluation results."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from rethinkskill.errors import ResultValidationError
from rethinkskill.utils.serde import snapshot_regular_file, strict_json_loads

STRUCTURED_INFRA_CLASSES = frozenset(
    {
        "target_timeout",
        "task_timeout",
        "optimizer_timeout",
        "transport_error",
        "authentication_failure",
        "authorization_failure",
        "dependency_failure",
        "oom_or_signal_137",
        "harness_traceback",
        "missing_artifact",
        "unknown_agent_invalid",
        "verifier_presemantic_failure",
        "lineage_or_hash_mismatch",
        "generated_code_isolation_violation",
        "artifact_runner_error",
        "workspace_conflict",
        "api_error",
        "provider_http_error",
        "response_schema_error",
        "authorization_error",
        "native_evaluation_invalid",
        "optimizer_response_schema_error",
        "optimizer_transport_error",
        "optimizer_invalid",
        "environment_initialization_error",
        "environment_error",
    }
)

INFRA_MARKERS = (
    "subprocess.timeoutexpired",
    "task-timeout-",
    "target-timeout-",
    "codex exec failed",
    "llm call failed",
    "authentication",
    "unauthorized",
    "forbidden",
    "rate limit",
    "too many requests",
    "connection reset",
    "service unavailable",
    "gateway timeout",
    "no module named",
    "out of memory",
    "signal 137",
    "missing artifact",
    "workspace conflict",
)


def structured_infrastructure_invalid(row: dict[str, Any]) -> bool:
    """Return whether one row has explicit infrastructure-failure metadata."""
    values = {
        str(row.get("failure_class") or "").strip().lower(),
        str(row.get("infra_class") or "").strip().lower(),
    }
    if values & STRUCTURED_INFRA_CLASSES:
        return True
    return row.get("infrastructure_invalid") is True


def heuristic_infrastructure_invalid(row: dict[str, Any]) -> bool:
    """Return whether unstructured free text only suggests infrastructure failure."""
    if structured_infrastructure_invalid(row):
        return False
    text = "\n".join(
        str(row.get(key) or "")
        for key in ("error", "failure", "fail_reason", "stderr", "traceback")
    ).lower()
    return any(marker in text for marker in INFRA_MARKERS)


def infrastructure_invalid(row: dict[str, Any]) -> bool:
    """Return whether structured or heuristic failure evidence is present."""
    return structured_infrastructure_invalid(row) or heuristic_infrastructure_invalid(row)


@dataclass(frozen=True, slots=True)
class LedgerReport:
    path: str
    total_lines: int
    valid_json_rows: int
    unique_ids: int
    duplicate_ids: tuple[str, ...]
    infrastructure_invalid_ids: tuple[str, ...]
    parse_errors: tuple[str, ...]
    missing_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    structured_infrastructure_invalid_ids: tuple[str, ...] = ()
    heuristic_infrastructure_invalid_ids: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if self.parse_errors or self.duplicate_ids or self.missing_ids or self.unknown_ids:
            return "INVALID_OR_INCOMPLETE"
        if self.infrastructure_invalid_ids:
            return "INFRASTRUCTURE_INVALID_ROWS_PRESENT"
        return "STRUCTURALLY_VALID"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": self.status,
            "path": self.path,
            "total_lines": self.total_lines,
            "valid_json_rows": self.valid_json_rows,
            "unique_ids": self.unique_ids,
            "duplicate_ids": list(self.duplicate_ids),
            "infrastructure_invalid_ids": list(self.infrastructure_invalid_ids),
            "structured_infrastructure_invalid_ids": list(
                self.structured_infrastructure_invalid_ids
            ),
            "heuristic_infrastructure_invalid_ids": list(self.heuristic_infrastructure_invalid_ids),
            "parse_errors": list(self.parse_errors),
            "missing_ids": list(self.missing_ids),
            "unknown_ids": list(self.unknown_ids),
        }


def validate_ids(values: Iterable[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ResultValidationError(f"{label} must be an iterable of task-ID strings")
    try:
        materialized = tuple(values)
    except TypeError as exc:
        raise ResultValidationError(f"{label} must be an iterable of task-ID strings") from exc
    seen: set[str] = set()
    normalized: list[str] = []
    for value in materialized:
        if (
            type(value) is not str
            or not value
            or value != value.strip()
            or (len(value) > 512)
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ResultValidationError(f"{label} contains an invalid task ID: {value!r}")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ResultValidationError(f"{label} task IDs must be valid UTF-8") from exc
        if value in seen:
            raise ResultValidationError(f"{label} contains a duplicate task ID: {value!r}")
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def recovery_plan(report: LedgerReport) -> dict[str, object]:
    """List only unambiguous infrastructure-invalid IDs as recovery-eligible."""
    if type(report) is not LedgerReport:
        raise ResultValidationError(
            "recovery planning requires a LedgerReport of the exact core type"
        )
    duplicate_ids = set(report.duplicate_ids)
    eligible_ids = [
        item_id
        for item_id in report.structured_infrastructure_invalid_ids
        if item_id not in duplicate_ids
    ]
    heuristic_ids = [
        item_id
        for item_id in report.heuristic_infrastructure_invalid_ids
        if item_id not in duplicate_ids
    ]
    return {
        "schema_version": 1,
        "status": "RECOVERY_PLAN_ONLY",
        "source": report.path,
        "eligible_reason": "confirmed_infrastructure_invalid",
        "eligible_ids": eligible_ids,
        "ineligible_heuristic_ids": heuristic_ids,
        "ineligible_duplicate_ids": list(report.duplicate_ids),
        "missing_ids": list(report.missing_ids),
        "mutations_performed": 0,
        "model_calls": 0,
    }


def load_ids_snapshot(payload: bytes) -> tuple[str, ...]:
    """Parse exact task IDs from one immutable byte snapshot."""
    if type(payload) is not bytes:
        raise ResultValidationError("task-ID snapshot must be exact bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResultValidationError("task-ID snapshot must be UTF-8") from exc
    values: list[str] = []
    for line in text.splitlines():
        value = line.strip()
        if value:
            values.append(value)
    return validate_ids(values, label="task-ID file")


def load_ids(path: Path) -> tuple[str, ...]:
    if not isinstance(path, Path):
        raise ResultValidationError(f"task-ID file is absent: {path}")
    try:
        snapshot = snapshot_regular_file(path)
    except (OSError, ValueError) as exc:
        raise ResultValidationError(f"task-ID file is absent: {path}") from exc
    return load_ids_snapshot(snapshot.payload)


def validate_ledger(path: Path, *, expected_ids: Iterable[str] | None = None) -> LedgerReport:
    """Validate structure, ID coverage, uniqueness, and infrastructure state."""
    expected_values = (
        None if expected_ids is None else validate_ids(expected_ids, label="expected IDs")
    )
    expected = None if expected_values is None else set(expected_values)
    if path.is_symlink():
        return _unreadable_report(
            path, expected=expected, error="results file must not be a symlink"
        )
    if not path.is_file():
        return _unreadable_report(path, expected=expected, error="results file is absent")
    snapshot = snapshot_regular_file(path)
    return validate_ledger_snapshot(snapshot.payload, path=path, expected_ids=expected_values)


def _unreadable_report(path: Path, *, expected: set[str] | None, error: str) -> LedgerReport:
    return LedgerReport(
        path=str(path.expanduser().absolute()),
        total_lines=0,
        valid_json_rows=0,
        unique_ids=0,
        duplicate_ids=(),
        infrastructure_invalid_ids=(),
        parse_errors=(error,),
        missing_ids=tuple(sorted(expected or ())),
        unknown_ids=(),
    )


def validate_ledger_snapshot(
    payload: bytes, *, path: Path, expected_ids: Iterable[str] | None = None
) -> LedgerReport:
    """Validate one immutable ledger byte snapshot."""
    if type(payload) is not bytes:
        raise ResultValidationError("ledger snapshot must be exact bytes")
    expected_values = (
        None if expected_ids is None else validate_ids(expected_ids, label="expected IDs")
    )
    expected = None if expected_values is None else set(expected_values)
    rows: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    infrastructure_ids: set[str] = set()
    structured_infrastructure_ids: set[str] = set()
    heuristic_infrastructure_ids: set[str] = set()
    parse_errors: list[str] = []
    total_lines = 0
    valid_rows = 0
    with BytesIO(payload) as handle:
        for line_number, raw in enumerate(handle, start=1):
            total_lines += 1
            try:
                value = strict_json_loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                parse_errors.append(f"line {line_number}: {type(exc).__name__}: {exc}")
                continue
            if not isinstance(value, dict):
                parse_errors.append(f"line {line_number}: expected JSON object")
                continue
            item_id = value.get("id")
            try:
                normalized_id = validate_ids((item_id,), label=f"line {line_number} ID")[0]
            except ResultValidationError as exc:
                parse_errors.append(str(exc))
                continue
            valid_rows += 1
            if normalized_id in rows:
                duplicates.add(normalized_id)
            if structured_infrastructure_invalid(value):
                infrastructure_ids.add(normalized_id)
                structured_infrastructure_ids.add(normalized_id)
            elif heuristic_infrastructure_invalid(value):
                infrastructure_ids.add(normalized_id)
                heuristic_infrastructure_ids.add(normalized_id)
            rows[normalized_id] = value
    observed = set(rows)
    return LedgerReport(
        path=str(path.expanduser().absolute()),
        total_lines=total_lines,
        valid_json_rows=valid_rows,
        unique_ids=len(rows),
        duplicate_ids=tuple(sorted(duplicates)),
        infrastructure_invalid_ids=tuple(sorted(infrastructure_ids)),
        parse_errors=tuple(parse_errors),
        missing_ids=tuple(sorted(expected - observed)) if expected is not None else (),
        unknown_ids=tuple(sorted(observed - expected)) if expected is not None else (),
        structured_infrastructure_invalid_ids=tuple(sorted(structured_infrastructure_ids)),
        heuristic_infrastructure_invalid_ids=tuple(sorted(heuristic_infrastructure_ids)),
    )
