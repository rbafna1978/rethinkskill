"""RethinkSkill evidence run."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from rethinkskill.errors import ResultValidationError
from rethinkskill.evaluation.results import validate_ids, validate_ledger_snapshot
from rethinkskill.evidence.common import (
    artifact_snapshot,
    directory,
    integer,
    invalid,
    json_object,
    json_object_snapshot,
    ledger_rows,
    manifest_receipt_pair,
    mapping,
    report_without_path,
    sha256_digest,
)
from rethinkskill.evidence.interactive import replay_interactive_evidence
from rethinkskill.providers.authorization import (
    TERMINAL_FAILURE_FILENAME,
    TERMINAL_FAILURE_SCHEMA_VERSION,
    terminal_failure_contract,
)
from rethinkskill.runtime.runner import (
    NATIVE_TASK_EVIDENCE_FILENAME,
    NATIVE_TASK_EVIDENCE_SCHEMA_VERSION,
    failure_phase,
    native_task_artifact_inventory,
)
from rethinkskill.runtime.types import ModelOutcome
from rethinkskill.utils.integrity import (
    EVOLUTION_RECEIPT_SCHEMA_VERSION,
    NATIVE_RECEIPT_SCHEMA_VERSION,
    exact_call_counts,
)
from rethinkskill.utils.release import regular_file_inventory
from rethinkskill.utils.serde import (
    canonical_json_bytes,
    snapshot_regular_file,
    strict_json_loads,
    thaw_json_mapping,
)

_WORKSPACE = re.compile("^[A-Za-z0-9][A-Za-z0-9._-]*$")


def task_workspaces(
    selection: dict[str, object], *, expected_ids: tuple[str, ...]
) -> dict[str, object]:
    """Validate the exact task-to-workspace bijection."""
    workspaces = mapping(selection.get("task_workspaces"), label="native task workspaces")
    if (
        set(workspaces) != set(expected_ids)
        or any(
            type(name) is not str or _WORKSPACE.fullmatch(name) is None
            for name in workspaces.values()
        )
        or len(set(workspaces.values())) != len(workspaces)
    ):
        invalid("native task workspace mapping is invalid")
    return workspaces


def replay_task_contract(
    selection: dict[str, object], *, expected_ids: tuple[str, ...], workspaces: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Replay the complete frozen task contract."""
    raw_contract = selection.get("task_contract")
    if type(raw_contract) is not list or len(raw_contract) != len(expected_ids):
        invalid("native task contract has an invalid row count")
    assets_by_id = mapping(selection.get("task_assets"), label="native task assets")
    rows: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(raw_contract):
        record = mapping(raw, label=f"native task contract row {index}")
        if set(record) != {"task_id", "workspace_name", "payload", "assets"}:
            invalid("native task contract row has an invalid schema")
        task_id = record.get("task_id")
        if type(task_id) is not str or task_id in rows:
            invalid("native task contract IDs are invalid")
        mapping(record.get("payload"), label=f"native task {task_id!r} payload")
        if record.get("workspace_name") != workspaces.get(task_id) or record.get(
            "assets"
        ) != assets_by_id.get(task_id):
            invalid("native task contract disagrees with selection metadata")
        rows[task_id] = record
    if tuple(rows) != expected_ids or set(assets_by_id) != set(expected_ids):
        invalid("native task contract order or coverage is invalid")
    return rows


def replay_rendered_contract(
    selection: dict[str, object], *, expected_ids: tuple[str, ...], workspaces: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Replay the complete frozen rendered task contract."""
    raw_contract = selection.get("rendered_contract")
    if type(raw_contract) is not list or len(raw_contract) != len(expected_ids):
        invalid("native rendered contract has an invalid row count")
    rows: dict[str, dict[str, object]] = {}
    expected_keys = {
        "task_id",
        "workspace_name",
        "task_markdown",
        "skill_markdown",
        "invocation",
        "attachments",
    }
    for index, raw in enumerate(raw_contract):
        record = mapping(raw, label=f"native rendered contract row {index}")
        task_id = record.get("task_id")
        if (
            set(record) != expected_keys
            or type(task_id) is not str
            or task_id in rows
            or (record.get("workspace_name") != workspaces.get(task_id))
            or any(
                type(record.get(field)) is not str or not record[field]
                for field in ("task_markdown", "skill_markdown", "invocation")
            )
            or (type(record.get("attachments")) is not list)
        ):
            invalid("native rendered contract row has an invalid schema")
        rows[task_id] = record
    if tuple(rows) != expected_ids:
        invalid("native rendered contract order or coverage is invalid")
    return rows


def validate_capability_binding(manifest: dict[str, object]) -> None:
    """Validate the joined benchmark capability binding."""
    from rethinkskill.runtime.runner import validate_public_native_capability

    benchmark = manifest.get("benchmark")
    if type(benchmark) is not str or not benchmark:
        invalid("native capability benchmark is invalid")
    try:
        capability = validate_public_native_capability(
            mapping(manifest.get("capability"), label="native capability"), benchmark=benchmark
        )
    except Exception:
        invalid("native capability binding is invalid")
    if thaw_json_mapping(capability) != manifest.get("capability"):
        invalid("native capability binding is not canonical")


@dataclass(frozen=True, slots=True)
class NativeRowsReplay:
    """Accounting and coverage derived only from persisted result rows."""

    failure_ids: tuple[str, ...]
    attempted_calls: int
    completed_calls: int
    scorer_replayed_rows: int
    task_evidence_rows: int
    interactive_step_evidence_rows: int


def replay_native_rows(
    *,
    run_root: Path,
    rows: Sequence[dict[str, object]],
    manifest: dict[str, object],
    receipt: dict[str, object],
    workspaces: dict[str, object],
    task_contract: dict[str, dict[str, object]],
    rendered_contract: dict[str, dict[str, object]],
    evaluation_authority: object,
) -> NativeRowsReplay:
    """Replay native task directories and their redundant evidence bindings."""
    tasks_root = directory(run_root / "tasks", label="native tasks directory")
    rows_by_id = {str(row.get("id")): row for row in rows}
    task_evidence = mapping(receipt.get("task_evidence"), label="native task evidence references")
    results_reference = mapping(receipt.get("results"), label="native results reference")
    manifest_reference = mapping(receipt.get("run_manifest"), label="native run manifest reference")
    if (
        results_reference.get("path") != "results.jsonl"
        or manifest_reference.get("path") != ".rethinkskill/RUN_MANIFEST.json"
        or set(task_evidence) != set(rows_by_id)
    ):
        invalid("native artifact references are invalid")
    try:
        task_entries = tuple(tasks_root.iterdir())
    except OSError:
        invalid("native tasks directory cannot be enumerated")
    expected_task_directories = {str(workspaces[task_id]) for task_id in rows_by_id}
    if {entry.name for entry in task_entries} != expected_task_directories:
        invalid("native task directory coverage is invalid")
    row_attempted = sum(
        integer(row.get("attempted_calls"), label="row attempted calls") for row in rows
    )
    row_completed = sum(
        integer(row.get("completed_calls"), label="row completed calls") for row in rows
    )
    scorer_replayed_rows = 0
    task_evidence_rows = 0
    interactive_step_evidence_rows = 0
    for task_id, row in rows_by_id.items():
        workspace_name = workspaces.get(task_id)
        if type(workspace_name) is not str or not workspace_name:
            invalid(f"native task {task_id!r} has no frozen workspace name")
        task_root = directory(
            tasks_root / workspace_name, label=f"native task {task_id!r} directory"
        )
        task_result = json_object(
            task_root / "RESULT.json", label=f"native task {task_id!r} result"
        )
        if task_result != row:
            invalid(f"native task {task_id!r} result differs from the ledger")
        if row.get("case_id") != task_id:
            invalid(f"native task {task_id!r} case_id binding is invalid")
        if row.get("failure_class") and (row.get("hard") != 0 or row.get("soft") != 0.0):
            invalid(f"native task {task_id!r} failure score is invalid")
        if evaluation_authority == "case_scorer" and (not row.get("failure_class")):
            replay_case_score(
                benchmark=str(manifest["benchmark"]),
                row=row,
                payload=mapping(
                    task_contract[task_id].get("payload"), label=f"native task {task_id!r} payload"
                ),
                task_root=task_root,
            )
            scorer_replayed_rows += 1
        interactive_step_evidence_rows += replay_native_task_evidence(
            run_root=run_root,
            task_id=task_id,
            workspace_name=workspace_name,
            evidence_reference=task_evidence[task_id],
            task_record=task_contract[task_id],
            rendered_record=rendered_contract[task_id],
            row=row,
            replay_interactive=True,
        )
        task_evidence_rows += 1
    failure_ids = tuple(
        sorted(
            str(row.get("id"))
            for row in rows
            if row.get("failure_class") or row.get("agent_ok") is not True
        )
    )
    return NativeRowsReplay(
        failure_ids=failure_ids,
        attempted_calls=row_attempted,
        completed_calls=row_completed,
        scorer_replayed_rows=scorer_replayed_rows,
        task_evidence_rows=task_evidence_rows,
        interactive_step_evidence_rows=interactive_step_evidence_rows,
    )


def replay_case_score(
    *, benchmark: str, row: dict[str, object], payload: dict[str, object], task_root: Path
) -> None:
    """Recompute one case-scorer-owned verdict from frozen task inputs."""
    from rethinkskill.evaluation.core import evaluate_case
    from rethinkskill.runtime.runner import verification_result_fields

    response = row.get("response")
    if type(response) is not str:
        invalid("native case-scorer row response must be text")
    scoring_row = {**payload, "case_id": row.get("case_id"), "frozen_response": response}
    try:
        expected = verification_result_fields(evaluate_case(benchmark, scoring_row, task_root))
    except Exception:
        invalid("native deterministic scorer replay failed")
    if any((row.get(key) != value for key, value in expected.items())):
        invalid("native deterministic score does not replay")


_ASSET_KEYS = {"source", "target", "media_type", "role", "sha256", "visibility"}


def read_task_utf8(path: Path, *, label: str) -> str:
    """Read one safe regular task artifact as UTF-8."""
    try:
        return snapshot_regular_file(path).payload.decode("utf-8")
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ResultValidationError(f"{label} is not safe UTF-8") from exc


def _asset_target(record: Mapping[str, object], *, label: str) -> PurePosixPath:
    if set(record) != _ASSET_KEYS:
        invalid(f"{label} has an invalid schema")
    target = record.get("target")
    if type(target) is not str:
        invalid(f"{label} target is invalid")
    path = PurePosixPath(target)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or ("\\" in target)
    ):
        invalid(f"{label} target is not portable")
    if record.get("visibility") not in {"workspace", "verifier"}:
        invalid(f"{label} visibility is invalid")
    for field in ("source", "media_type", "role"):
        if type(record.get(field)) is not str or not record[field]:
            invalid(f"{label} {field} is invalid")
    sha256_digest(record.get("sha256"), label=f"{label} SHA-256")
    return path


def validate_materialized_task_contract(
    *,
    task_root: Path,
    task_record: Mapping[str, object],
    rendered_record: Mapping[str, object],
    task_id: str,
) -> None:
    """Replay materialized prompt, skill, invocation, and asset bindings."""
    for relative, field, label in (
        ("workspace/task.md", "task_markdown", "task contract"),
        (
            "workspace/.agents/skills/rethinkskill-target/SKILL.md",
            "skill_markdown",
            "skill contract",
        ),
        ("invocation.txt", "invocation", "invocation contract"),
    ):
        expected = rendered_record.get(field)
        if type(expected) is not str or not expected:
            invalid(f"native task {task_id!r} {label} is invalid")
        if (
            read_task_utf8(task_root / relative, label=f"native task {task_id!r} {label}")
            != expected
        ):
            invalid(f"native task {task_id!r} {label} drifted")
    assets = task_record.get("assets")
    attachments = rendered_record.get("attachments")
    if type(assets) is not list or type(attachments) is not list:
        invalid(f"native task {task_id!r} asset contract is invalid")
    normalized_assets: list[dict[str, object]] = []
    for index, raw in enumerate(assets):
        asset = mapping(raw, label=f"native task {task_id!r} asset {index}")
        target = _asset_target(asset, label=f"native task {task_id!r} asset {index}")
        root = task_root / str(asset["visibility"])
        try:
            snapshot = snapshot_regular_file(root.joinpath(*target.parts))
        except (OSError, ValueError) as exc:
            raise ResultValidationError(
                f"native task {task_id!r} materialized asset is unsafe"
            ) from exc
        if snapshot.sha256 != asset.get("sha256"):
            invalid(f"native task {task_id!r} materialized asset drifted")
        normalized_assets.append(dict(asset))
    for index, raw in enumerate(attachments):
        attachment = mapping(raw, label=f"native task {task_id!r} attachment {index}")
        _asset_target(attachment, label=f"native task {task_id!r} attachment {index}")
        if attachment.get("visibility") != "workspace" or dict(attachment) not in normalized_assets:
            invalid(f"native task {task_id!r} attachment binding is invalid")
    if len({canonical_json_bytes(item) for item in normalized_assets}) != len(normalized_assets):
        invalid(f"native task {task_id!r} assets contain duplicates")


def validate_run_evidence(run_root: Path) -> dict[str, object]:
    """Replay one current run's immutable evidence without model calls."""
    root = run_root.expanduser().absolute()
    directory(root, label="run root")
    control = directory(root / ".rethinkskill", label="run control directory")
    normal_kinds = [
        name
        for name, marker in (
            ("native", control / "PROCESS_RECEIPT.json"),
            ("evolution", control / "EVOLUTION_RECEIPT.json"),
        )
        if marker.exists() or marker.is_symlink()
    ]
    failure_path = control / TERMINAL_FAILURE_FILENAME
    failure_present = failure_path.exists() or failure_path.is_symlink()
    if failure_present and normal_kinds:
        invalid("terminal failure evidence cannot coexist with a normal run")
    if failure_present:
        result = validate_terminal_failure_evidence(root)
    elif len(normal_kinds) != 1:
        invalid("run evidence topology must contain exactly one current run kind")
    else:
        run_kind = normal_kinds[0]
        if run_kind == "native":
            result = validate_native_evidence(root)
        elif run_kind == "evolution":
            from rethinkskill.evidence.evolution import validate_evolution_evidence

            result = validate_evolution_evidence(root)
    return {
        "schema_version": 1,
        "status": "RETHINKSKILL_RUN_EVIDENCE_VALIDATED",
        "valid": True,
        "run_root": str(root),
        **result,
        "scientific_completion_claimed": False,
        "model_calls": 0,
    }


_EXCEPTION_TYPE = re.compile("^[A-Za-z][A-Za-z0-9_]{0,127}$")

_MANIFEST_CONTRACTS = {
    "native": (
        "RETHINKSKILL_NATIVE_EXECUTION_AUTHORIZED",
        NATIVE_RECEIPT_SCHEMA_VERSION,
        "process_receipt_schema_version",
    ),
    "evolution": (
        "RETHINKSKILL_EVOLUTION_AUTHORIZED",
        EVOLUTION_RECEIPT_SCHEMA_VERSION,
        "evolution_receipt_schema_version",
    ),
}


def _validate_calls(value: object, *, components: tuple[str, ...]) -> dict[str, dict[str, object]]:
    calls = mapping(value, label="terminal failure calls")
    if set(calls) != set(components):
        invalid("terminal failure call components are invalid")
    observed: dict[str, dict[str, object]] = {}
    for name in components:
        record = mapping(calls[name], label=f"{name} call accounting")
        if set(record) != {"attempted", "completed", "accounting_known"}:
            invalid(f"{name} call accounting schema is invalid")
        try:
            counts = exact_call_counts(
                record.get("attempted"), record.get("completed"), label=f"terminal {name} calls"
            )
        except (TypeError, ValueError):
            invalid(f"{name} call accounting is invalid")
        known = record.get("accounting_known")
        if type(known) is not bool:
            invalid(f"{name} accounting-known is invalid")
        observed[name] = {
            "attempted": counts.attempted,
            "completed": counts.completed,
            "accounting_known": known,
        }
    return observed


def validate_terminal_failure_evidence(run_root: Path) -> dict[str, object]:
    """Validate one partial run and its redacted terminal failure record."""
    root = run_root.expanduser().absolute()
    failure_path = root / ".rethinkskill" / TERMINAL_FAILURE_FILENAME
    evidence, failure_snapshot = json_object_snapshot(
        failure_path, label="terminal failure evidence"
    )
    expected_keys = {
        "schema_version",
        "status",
        "created_at",
        "run_kind",
        "phase",
        "exception_type",
        "calls",
        "run_manifest",
        "artifacts",
        "scientific_completion_claimed",
    }
    if (
        set(evidence) != expected_keys
        or evidence.get("schema_version") != TERMINAL_FAILURE_SCHEMA_VERSION
        or evidence.get("status") != "RETHINKSKILL_FRAMEWORK_FAILURE"
        or (evidence.get("scientific_completion_claimed") is not False)
    ):
        invalid("terminal failure evidence schema is invalid")
    run_kind = evidence.get("run_kind")
    if type(run_kind) is not str:
        invalid("terminal failure run kind is invalid")
    contract = terminal_failure_contract(run_kind)
    phase = evidence.get("phase")
    if type(phase) is not str or phase not in contract["phases"]:
        invalid("terminal failure phase is invalid")
    exception_type = evidence.get("exception_type")
    if type(exception_type) is not str or _EXCEPTION_TYPE.fullmatch(exception_type) is None:
        invalid("terminal failure exception type is invalid")
    receipt_path = root / str(contract["receipt"])
    if receipt_path.exists() or receipt_path.is_symlink():
        invalid("terminal failure evidence coexists with a normal receipt")
    manifest_relative = str(contract["manifest"])
    manifest_reference, manifest_snapshot = artifact_snapshot(
        evidence.get("run_manifest"),
        path=root / manifest_relative,
        label=f"{run_kind} run manifest",
    )
    if manifest_reference.get("path") != manifest_relative:
        invalid("terminal failure run-manifest path is invalid")
    try:
        manifest = strict_json_loads(manifest_snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        invalid("terminal failure run manifest is invalid")
    if type(manifest) is not dict:
        invalid("terminal failure run manifest is not an object")
    status, receipt_schema, declaration = _MANIFEST_CONTRACTS[run_kind]
    if manifest.get("status") != status or manifest.get(declaration) != receipt_schema:
        invalid("terminal failure run manifest contract is invalid")
    components = contract["components"]
    if type(components) is not tuple:
        invalid("terminal failure component contract is invalid")
    calls = _validate_calls(evidence.get("calls"), components=components)
    artifacts = evidence.get("artifacts")
    if type(artifacts) is not list:
        invalid("terminal failure artifact inventory is invalid")
    try:
        observed_artifacts = list(
            regular_file_inventory(
                root,
                relative_to=root,
                excluded=(failure_path,),
                label=f"{run_kind} partial run artifacts",
            )
        )
    except (OSError, ResultValidationError, ValueError):
        invalid("terminal failure artifact inventory cannot be replayed")
    if artifacts != observed_artifacts:
        invalid("terminal failure artifact inventory does not replay")
    return {
        "run_kind": run_kind,
        "terminal_status": "RETHINKSKILL_FRAMEWORK_FAILURE",
        "failure_phase": phase,
        "exception_type": exception_type,
        "calls": calls,
        "artifacts": {
            "failure_sha256": failure_snapshot.sha256,
            "manifest_sha256": manifest_snapshot.sha256,
            "partial_artifact_rows": len(artifacts),
        },
        "model_calls": 0,
    }


def validate_native_evidence(run_root: Path) -> dict[str, object]:
    control = run_root / ".rethinkskill"
    manifest_path = control / "RUN_MANIFEST.json"
    receipt_path = control / "PROCESS_RECEIPT.json"
    manifest, receipt, manifest_snapshot, receipt_snapshot = manifest_receipt_pair(
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        manifest_label="native run manifest",
        receipt_label="native process receipt",
        declaration_field="process_receipt_schema_version",
        schema_version=NATIVE_RECEIPT_SCHEMA_VERSION,
    )
    if manifest.get("status") != "RETHINKSKILL_NATIVE_EXECUTION_AUTHORIZED":
        invalid("native run manifest has an invalid authorization status")
    if receipt.get("status") not in {
        "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED",
        "RETHINKSKILL_NATIVE_EXECUTION_INVALID",
    }:
        invalid("native process receipt has an invalid terminal status")
    selection = mapping(manifest.get("selection"), label="native selection")
    expected_ids = validate_ids(selection.get("task_ids", ()), label="native manifest task IDs")
    selection_count = integer(selection.get("count"), label="native selection count")
    if not expected_ids or selection_count != len(expected_ids):
        invalid("native selection count does not match non-empty task IDs")
    results_path = run_root / "results.jsonl"
    _, results_snapshot = artifact_snapshot(
        receipt.get("results"), path=results_path, label="native results ledger"
    )
    rows = ledger_rows(results_snapshot, label="native results ledger")
    report = validate_ledger_snapshot(
        results_snapshot.payload, path=results_path, expected_ids=expected_ids
    ).to_dict()
    if report_without_path(
        receipt.get("ledger_validation"), label="stored native ledger validation"
    ) != report_without_path(report, label="replayed native ledger validation"):
        invalid("native ledger validation does not replay exactly")
    attempted = integer(receipt.get("attempted_calls"), label="attempted calls")
    completed = integer(receipt.get("completed_calls"), label="completed calls")
    if completed > attempted:
        invalid("native completed calls exceed attempted calls")
    if manifest.get("schema_version") != NATIVE_RECEIPT_SCHEMA_VERSION:
        invalid("native run manifest schema does not match its receipt")
    validate_capability_binding(manifest)
    call_accounting_known = receipt.get("call_accounting_known")
    if type(call_accounting_known) is not bool:
        invalid("native call_accounting_known must be boolean")
    workspaces = task_workspaces(selection, expected_ids=expected_ids)
    task_contract = replay_task_contract(
        selection, expected_ids=expected_ids, workspaces=workspaces
    )
    rendered_contract = replay_rendered_contract(
        selection, expected_ids=expected_ids, workspaces=workspaces
    )
    capability = mapping(manifest.get("capability"), label="native capability")
    native_execution = mapping(
        capability.get("native_execution"), label="native execution capability"
    )
    evaluation_authority = native_execution.get("evaluation_authority")
    if evaluation_authority not in {"case_scorer", "harness"}:
        invalid("native evaluation authority is invalid")
    if evaluation_authority == "case_scorer":
        scoring_registration = mapping(
            capability.get("scoring_registration"), label="native scoring registration"
        )
        if scoring_registration.get("source") != "builtin":
            invalid("native scorer replay requires a frozen built-in scorer")
    row_replay = replay_native_rows(
        run_root=run_root,
        rows=rows,
        manifest=manifest,
        receipt=receipt,
        workspaces=workspaces,
        task_contract=task_contract,
        rendered_contract=rendered_contract,
        evaluation_authority=evaluation_authority,
    )
    if row_replay.attempted_calls != attempted or row_replay.completed_calls != completed:
        invalid("native receipt call counts do not match result rows")
    failure_ids = list(row_replay.failure_ids)
    execution_failure_ids = receipt.get("execution_failure_ids")
    selected_tasks = integer(receipt.get("selected_tasks"), label="native selected tasks")
    executed_tasks = integer(receipt.get("executed_tasks"), label="native executed tasks")
    if (
        type(execution_failure_ids) is not list
        or execution_failure_ids != failure_ids
        or receipt.get("benchmark") != manifest.get("benchmark")
        or (selected_tasks != len(expected_ids))
        or (executed_tasks != len(rows))
        or (receipt.get("executor") != manifest.get("executor"))
    ):
        invalid("native receipt summary does not match frozen artifacts")
    successful = (
        report["status"] == "STRUCTURALLY_VALID"
        and call_accounting_known
        and (attempted >= len(expected_ids))
        and (completed == attempted)
        and (not failure_ids)
    )
    expected_status = (
        "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED"
        if successful
        else "RETHINKSKILL_NATIVE_EXECUTION_INVALID"
    )
    if receipt.get("status") != expected_status:
        invalid("native terminal status does not replay from frozen evidence")
    return {
        "run_kind": "native",
        "terminal_status": expected_status,
        "artifacts": {
            "manifest_sha256": manifest_snapshot.sha256,
            "receipt_sha256": receipt_snapshot.sha256,
            "results_sha256": results_snapshot.sha256,
            "result_rows": len(rows),
            "scorer_replayed_rows": row_replay.scorer_replayed_rows,
            "task_evidence_rows": row_replay.task_evidence_rows,
            "interactive_step_evidence_rows": row_replay.interactive_step_evidence_rows,
        },
    }


_EVIDENCE_KEYS = {
    "schema_version",
    "status",
    "task_id",
    "workspace_name",
    "result",
    "artifacts",
}


def replay_native_task_evidence(
    *,
    run_root: Path,
    task_id: str,
    workspace_name: str,
    evidence_reference: object,
    task_record: Mapping[str, object],
    rendered_record: Mapping[str, object],
    row: Mapping[str, object],
    replay_interactive: bool,
) -> int:
    """Replay one task envelope, artifact inventory, and result transition."""
    relative_path = Path("tasks") / workspace_name / NATIVE_TASK_EVIDENCE_FILENAME
    reference, snapshot = artifact_snapshot(
        evidence_reference, path=run_root / relative_path, label=f"native task {task_id!r} evidence"
    )
    if reference.get("path") != relative_path.as_posix():
        invalid(f"native task {task_id!r} evidence path is not portable")
    try:
        evidence = strict_json_loads(snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultValidationError(f"native task {task_id!r} evidence is not strict JSON") from exc
    if type(evidence) is not dict or set(evidence) != _EVIDENCE_KEYS:
        invalid(f"native task {task_id!r} evidence schema is invalid")
    if (
        evidence.get("schema_version") != NATIVE_TASK_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "RETHINKSKILL_NATIVE_TASK_EVIDENCE"
        or evidence.get("task_id") != task_id
        or (evidence.get("workspace_name") != workspace_name)
        or (
            mapping(evidence.get("result"), label=f"native task {task_id!r} frozen result")
            != dict(row)
        )
    ):
        invalid(f"native task {task_id!r} evidence binding is invalid")
    artifacts = evidence.get("artifacts")
    if type(artifacts) is not list or artifacts != list(
        native_task_artifact_inventory(run_root, workspace_name=workspace_name)
    ):
        invalid(f"native task {task_id!r} artifact inventory drifted")
    task_root = run_root / "tasks" / workspace_name
    validate_materialized_task_contract(
        task_root=task_root,
        task_record=task_record,
        rendered_record=rendered_record,
        task_id=task_id,
    )
    return replay_native_task_outcome(
        task_root=task_root, task_id=task_id, row=row, replay_interactive=replay_interactive
    )


_EXECUTION_KEYS = {
    "status",
    "response",
    "process",
    "attempted_calls",
    "completed_calls",
    "failure_class",
    "failure",
}


def replay_native_task_outcome(
    *, task_root: Path, task_id: str, row: Mapping[str, object], replay_interactive: bool
) -> int:
    """Replay the frozen ModelOutcome and its exact ledger transition."""
    execution = json_object(
        task_root / "EXECUTION.json", label=f"native task {task_id!r} execution"
    )
    if set(execution) != _EXECUTION_KEYS:
        invalid(f"native task {task_id!r} execution schema is invalid")
    raw = read_task_utf8(task_root / "raw.txt", label=f"native task {task_id!r} raw response")
    try:
        outcome = ModelOutcome(
            status=execution.get("status"),
            response=execution.get("response"),
            raw=raw,
            process=mapping(execution.get("process"), label=f"native task {task_id!r} process"),
            attempted_calls=execution.get("attempted_calls"),
            completed_calls=execution.get("completed_calls"),
            failure_class=execution.get("failure_class"),
            failure=execution.get("failure"),
        )
        outcome.validate()
    except (TypeError, ValueError) as exc:
        raise ResultValidationError(f"native task {task_id!r} model outcome is invalid") from exc
    interactive_steps = (
        replay_interactive_evidence(task_root=task_root, outcome=outcome)
        if replay_interactive
        else 0
    )
    if (
        row.get("response") != outcome.response
        or row.get("attempted_calls") != outcome.attempted_calls
        or row.get("completed_calls") != outcome.completed_calls
    ):
        invalid(f"native task {task_id!r} execution binding drifted")
    row_process = mapping(row.get("execution"), label=f"native task {task_id!r} row process")
    failure_class = row.get("failure_class")
    if failure_class == "verifier_presemantic_failure":
        boundary = mapping(
            row_process.get("verifier_boundary"), label=f"native task {task_id!r} verifier boundary"
        )
        expected_process = {**thaw_json_mapping(outcome.process), "verifier_boundary": boundary}
        if (
            set(boundary) != {"exception_type"}
            or type(boundary.get("exception_type")) is not str
            or (not outcome.ok)
            or (row_process != expected_process)
            or (row.get("agent_ok") is not False)
            or (row.get("hard") != 0)
            or (row.get("soft") != 0.0)
            or (row.get("phase") != "verification")
            or (
                row.get("fail_reason")
                != f"verifier boundary raised an unhandled {boundary.get('exception_type')}"
            )
        ):
            invalid(f"native task {task_id!r} verifier failure does not replay")
        return interactive_steps
    if row_process != thaw_json_mapping(outcome.process):
        invalid(f"native task {task_id!r} process evidence drifted")
    if outcome.failure_class:
        if (
            row.get("agent_ok") is not False
            or failure_class != outcome.failure_class
            or row.get("fail_reason") != outcome.failure
            or (row.get("phase") != failure_phase(outcome.failure_class))
            or (row.get("hard") != 0)
            or (row.get("soft") != 0.0)
        ):
            invalid(f"native task {task_id!r} failure outcome does not replay")
    elif row.get("agent_ok") is not True or "failure_class" in row:
        invalid(f"native task {task_id!r} successful outcome does not replay")
    return interactive_steps
