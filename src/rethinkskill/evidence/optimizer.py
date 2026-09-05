"""RethinkSkill evidence optimizer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rethinkskill.errors import ResultValidationError
from rethinkskill.evidence.common import (
    artifact_snapshot,
    invalid,
    json_object,
    mapping,
)
from rethinkskill.evolution.loop import (
    OPTIMIZER_EVIDENCE_FILENAME,
    OPTIMIZER_EVIDENCE_SCHEMA_VERSION,
    optimizer_artifact_inventory,
    skill_hash,
)
from rethinkskill.runtime.tasks import RenderedTask
from rethinkskill.runtime.types import ModelOutcome
from rethinkskill.utils.serde import (
    snapshot_regular_file,
    strict_json_loads,
)

_EVIDENCE_KEYS = {
    "schema_version",
    "status",
    "round_no",
    "parent_sha256",
    "proposal",
    "artifacts",
}

_EXECUTION_KEYS = {
    "status",
    "response",
    "process",
    "attempted_calls",
    "completed_calls",
    "failure_class",
    "failure",
}

_PROPOSAL_KEYS = {
    "valid",
    "candidate_sha256",
    "operation",
    "rationale",
    "attempted_calls",
    "completed_calls",
    "failure_class",
    "failure",
}


def _read_utf8(path: Path, *, label: str) -> str:
    try:
        return snapshot_regular_file(path).payload.decode("utf-8")
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ResultValidationError(f"{label} is not safe UTF-8") from exc


def _replay_model_optimizer(
    *,
    run_root: Path,
    round_no: int,
    parent_sha256: str,
    proposal: Mapping[str, object],
    rendered: RenderedTask,
) -> None:
    relative_root = Path("optimizer") / f"round_{round_no:04d}"
    execution = json_object(
        run_root / relative_root / "EXECUTION.json", label=f"optimizer round {round_no} execution"
    )
    if set(execution) != _EXECUTION_KEYS:
        invalid(f"optimizer round {round_no} execution schema is invalid")
    raw = _read_utf8(
        run_root / relative_root / "raw.txt", label=f"optimizer round {round_no} raw response"
    )
    for path, expected, label in (
        (run_root / relative_root / "workspace/task.md", rendered.task_markdown, "task contract"),
        (
            run_root / relative_root / "workspace/.agents/skills/rethinkskill-optimizer/SKILL.md",
            rendered.skill_markdown,
            "skill contract",
        ),
        (run_root / relative_root / "invocation.txt", rendered.invocation, "invocation contract"),
    ):
        if _read_utf8(path, label=f"optimizer round {round_no} {label}") != expected:
            invalid(f"optimizer round {round_no} {label} drifted")
    try:
        outcome = ModelOutcome(
            status=execution.get("status"),
            response=execution.get("response"),
            raw=raw,
            process=mapping(execution.get("process"), label=f"optimizer round {round_no} process"),
            attempted_calls=execution.get("attempted_calls"),
            completed_calls=execution.get("completed_calls"),
            failure_class=execution.get("failure_class"),
            failure=execution.get("failure"),
        )
        outcome.validate()
    except (TypeError, ValueError) as exc:
        raise ResultValidationError(f"optimizer round {round_no} model outcome is invalid") from exc
    if (
        proposal.get("attempted_calls") != outcome.attempted_calls
        or proposal.get("completed_calls") != outcome.completed_calls
    ):
        invalid(f"optimizer round {round_no} call accounting drifted")
    if not outcome.ok:
        expected = {
            "valid": False,
            "candidate_sha256": parent_sha256,
            "operation": "noop",
            "rationale": "",
            "attempted_calls": outcome.attempted_calls,
            "completed_calls": outcome.completed_calls,
            "failure_class": outcome.failure_class or "optimizer_invalid",
            "failure": outcome.failure,
        }
        if dict(proposal) != expected:
            invalid(f"optimizer round {round_no} failed proposal does not replay")
        return
    try:
        value = strict_json_loads(outcome.response.strip())
    except json.JSONDecodeError:
        value = None
    valid_contract = (
        type(value) is dict
        and set(value) == {"operation", "candidate_skill", "rationale"}
        and all(type(item) is str for item in value.values())
    )
    if not valid_contract:
        if (
            proposal.get("valid") is not False
            or proposal.get("candidate_sha256") != parent_sha256
            or proposal.get("operation") != "noop"
            or (proposal.get("rationale") != "")
            or (proposal.get("failure_class") != "optimizer_response_schema_error")
            or (type(proposal.get("failure")) is not str)
            or (not proposal["failure"])
        ):
            invalid(f"optimizer round {round_no} schema failure does not replay")
        return
    assert isinstance(value, dict)
    expected = {
        "valid": True,
        "candidate_sha256": skill_hash(value["candidate_skill"]),
        "operation": value["operation"],
        "rationale": value["rationale"],
        "attempted_calls": outcome.attempted_calls,
        "completed_calls": outcome.completed_calls,
        "failure_class": None,
        "failure": "",
    }
    if dict(proposal) != expected:
        invalid(f"optimizer round {round_no} proposal response does not replay")


def replay_optimizer_proposal(
    *,
    run_root: Path,
    optimizer_manifest: Mapping[str, object],
    evidence_reference: object,
    proposal: Mapping[str, object] | None,
    round_no: int,
    parent_sha256: str,
    rendered: RenderedTask | None,
    replay_model: bool = True,
) -> dict[str, object]:
    """Validate one portable optimizer inventory and replay known strategies."""
    relative_path = Path("optimizer") / f"round_{round_no:04d}" / OPTIMIZER_EVIDENCE_FILENAME
    reference, snapshot = artifact_snapshot(
        evidence_reference,
        path=run_root / relative_path,
        label=f"optimizer round {round_no} evidence",
    )
    if reference.get("path") != relative_path.as_posix():
        invalid(f"optimizer round {round_no} evidence path is not portable")
    try:
        evidence = strict_json_loads(snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultValidationError(
            f"optimizer round {round_no} evidence is not strict JSON"
        ) from exc
    if type(evidence) is not dict or set(evidence) != _EVIDENCE_KEYS:
        invalid(f"optimizer round {round_no} evidence schema is invalid")
    frozen_proposal = mapping(
        evidence.get("proposal"), label=f"optimizer round {round_no} frozen proposal"
    )
    if set(frozen_proposal) != _PROPOSAL_KEYS:
        invalid(f"optimizer round {round_no} frozen proposal schema is invalid")
    if (
        evidence.get("schema_version") != OPTIMIZER_EVIDENCE_SCHEMA_VERSION
        or evidence.get("status") != "RETHINKSKILL_OPTIMIZER_EVIDENCE"
        or evidence.get("round_no") != round_no
        or (evidence.get("parent_sha256") != parent_sha256)
        or (proposal is not None and frozen_proposal != dict(proposal))
    ):
        invalid(f"optimizer round {round_no} evidence binding is invalid")
    artifacts = evidence.get("artifacts")
    if type(artifacts) is not list or artifacts != list(
        optimizer_artifact_inventory(run_root, round_no=round_no)
    ):
        invalid(f"optimizer round {round_no} artifact inventory drifted")
    if replay_model and optimizer_manifest.get("kind") == "model-skill-optimizer":
        if type(rendered) is not RenderedTask:
            invalid(f"optimizer round {round_no} rendered contract is absent")
        _replay_model_optimizer(
            run_root=run_root,
            round_no=round_no,
            parent_sha256=parent_sha256,
            proposal=frozen_proposal,
            rendered=rendered,
        )
    return frozen_proposal
