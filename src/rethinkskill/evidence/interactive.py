"""RethinkSkill evidence interactive."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rethinkskill.evidence.common import artifact_snapshot, invalid, mapping
from rethinkskill.runtime.interactive import INTERACTIVE_STEP_EVIDENCE_SCHEMA_VERSION
from rethinkskill.runtime.types import ModelOutcome
from rethinkskill.utils.serde import strict_json_loads, thaw_json_mapping

_STEP_KEYS = {"schema_version", "step", "rendered", "invocation", "raw", "execution"}

_RENDERED_KEYS = {"task_markdown", "skill_markdown", "invocation", "attachments"}

_EXECUTION_KEYS = {
    "status",
    "response",
    "process",
    "attempted_calls",
    "completed_calls",
    "failure_class",
    "failure",
}

_RUNNER_CONTROL_KEYS = {"kind", "transcript", "evidence_schema_version", "step_evidence"}


def _text_artifact(workspace: Path, reference: object, *, expected_path: str, label: str) -> str:
    value, snapshot = artifact_snapshot(
        reference, path=workspace.joinpath(*Path(expected_path).parts), label=label
    )
    if value.get("path") != expected_path:
        invalid(f"{label} path is not portable")
    try:
        return snapshot.payload.decode("utf-8")
    except UnicodeDecodeError:
        invalid(f"{label} is not UTF-8")


def _step_outcome(workspace: Path, record: Mapping[str, object], *, step: int) -> ModelOutcome:
    prefix = f"steps/{step:03d}"
    raw = _text_artifact(
        workspace,
        record.get("raw"),
        expected_path=f"{prefix}/raw.txt",
        label=f"interactive step {step} raw response",
    )
    execution_reference, execution_snapshot = artifact_snapshot(
        record.get("execution"),
        path=workspace / "steps" / f"{step:03d}" / "EXECUTION.json",
        label=f"interactive step {step} execution",
    )
    if execution_reference.get("path") != f"{prefix}/EXECUTION.json":
        invalid(f"interactive step {step} execution path is not portable")
    try:
        execution = strict_json_loads(execution_snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        invalid(f"interactive step {step} execution is not strict JSON")
    if type(execution) is not dict or set(execution) != _EXECUTION_KEYS:
        invalid(f"interactive step {step} execution schema is invalid")
    try:
        outcome = ModelOutcome(
            status=execution.get("status"),
            response=execution.get("response"),
            raw=raw,
            process=mapping(execution.get("process"), label=f"interactive step {step} process"),
            attempted_calls=execution.get("attempted_calls"),
            completed_calls=execution.get("completed_calls"),
            failure_class=execution.get("failure_class"),
            failure=execution.get("failure"),
        )
        outcome.validate()
    except (TypeError, ValueError) as exc:
        invalid(f"interactive step {step} model outcome is invalid: {type(exc).__name__}")
    return outcome


def _rendered_step(workspace: Path, record: Mapping[str, object], *, step: int) -> None:
    rendered = mapping(record.get("rendered"), label=f"interactive step {step} rendered contract")
    if set(rendered) != _RENDERED_KEYS or rendered.get("attachments") != []:
        invalid(f"interactive step {step} rendered contract is invalid")
    prefix = f"steps/{step:03d}"
    invocation = _text_artifact(
        workspace,
        record.get("invocation"),
        expected_path=f"{prefix}/invocation.txt",
        label=f"interactive step {step} invocation",
    )
    expected_files = (
        (workspace / "steps" / f"{step:03d}" / "workspace" / "task.md", "task_markdown"),
        (
            workspace
            / "steps"
            / f"{step:03d}"
            / "workspace"
            / ".agents"
            / "skills"
            / "rethinkskill-target"
            / "SKILL.md",
            "skill_markdown",
        ),
    )
    for path, field in expected_files:
        try:
            observed = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            invalid(f"interactive step {step} rendered file is unsafe: {type(exc).__name__}")
        if type(rendered.get(field)) is not str or observed != rendered[field]:
            invalid(f"interactive step {step} rendered contract drifted")
    if type(rendered.get("invocation")) is not str or invocation != rendered["invocation"]:
        invalid(f"interactive step {step} invocation contract drifted")


def replay_interactive_evidence(*, task_root: Path, outcome: ModelOutcome) -> int:
    """Replay every interactive step and return the validated step count."""
    process = thaw_json_mapping(outcome.process)
    if "interactive_runner" not in process:
        return 0
    runner = mapping(process.get("interactive_runner"), label="interactive runner evidence")
    transcript = runner.get("transcript")
    records = runner.get("step_evidence")
    if (
        runner.get("evidence_schema_version") != INTERACTIVE_STEP_EVIDENCE_SCHEMA_VERSION
        or type(runner.get("kind")) is not str
        or (not runner["kind"])
        or (type(transcript) is not list)
        or (type(records) is not list)
    ):
        invalid("interactive runner evidence schema is invalid")
    expected_raw = json.dumps(transcript, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if outcome.raw != expected_raw:
        invalid("interactive transcript raw evidence drifted")
    workspace = task_root / "workspace"
    step_outcomes: list[ModelOutcome] = []
    for step, raw_record in enumerate(records):
        record = mapping(raw_record, label=f"interactive step {step} evidence")
        if (
            set(record) != _STEP_KEYS
            or record.get("schema_version") != INTERACTIVE_STEP_EVIDENCE_SCHEMA_VERSION
            or record.get("step") != step
        ):
            invalid(f"interactive step {step} evidence schema is invalid")
        _rendered_step(workspace, record, step=step)
        step_outcomes.append(_step_outcome(workspace, record, step=step))
    if len(transcript) > len(step_outcomes):
        invalid("interactive transcript exceeds step evidence")
    for step, raw_row in enumerate(transcript):
        row = mapping(raw_row, label=f"interactive transcript row {step}")
        if row.get("step") != step or row.get("model_response") != step_outcomes[step].response:
            invalid("interactive transcript does not bind step responses")
    attempted = sum(value.attempted_calls for value in step_outcomes)
    completed = sum(value.completed_calls for value in step_outcomes)
    if attempted != outcome.attempted_calls or completed != outcome.completed_calls:
        invalid("interactive step call accounting does not replay")
    failed_steps = [value for value in step_outcomes if not value.ok]
    if outcome.ok:
        summary = {key: value for key, value in runner.items() if key not in _RUNNER_CONTROL_KEYS}
        expected_response = json.dumps(summary, ensure_ascii=True, sort_keys=True, allow_nan=False)
        if (
            len(step_outcomes) != len(transcript)
            or failed_steps
            or outcome.response != expected_response
            or (set(process) != {"interactive_runner"})
        ):
            invalid("interactive successful outcome does not replay")
    elif failed_steps:
        final = step_outcomes[-1]
        if (
            failed_steps != [final]
            or len(step_outcomes) != len(transcript) + 1
            or outcome.failure_class != final.failure_class
            or (outcome.failure != final.failure)
            or outcome.response
            or (process != {**thaw_json_mapping(final.process), "interactive_runner": runner})
        ):
            invalid("interactive executor failure does not replay")
    elif (
        len(step_outcomes) not in {len(transcript), len(transcript) + 1}
        or outcome.response
        or set(process) != {"interactive_runner"}
    ):
        invalid("interactive harness failure does not replay")
    return len(step_outcomes)
