"""RethinkSkill runtime interactive."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rethinkskill.runtime.tasks import RenderedTask, materialize_rendered_workspace
from rethinkskill.runtime.types import ModelExecutor, ModelOutcome, freeze_model_outcome
from rethinkskill.utils.integrity import boundary_failure
from rethinkskill.utils.serde import (
    atomic_write_json_snapshot,
    atomic_write_snapshot,
    freeze_json_mapping,
    thaw_json_mapping,
)


class InteractiveEpisode(Protocol):
    """Small environment lifecycle consumed by the shared episode kernel."""

    def reset(self) -> object: ...

    def step(self, action: str) -> object: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class InteractiveEpisodePolicy:
    """Adapter-owned semantics injected into the core execution loop."""

    kind: str
    initialization_label: str
    environment_label: str
    action_failure: str
    open_episode: Callable[[], InteractiveEpisode]
    state_done: Callable[[object], bool]
    render_step: Callable[[RenderedTask, object, int, Sequence[Mapping[str, object]]], RenderedTask]
    parse_action: Callable[[str], str]
    transcript_row: Callable[[int, object, str, str, object], Mapping[str, object]]
    behavior_failure_row: Callable[[int, object, str, str], Mapping[str, object]]
    summarize: Callable[[object | None, Sequence[Mapping[str, object]], str], Mapping[str, object]]

    def validate(self) -> None:
        for label, value in (
            ("kind", self.kind),
            ("initialization_label", self.initialization_label),
            ("environment_label", self.environment_label),
            ("action_failure", self.action_failure),
        ):
            if type(value) is not str or not value.strip() or "\x00" in value:
                raise ValueError(f"interactive episode {label} must be exact non-empty text")
        for label, callback in (
            ("open_episode", self.open_episode),
            ("state_done", self.state_done),
            ("render_step", self.render_step),
            ("parse_action", self.parse_action),
            ("transcript_row", self.transcript_row),
            ("behavior_failure_row", self.behavior_failure_row),
            ("summarize", self.summarize),
        ):
            if not callable(callback):
                raise ValueError(f"interactive episode {label} must be callable")


INTERACTIVE_STEP_EVIDENCE_SCHEMA_VERSION = 1


def write_interactive_step_inputs(
    workspace: Path, *, step_index: int, rendered: RenderedTask
) -> tuple[Path, dict[str, object]]:
    """Persist the invocation and freeze the exact rendered step contract."""
    step_root = workspace / "steps" / f"{step_index:03d}"
    invocation_path = step_root / "invocation.txt"
    invocation_snapshot = atomic_write_snapshot(
        invocation_path, rendered.invocation.encode("utf-8")
    )
    prefix = f"steps/{step_index:03d}"
    return (
        step_root,
        {
            "schema_version": INTERACTIVE_STEP_EVIDENCE_SCHEMA_VERSION,
            "step": step_index,
            "rendered": {
                "task_markdown": rendered.task_markdown,
                "skill_markdown": rendered.skill_markdown,
                "invocation": rendered.invocation,
                "attachments": [attachment.public() for attachment in rendered.attachments],
            },
            "invocation": {
                "path": f"{prefix}/invocation.txt",
                "sha256": invocation_snapshot.sha256,
            },
        },
    )


def complete_interactive_step_evidence(
    step_root: Path, evidence: dict[str, object], *, outcome: ModelOutcome
) -> dict[str, object]:
    """Persist the validated outcome and bind its raw and structured bytes."""
    raw_snapshot = atomic_write_snapshot(step_root / "raw.txt", outcome.raw.encode("utf-8"))
    execution_snapshot = atomic_write_json_snapshot(step_root / "EXECUTION.json", outcome.public())
    prefix = f"steps/{int(evidence['step']):03d}"
    return {
        **evidence,
        "raw": {"path": f"{prefix}/raw.txt", "sha256": raw_snapshot.sha256},
        "execution": {"path": f"{prefix}/EXECUTION.json", "sha256": execution_snapshot.sha256},
    }


@dataclass(slots=True)
class InteractiveRunState:
    """Own environment state, calls, transcript, and step evidence."""

    environment_state: object | None = None
    attempted: int = 0
    completed: int = 0
    transcript: list[dict[str, object]] = field(default_factory=list)
    step_evidence: list[dict[str, object]] = field(default_factory=list)
    behavior_failure: str = ""


def freeze_row(value: Mapping[str, object]) -> dict[str, object]:
    return thaw_json_mapping(freeze_json_mapping(value))


def serialize(value: object, *, ensure_ascii: bool) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=True, allow_nan=False)


def runner_identity(policy: InteractiveEpisodePolicy) -> dict[str, object]:
    return {"kind": policy.kind}


def failed_outcome(
    policy: InteractiveEpisodePolicy,
    *,
    transcript: Sequence[Mapping[str, object]],
    step_evidence: Sequence[Mapping[str, object]],
    attempted: int,
    completed: int,
    failure_class: str,
    failure: str,
) -> ModelOutcome:
    frozen_transcript = [freeze_row(row) for row in transcript]
    return ModelOutcome(
        status="FAILED",
        response="",
        raw=serialize(frozen_transcript, ensure_ascii=False),
        process={
            "interactive_runner": {
                **runner_identity(policy),
                "transcript": frozen_transcript,
                "evidence_schema_version": 1,
                "step_evidence": [freeze_row(row) for row in step_evidence],
            }
        },
        attempted_calls=attempted,
        completed_calls=completed,
        failure_class=failure_class,
        failure=failure,
    )


def executor_failure(
    policy: InteractiveEpisodePolicy,
    outcome: ModelOutcome,
    *,
    transcript: Sequence[Mapping[str, object]],
    step_evidence: Sequence[Mapping[str, object]],
    attempted: int,
    completed: int,
) -> ModelOutcome:
    frozen_transcript = [freeze_row(row) for row in transcript]
    return ModelOutcome(
        status="FAILED",
        response="",
        raw=serialize(frozen_transcript, ensure_ascii=False),
        process={
            **thaw_json_mapping(outcome.process),
            "interactive_runner": {
                **runner_identity(policy),
                "transcript": frozen_transcript,
                "evidence_schema_version": 1,
                "step_evidence": [freeze_row(row) for row in step_evidence],
            },
        },
        attempted_calls=attempted,
        completed_calls=completed,
        failure_class=outcome.failure_class,
        failure=outcome.failure,
    )


def execute_interactive_steps(
    policy: InteractiveEpisodePolicy,
    rendered: RenderedTask,
    executor: ModelExecutor,
    episode: InteractiveEpisode,
    state: InteractiveRunState,
    *,
    workspace: Path,
    timeout_seconds: int,
    max_steps: int,
) -> ModelOutcome | None:
    """Run the selected step budget, returning only an early failure."""
    for step_index in range(max_steps):
        try:
            done = policy.state_done(state.environment_state)
        except Exception as exc:
            return failed_outcome(
                policy,
                transcript=state.transcript,
                step_evidence=state.step_evidence,
                attempted=state.attempted,
                completed=state.completed,
                failure_class="environment_error",
                failure=boundary_failure(policy.environment_label, exc),
            )
        if type(done) is not bool:
            return failed_outcome(
                policy,
                transcript=state.transcript,
                step_evidence=state.step_evidence,
                attempted=state.attempted,
                completed=state.completed,
                failure_class="interactive_harness_error",
                failure="interactive state_done returned a non-boolean",
            )
        if done:
            break
        try:
            step_rendered = policy.render_step(
                rendered, state.environment_state, step_index, tuple(state.transcript)
            )
            if type(step_rendered) is not RenderedTask:
                raise TypeError("render_step returned a non-RenderedTask")
            step_rendered.validate()
            if step_rendered.attachments:
                raise ValueError("interactive step attachments are not supported")
            step_workspace = workspace / "steps" / f"{step_index:03d}" / "workspace"
            materialize_rendered_workspace(
                step_rendered, step_workspace, skill_name="rethinkskill-target"
            )
            step_root, pending_step_evidence = write_interactive_step_inputs(
                workspace, step_index=step_index, rendered=step_rendered
            )
        except Exception as exc:
            return failed_outcome(
                policy,
                transcript=state.transcript,
                step_evidence=state.step_evidence,
                attempted=state.attempted,
                completed=state.completed,
                failure_class="interactive_harness_error",
                failure=boundary_failure("interactive step rendering", exc),
            )
        outcome = freeze_model_outcome(
            executor.execute(
                step_rendered, workspace=step_workspace, timeout_seconds=timeout_seconds
            )
        )
        state.step_evidence.append(
            complete_interactive_step_evidence(step_root, pending_step_evidence, outcome=outcome)
        )
        state.attempted += outcome.attempted_calls
        state.completed += outcome.completed_calls
        if not outcome.ok:
            return executor_failure(
                policy,
                outcome,
                transcript=state.transcript,
                step_evidence=state.step_evidence,
                attempted=state.attempted,
                completed=state.completed,
            )
        try:
            action = policy.parse_action(outcome.response)
        except ValueError:
            state.behavior_failure = policy.action_failure
            try:
                state.transcript.append(
                    freeze_row(
                        policy.behavior_failure_row(
                            step_index,
                            state.environment_state,
                            outcome.response,
                            state.behavior_failure,
                        )
                    )
                )
            except Exception as exc:
                return failed_outcome(
                    policy,
                    transcript=state.transcript,
                    step_evidence=state.step_evidence,
                    attempted=state.attempted,
                    completed=state.completed,
                    failure_class="interactive_harness_error",
                    failure=boundary_failure("interactive behavior transcript", exc),
                )
            break
        except Exception as exc:
            return failed_outcome(
                policy,
                transcript=state.transcript,
                step_evidence=state.step_evidence,
                attempted=state.attempted,
                completed=state.completed,
                failure_class="interactive_harness_error",
                failure=boundary_failure("interactive action parser", exc),
            )
        if type(action) is not str or not action or "\x00" in action:
            return failed_outcome(
                policy,
                transcript=state.transcript,
                step_evidence=state.step_evidence,
                attempted=state.attempted,
                completed=state.completed,
                failure_class="interactive_harness_error",
                failure="interactive parse_action returned invalid text",
            )
        previous = state.environment_state
        try:
            state.environment_state = episode.step(action)
        except Exception as exc:
            return failed_outcome(
                policy,
                transcript=state.transcript,
                step_evidence=state.step_evidence,
                attempted=state.attempted,
                completed=state.completed,
                failure_class="environment_error",
                failure=boundary_failure(policy.environment_label, exc),
            )
        try:
            state.transcript.append(
                freeze_row(
                    policy.transcript_row(
                        step_index, previous, outcome.response, action, state.environment_state
                    )
                )
            )
        except Exception as exc:
            return failed_outcome(
                policy,
                transcript=state.transcript,
                step_evidence=state.step_evidence,
                attempted=state.attempted,
                completed=state.completed,
                failure_class="interactive_harness_error",
                failure=boundary_failure("interactive transcript", exc),
            )
    return None


def execute_interactive_episode(
    policy: InteractiveEpisodePolicy,
    rendered: RenderedTask,
    executor: ModelExecutor,
    *,
    workspace: Path,
    timeout_seconds: int,
    max_steps: int,
) -> ModelOutcome:
    """Execute one bounded environment episode with exact call accounting."""
    if type(policy) is not InteractiveEpisodePolicy:
        raise TypeError("interactive execution requires an exact policy")
    policy.validate()
    if type(rendered) is not RenderedTask:
        raise TypeError("interactive execution requires an exact RenderedTask")
    if not isinstance(workspace, Path):
        raise TypeError("interactive execution workspace must be a Path")
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("interactive timeout_seconds must be positive")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("interactive max_steps must be positive")
    state = InteractiveRunState()
    try:
        episode = policy.open_episode()
    except Exception as exc:
        return failed_outcome(
            policy,
            transcript=state.transcript,
            step_evidence=state.step_evidence,
            attempted=0,
            completed=0,
            failure_class="environment_initialization_error",
            failure=boundary_failure(policy.initialization_label, exc),
        )
    try:
        try:
            state.environment_state = episode.reset()
        except Exception as exc:
            return failed_outcome(
                policy,
                transcript=state.transcript,
                step_evidence=state.step_evidence,
                attempted=state.attempted,
                completed=state.completed,
                failure_class="environment_error",
                failure=boundary_failure(policy.environment_label, exc),
            )
        failure = execute_interactive_steps(
            policy,
            rendered,
            executor,
            episode,
            state,
            workspace=workspace,
            timeout_seconds=timeout_seconds,
            max_steps=max_steps,
        )
        if failure is not None:
            return failure
    finally:
        try:
            episode.close()
        except Exception:
            pass
    if state.attempted < 1:
        return failed_outcome(
            policy,
            transcript=state.transcript,
            step_evidence=state.step_evidence,
            attempted=state.attempted,
            completed=state.completed,
            failure_class="environment_no_model_call",
            failure="interactive environment ended before a model action",
        )
    try:
        summary = freeze_row(
            policy.summarize(
                state.environment_state, tuple(state.transcript), state.behavior_failure
            )
        )
        if set(summary) & {"kind", "transcript", "evidence_schema_version", "step_evidence"}:
            raise ValueError("interactive summary uses a reserved evidence field")
    except Exception as exc:
        return failed_outcome(
            policy,
            transcript=state.transcript,
            step_evidence=state.step_evidence,
            attempted=state.attempted,
            completed=state.completed,
            failure_class="interactive_harness_error",
            failure=boundary_failure("interactive summary", exc),
        )
    return ModelOutcome(
        status="COMPLETED",
        response=serialize(summary, ensure_ascii=True),
        raw=serialize(state.transcript, ensure_ascii=False),
        process={
            "interactive_runner": {
                **runner_identity(policy),
                **summary,
                "transcript": state.transcript,
                "evidence_schema_version": 1,
                "step_evidence": state.step_evidence,
            }
        },
        attempted_calls=state.attempted,
        completed_calls=state.completed,
    )
