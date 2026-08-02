"""RethinkSkill evidence evolution."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rethinkskill.errors import ResultValidationError
from rethinkskill.evaluation.results import validate_ids
from rethinkskill.evidence.common import (
    artifact_snapshot,
    directory,
    integer,
    invalid,
    json_object,
    ledger_rows,
    manifest_receipt_pair,
    mapping,
    sha256_digest,
)
from rethinkskill.evidence.optimizer import replay_optimizer_proposal
from rethinkskill.evidence.run import validate_native_evidence
from rethinkskill.evolution.loop import EvolutionRound, skill_hash
from rethinkskill.evolution.native import evaluation_outcome
from rethinkskill.evolution.protocol import GatePolicy, GateState, gate_decision
from rethinkskill.providers.catalog import ProviderRegistration
from rethinkskill.runtime.runner import validate_public_native_capability
from rethinkskill.utils.integrity import EVOLUTION_RECEIPT_SCHEMA_VERSION
from rethinkskill.utils.plugins import ExtensionRegistration
from rethinkskill.utils.serde import snapshot_regular_file, thaw_json_mapping

_ROLE = re.compile("^[A-Za-z0-9][A-Za-z0-9._-]*$")

_NATIVE_EVALUATOR_KEYS = {
    "kind",
    "benchmark",
    "capability",
    "dataset",
    "selections",
    "dataset_files",
    "seed",
    "asset_root",
    "executor",
}

_MODEL_OPTIMIZER_KEYS = {
    "kind",
    "output_contract",
    "executor",
    "optimizer_strategy",
    "optimizer_registration",
}

_OPTIMIZER_SPEC_KEYS = {
    "name",
    "implementation",
    "description",
    "proposal_contract",
    "executor_invocations_per_proposal",
    "owns_evaluation",
    "owns_acceptance",
    "public_entry_points_supported",
}


def _component(value: object, *, label: str) -> dict[str, object]:
    component = mapping(value, label=label)
    kind = component.get("kind")
    if type(kind) is not str or not kind.strip():
        invalid(f"{label} kind must be non-empty text")
    return component


def _validate_executor(value: object, *, label: str) -> None:
    executor = _component(value, label=label)
    registration = executor.get("provider_registration")
    if registration is not None:
        try:
            ProviderRegistration.from_public(registration)
        except Exception:
            invalid(f"{label} provider registration is invalid")


def _validate_native_evaluator(evaluator: dict[str, object], *, benchmark: str) -> None:
    if set(evaluator) != _NATIVE_EVALUATOR_KEYS:
        invalid("native evolution evaluator manifest has an invalid schema")
    if evaluator.get("benchmark") != benchmark:
        invalid("native evolution evaluator benchmark does not match")
    try:
        capability = validate_public_native_capability(
            mapping(evaluator.get("capability"), label="native evolution capability"),
            benchmark=benchmark,
        )
    except Exception:
        invalid("native evolution evaluator capability is invalid")
    if thaw_json_mapping(capability) != evaluator.get("capability"):
        invalid("native evolution evaluator capability is not canonical")
    if type(evaluator.get("dataset")) is not str or not evaluator["dataset"]:
        invalid("native evolution evaluator dataset path is invalid")
    if type(evaluator.get("seed")) is not int:
        invalid("native evolution evaluator seed is invalid")
    asset_root = evaluator.get("asset_root")
    if asset_root is not None and (type(asset_root) is not str or not asset_root):
        invalid("native evolution evaluator asset root is invalid")
    dataset_files = evaluator.get("dataset_files")
    if type(dataset_files) is not list or not dataset_files:
        invalid("native evolution evaluator dataset freeze is invalid")
    observed_paths: set[str] = set()
    for index, raw in enumerate(dataset_files):
        record = mapping(raw, label=f"native dataset file {index}")
        if (
            set(record) != {"path", "sha256"}
            or type(record.get("path")) is not str
            or (not record["path"])
            or (record["path"] in observed_paths)
        ):
            invalid("native evolution evaluator dataset freeze is invalid")
        observed_paths.add(record["path"])
        sha256_digest(record.get("sha256"), label=f"native dataset file {index} SHA-256")
    selections = mapping(evaluator.get("selections"), label="native evolution selections")
    if not selections:
        invalid("native evolution evaluator has no selections")
    all_ids: set[str] = set()
    for role, raw in selections.items():
        if type(role) is not str or _ROLE.fullmatch(role) is None:
            invalid("native evolution evaluator selection role is unsafe")
        selection = mapping(raw, label=f"native selection {role!r}")
        if set(selection) != {"split", "task_ids"} or (
            type(selection.get("split")) is not str or not selection["split"].strip()
        ):
            invalid(f"native selection {role!r} has an invalid schema")
        try:
            task_ids = validate_ids(
                selection.get("task_ids", ()), label=f"native selection {role!r} task IDs"
            )
        except Exception:
            invalid(f"native selection {role!r} task IDs are invalid")
        if not task_ids or all_ids.intersection(task_ids):
            invalid("native evolution evaluator selections overlap or are empty")
        all_ids.update(task_ids)
    _validate_executor(evaluator.get("executor"), label="native evolution evaluator executor")


def _validate_optimizer(optimizer: dict[str, object]) -> None:
    strategy_raw = optimizer.get("optimizer_strategy")
    registration_raw = optimizer.get("optimizer_registration")
    if (strategy_raw is None) != (registration_raw is None):
        invalid("evolution optimizer provenance is incomplete")
    if strategy_raw is None:
        return
    strategy = mapping(strategy_raw, label="evolution optimizer strategy")
    if set(strategy) != _OPTIMIZER_SPEC_KEYS:
        invalid("evolution optimizer strategy has an invalid schema")
    for field in ("name", "implementation", "description", "proposal_contract"):
        if type(strategy.get(field)) is not str or not strategy[field].strip():
            invalid(f"evolution optimizer strategy {field} is invalid")
    invocation_limit = integer(
        strategy.get("executor_invocations_per_proposal"),
        label="optimizer executor invocation limit",
    )
    if (
        strategy.get("owns_evaluation") is not False
        or strategy.get("owns_acceptance") is not False
        or strategy.get("public_entry_points_supported") is not True
    ):
        invalid("evolution optimizer strategy ownership is invalid")
    try:
        registration = ExtensionRegistration.from_public(registration_raw)
    except Exception:
        invalid("evolution optimizer registration is invalid")
    if (
        registration.name != strategy.get("name")
        or registration.component_kind != "optimizer_strategy"
    ):
        invalid("evolution optimizer registration does not match its strategy")
    if optimizer.get("kind") == "model-skill-optimizer":
        if set(optimizer) != _MODEL_OPTIMIZER_KEYS:
            invalid("model skill optimizer manifest has an invalid schema")
        if (
            optimizer.get("output_contract") != strategy.get("proposal_contract")
            or optimizer.get("output_contract") != "strict_json_full_skill_candidate"
            or invocation_limit != 1
        ):
            invalid("model skill optimizer contract is inconsistent")
        _validate_executor(optimizer.get("executor"), label="evolution optimizer executor")


def validate_evolution_components(
    manifest: Mapping[str, object], *, benchmark: str
) -> tuple[str, str]:
    """Validate v4 component manifests and return component kinds."""
    evaluator = _component(manifest.get("evaluator"), label="evolution evaluator")
    optimizer = _component(manifest.get("optimizer"), label="evolution optimizer")
    if evaluator["kind"] == "native-skill-evaluator":
        _validate_native_evaluator(evaluator, benchmark=benchmark)
    _validate_optimizer(optimizer)
    return (str(evaluator["kind"]), str(optimizer["kind"]))


def replay_native_evaluation(
    *,
    run_root: Path,
    evaluator_manifest: Mapping[str, object],
    outcome: Mapping[str, object],
    role: str,
    round_no: int,
    expected_skill_sha256: str,
    feedback_categories: Sequence[str] | None = None,
) -> None:
    """Replay one nested native run and bind it to its outer outcome."""
    if _ROLE.fullmatch(role) is None:
        invalid("native evolution reference role is unsafe")
    relative_results = Path("evaluations") / f"{role}_round_{round_no:04d}" / "results.jsonl"
    child_root = directory(
        run_root / relative_results.parent, label=f"native evolution {role} round {round_no} run"
    )
    child_report = validate_native_evidence(child_root)
    results_digest = mapping(
        child_report.get("artifacts"), label="nested native replay artifacts"
    ).get("results_sha256")
    expected_reference = f"{relative_results.as_posix()}#sha256={results_digest}"
    if outcome.get("reference") != expected_reference:
        invalid("native evolution outcome reference does not match its child run")
    child_manifest = json_object(
        child_root / ".rethinkskill/RUN_MANIFEST.json", label="nested native run manifest"
    )
    child_receipt = json_object(
        child_root / ".rethinkskill/PROCESS_RECEIPT.json", label="nested native process receipt"
    )
    selection = mapping(child_manifest.get("selection"), label="nested native selection")
    skill = mapping(child_manifest.get("skill"), label="nested native skill")
    evaluator_selections = mapping(
        evaluator_manifest.get("selections"), label="native evolution selections"
    )
    expected_selection = mapping(
        evaluator_selections.get(role), label=f"native evolution selection {role!r}"
    )
    if (
        child_manifest.get("benchmark") != evaluator_manifest.get("benchmark")
        or child_manifest.get("capability") != evaluator_manifest.get("capability")
        or child_manifest.get("executor") != evaluator_manifest.get("executor")
        or (selection.get("split") != expected_selection.get("split"))
        or (selection.get("task_ids") != expected_selection.get("task_ids"))
        or (skill.get("sha256") != expected_skill_sha256)
    ):
        invalid("nested native run does not match its evolution binding")
    selected_tasks = integer(
        child_receipt.get("selected_tasks"), label="nested native selected tasks", minimum=1
    )
    replayed = evaluation_outcome(
        receipt=child_receipt,
        results_path=child_root / "results.jsonl",
        selected_tasks=selected_tasks,
        reference_path=relative_results.as_posix(),
    ).public()
    if feedback_categories is not None:
        allowed = set(feedback_categories)
        replayed["feedback"] = [
            item for item in replayed["feedback"] if str(item.get("category", "")) in allowed
        ]
    if replayed != outcome:
        invalid("native evolution outcome does not replay from its child run")


def call_record_complete(value: object, *, label: str) -> bool:
    record = mapping(value, label=label)
    if set(record) != {"attempted", "completed", "accounting_known"}:
        invalid(f"{label} has an invalid schema")
    attempted = integer(record.get("attempted"), label=f"{label} attempted")
    completed = integer(record.get("completed"), label=f"{label} completed")
    known = record.get("accounting_known")
    if type(known) is not bool or completed > attempted:
        invalid(f"{label} has invalid call accounting")
    return known and attempted == completed


def score(value: object, *, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or (not 0.0 <= value <= 1.0):
        invalid(f"{label} must be a finite score in [0, 1]")
    return float(value)


def outcome_calls(value: object, *, label: str) -> tuple[int, int, bool]:
    outcome = mapping(value, label=label)
    expected = {
        "valid",
        "hard",
        "soft",
        "feedback",
        "reference",
        "attempted_calls",
        "completed_calls",
        "failure_class",
        "failure",
    }
    if set(outcome) != expected:
        invalid(f"{label} has an invalid schema")
    valid = outcome.get("valid")
    if type(valid) is not bool:
        invalid(f"{label} valid flag must be boolean")
    score(outcome.get("hard"), label=f"{label} hard")
    score(outcome.get("soft"), label=f"{label} soft")
    attempted = integer(outcome.get("attempted_calls"), label=f"{label} attempted calls")
    completed = integer(outcome.get("completed_calls"), label=f"{label} completed calls")
    if completed > attempted:
        invalid(f"{label} completed calls exceed attempted calls")
    failure_class = outcome.get("failure_class")
    if (
        valid
        and failure_class is not None
        or (not valid and (type(failure_class) is not str or not failure_class.strip()))
    ):
        invalid(f"{label} failure state is inconsistent")
    if (
        type(outcome.get("feedback")) is not list
        or type(outcome.get("reference")) is not str
        or (not outcome["reference"])
        or (type(outcome.get("failure")) is not str)
    ):
        invalid(f"{label} payload is malformed")
    return (attempted, completed, valid)


def gate_state(value: object, *, label: str) -> GateState:
    record = mapping(value, label=label)
    expected = {
        "current_hard",
        "current_soft",
        "best_hard",
        "best_soft",
        "best_round",
        "noop_streak",
        "regression_streak",
    }
    if set(record) != expected:
        invalid(f"{label} has an invalid schema")
    state = GateState(
        current_hard=score(record.get("current_hard"), label="current hard"),
        current_soft=score(record.get("current_soft"), label="current soft"),
        best_hard=score(record.get("best_hard"), label="best hard"),
        best_soft=score(record.get("best_soft"), label="best soft"),
        best_round=integer(record.get("best_round"), label="best round"),
        noop_streak=integer(record.get("noop_streak"), label="noop streak"),
        regression_streak=integer(record.get("regression_streak"), label="regression streak"),
    )
    state.validate()
    return state


def gate_public(state: GateState) -> dict[str, object]:
    return {
        "current_hard": state.current_hard,
        "current_soft": state.current_soft,
        "best_hard": state.best_hard,
        "best_soft": state.best_soft,
        "best_round": state.best_round,
        "noop_streak": state.noop_streak,
        "regression_streak": state.regression_streak,
    }


def boundary_reference(
    outcome: Mapping[str, object], *, stage: str, round_no: int, role: str
) -> bool:
    reference = outcome.get("reference")
    if not isinstance(reference, str) or not reference.startswith("boundary://"):
        return False
    if (
        reference != f"boundary://{stage}/{round_no}/{role}"
        or outcome.get("valid") is not False
        or outcome.get("hard") != 0.0
        or (outcome.get("soft") != 0.0)
        or (outcome.get("feedback") != [])
        or (outcome.get("attempted_calls") != 0)
        or (outcome.get("completed_calls") != 0)
        or (outcome.get("failure_class") != "native_evaluation_invalid")
    ):
        invalid("evolution evaluator boundary outcome is inconsistent")
    return True


if TYPE_CHECKING:
    from rethinkskill.runtime.tasks import RenderedTask

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


@dataclass(frozen=True, slots=True)
class EvolutionProposalReplay:
    """Validated proposal state and exact optimizer call accounting."""

    valid: bool
    operation: str
    attempted_calls: int
    completed_calls: int


def _render_model_optimizer_round(
    *,
    round_no: int,
    current_skill_text: str,
    parent_digest: str,
    categories: tuple[str, ...],
    training: Mapping[str, object],
    history: Sequence[EvolutionRound],
) -> RenderedTask:
    """Reconstruct a model task only when that optimizer kind was recorded."""
    from rethinkskill.evolution.loop import EvaluationOutcome, OptimizationContext
    from rethinkskill.evolution.optimizer import render_model_optimizer_task

    feedback = training.get("feedback")
    if type(feedback) is not list:
        invalid(f"evolution round {round_no} training feedback is invalid")
    training_outcome = EvaluationOutcome(
        valid=True,
        hard=training["hard"],
        soft=training["soft"],
        feedback=tuple(
            mapping(item, label=f"evolution round {round_no} feedback item") for item in feedback
        ),
        reference=training["reference"],
        attempted_calls=training["attempted_calls"],
        completed_calls=training["completed_calls"],
        failure_class=None,
        failure=training["failure"],
    )
    training_outcome.validate()
    return render_model_optimizer_task(
        OptimizationContext(
            round_no=round_no,
            current_skill=current_skill_text,
            current_sha256=parent_digest,
            feedback_categories=categories,
            training=training_outcome,
            history=tuple(history),
        )
    )


def replay_evolution_proposal(
    *,
    run_root: Path,
    round_no: int,
    row: Mapping[str, object],
    candidate_sha256: str,
    parent_digest: str,
    current_skill_text: str,
    categories: tuple[str, ...],
    training: Mapping[str, object],
    history: Sequence[EvolutionRound],
    optimizer_kind: str | None,
    optimizer_manifest: Mapping[str, object] | None,
) -> EvolutionProposalReplay:
    """Validate one proposal and replay its optimizer evidence."""
    proposal = mapping(row.get("proposal"), label=f"evolution round {round_no} proposal")
    if set(proposal) != _PROPOSAL_KEYS:
        invalid(f"evolution round {round_no} proposal schema is invalid")
    proposal_valid = proposal.get("valid")
    if type(proposal_valid) is not bool:
        invalid(f"evolution round {round_no} proposal valid flag is invalid")
    attempted = integer(
        proposal.get("attempted_calls"),
        label=f"evolution round {round_no} optimizer attempted calls",
    )
    completed = integer(
        proposal.get("completed_calls"),
        label=f"evolution round {round_no} optimizer completed calls",
    )
    if completed > attempted:
        invalid(f"evolution round {round_no} optimizer calls are invalid")
    operation = proposal.get("operation")
    if (
        operation not in {"add", "delete", "replace", "noop"}
        or proposal.get("candidate_sha256") != candidate_sha256
        or row.get("operation") != operation
    ):
        invalid(f"evolution round {round_no} proposal binding is invalid")
    failure_class = proposal.get("failure_class")
    if (
        type(proposal.get("rationale")) is not str
        or type(proposal.get("failure")) is not str
        or (proposal_valid and (failure_class is not None or proposal["failure"] != ""))
        or (not proposal_valid and (type(failure_class) is not str or not failure_class.strip()))
    ):
        invalid(f"evolution round {round_no} proposal state is invalid")
    rendered = None
    if optimizer_kind == "model-skill-optimizer":
        rendered = _render_model_optimizer_round(
            round_no=round_no,
            current_skill_text=current_skill_text,
            parent_digest=parent_digest,
            categories=categories,
            training=training,
            history=history,
        )
    if optimizer_manifest is None:
        invalid("evolution optimizer manifest is unavailable")
    replay_optimizer_proposal(
        run_root=run_root,
        optimizer_manifest=optimizer_manifest,
        evidence_reference=row.get("optimizer_evidence"),
        proposal=proposal,
        round_no=round_no,
        parent_sha256=parent_digest,
        rendered=rendered,
    )
    return EvolutionProposalReplay(
        valid=proposal_valid,
        operation=operation,
        attempted_calls=attempted,
        completed_calls=completed,
    )


@dataclass(frozen=True, slots=True)
class EvolutionTerminalInputs:
    """Round-replay outputs required to validate one terminal receipt."""

    run_root: Path
    receipt: Mapping[str, object]
    rows: tuple[dict[str, object], ...]
    skill_digests: Mapping[str, str]
    initial_digest: str
    optimizer_manifest: Mapping[str, object] | None
    parent_digest: str
    gate_state: GateState
    rounds_requested: int
    target_attempted: int
    target_completed: int
    optimizer_attempted: int
    optimizer_completed: int
    optimizer_evidence_runs: int


@dataclass(frozen=True, slots=True)
class EvolutionTerminalReplay:
    """Validated terminal status and replayed optimizer-evidence count."""

    status: str
    optimizer_evidence_runs: int


def reconcile_evolution_terminal(inputs: EvolutionTerminalInputs) -> EvolutionTerminalReplay:
    """Reconcile terminal skills, calls, optimizer artifacts, and status."""
    run_root = inputs.run_root
    receipt = inputs.receipt
    rows = inputs.rows
    skill_digests = inputs.skill_digests
    gate_state = inputs.gate_state
    optimizer_evidence_runs = inputs.optimizer_evidence_runs
    if rows:
        if (
            rows[-1].get("current_sha256") != skill_digests["final_current"]
            or rows[-1].get("best_sha256") != skill_digests["best"]
        ):
            invalid("terminal evolution skills do not match the final round")
    elif (
        skill_digests["final_current"] != inputs.initial_digest
        or skill_digests["best"] != inputs.initial_digest
    ):
        invalid("zero-round evolution skills do not match the initial skill")
    optimizer_manifest = inputs.optimizer_manifest
    if optimizer_manifest is None:
        invalid("evolution optimizer manifest is unavailable")
    if "terminal_optimizer_evidence" not in receipt:
        invalid("evolution receipt omits terminal optimizer evidence")
    terminal_optimizer_reference = receipt.get("terminal_optimizer_evidence")
    optimizer_calls = mapping(receipt.get("optimizer_calls"), label="optimizer calls")
    optimizer_accounting_known = optimizer_calls.get("accounting_known")
    if (
        type(optimizer_accounting_known) is not bool
        or (terminal_optimizer_reference is None) != optimizer_accounting_known
    ):
        invalid("terminal optimizer evidence accounting is inconsistent")
    expected_optimizer_rounds = set(range(1, len(rows) + 1))
    if terminal_optimizer_reference is not None:
        failure_round = receipt.get("failure_round")
        if receipt.get("failure_stage") != "optimization" or failure_round != len(rows) + 1:
            invalid("terminal optimizer evidence state is inconsistent")
        terminal_proposal = replay_optimizer_proposal(
            run_root=run_root,
            optimizer_manifest=optimizer_manifest,
            evidence_reference=terminal_optimizer_reference,
            proposal=None,
            round_no=failure_round,
            parent_sha256=inputs.parent_digest,
            rendered=None,
            replay_model=False,
        )
        if (
            terminal_proposal.get("valid") is not False
            or terminal_proposal.get("candidate_sha256") != inputs.parent_digest
            or terminal_proposal.get("operation") != "noop"
            or (type(terminal_proposal.get("rationale")) is not str)
            or (terminal_proposal.get("attempted_calls") != 0)
            or (terminal_proposal.get("completed_calls") != 0)
            or (type(terminal_proposal.get("failure_class")) is not str)
            or (not terminal_proposal["failure_class"].strip())
            or (type(terminal_proposal.get("failure")) is not str)
            or (terminal_proposal.get("failure_class") != receipt.get("failure_class"))
            or (terminal_proposal.get("failure") != receipt.get("failure"))
        ):
            invalid("terminal optimizer proposal is inconsistent")
        expected_optimizer_rounds.add(failure_round)
        optimizer_evidence_runs += 1
    optimizer_root = run_root / "optimizer"
    if expected_optimizer_rounds:
        directory(optimizer_root, label="evolution optimizer directory")
        observed_rounds: set[str] = set()
        for entry in optimizer_root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                invalid("evolution optimizer directory has an unsafe entry")
            if not entry.name.startswith("round_"):
                invalid("evolution optimizer directory has an unknown entry")
            observed_rounds.add(entry.name)
        expected_round_names = {f"round_{round_no:04d}" for round_no in expected_optimizer_rounds}
        if observed_rounds != expected_round_names:
            invalid("evolution optimizer artifact rounds do not match lineage")
    elif optimizer_root.exists() or optimizer_root.is_symlink():
        invalid("evolution run has unbound optimizer artifacts")
    target_complete = call_record_complete(receipt.get("target_calls"), label="target calls")
    optimizer_complete = call_record_complete(
        receipt.get("optimizer_calls"), label="optimizer calls"
    )
    target_record = mapping(receipt.get("target_calls"), label="target calls")
    optimizer_record = mapping(receipt.get("optimizer_calls"), label="optimizer calls")
    for label, record, attempted, completed in (
        ("target", target_record, inputs.target_attempted, inputs.target_completed),
        ("optimizer", optimizer_record, inputs.optimizer_attempted, inputs.optimizer_completed),
    ):
        recorded_attempted = integer(record.get("attempted"), label=f"{label} attempted calls")
        recorded_completed = integer(record.get("completed"), label=f"{label} completed calls")
        if attempted > recorded_attempted or completed > recorded_completed:
            invalid(f"evolution {label} calls omit recorded round outcomes")
        if receipt.get("status") == "RETHINKSKILL_EVOLUTION_VALIDATED" and (
            attempted != recorded_attempted or completed != recorded_completed
        ):
            invalid(f"evolution {label} exact calls do not replay")
    final_record = mapping(receipt.get("final_current"), label="final current skill record")
    best_record = mapping(receipt.get("best"), label="best skill record")
    if (
        final_record.get("hard") != gate_state.current_hard
        or final_record.get("soft") != gate_state.current_soft
        or best_record.get("hard") != gate_state.best_hard
        or (best_record.get("soft") != gate_state.best_soft)
        or (best_record.get("round") != gate_state.best_round)
    ):
        invalid("evolution terminal scores do not replay from gate state")
    successful = (
        receipt.get("failure_class") is None
        and target_complete
        and optimizer_complete
        and (len(rows) == inputs.rounds_requested)
    )
    expected_status = (
        "RETHINKSKILL_EVOLUTION_VALIDATED" if successful else "RETHINKSKILL_EVOLUTION_INVALID"
    )
    if receipt.get("status") != expected_status:
        invalid("evolution terminal status does not replay from frozen evidence")
    return EvolutionTerminalReplay(
        status=expected_status, optimizer_evidence_runs=optimizer_evidence_runs
    )


@dataclass(frozen=True, slots=True)
class EvolutionRoundReplay:
    """Terminal state and accounting derived only from recorded rounds."""

    gate_state: GateState
    parent_digest: str
    target_attempted: int
    target_completed: int
    optimizer_attempted: int
    optimizer_completed: int
    nested_native_runs: int
    optimizer_evidence_runs: int


def replay_evolution_rounds(
    *,
    run_root: Path,
    rows: Sequence[Mapping[str, object]],
    evaluator_kind: str | None,
    optimizer_kind: str | None,
    evaluator_manifest: Mapping[str, object],
    optimizer_manifest: Mapping[str, object] | None,
    train_role: object,
    validation_role: object,
    feedback_categories: object,
    skills_root: Path,
    rounds_root: Path,
    policy: GatePolicy,
    initial_gate_state: GateState,
    initial_digest: str,
    initial_skill_text: str,
    target_attempted: int,
    target_completed: int,
    nested_native_runs: int,
) -> EvolutionRoundReplay:
    """Replay every complete ledger round without terminal receipt policy."""
    gate_state = initial_gate_state
    parent_digest = initial_digest
    best_digest = initial_digest
    current_skill_text = initial_skill_text
    optimizer_attempted = 0
    optimizer_completed = 0
    optimizer_evidence_runs = 0
    history: list[EvolutionRound] = []
    categories = tuple(feedback_categories) if isinstance(feedback_categories, list) else ()
    for round_no, row in enumerate(rows, start=1):
        round_path = rounds_root / f"round_{round_no:04d}.json"
        if json_object(round_path, label=f"evolution round {round_no} artifact") != row:
            invalid(f"evolution round {round_no} differs from the ledger")
        candidate_path = skills_root / f"round_{round_no:04d}_candidate.md"
        try:
            candidate_snapshot = snapshot_regular_file(candidate_path)
            candidate_text = candidate_snapshot.payload.decode("utf-8")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise ResultValidationError(
                f"evolution round {round_no} candidate skill is unsafe"
            ) from exc
        if candidate_snapshot.sha256 != row.get("candidate_sha256"):
            invalid(f"evolution round {round_no} candidate skill drifted")
        if row.get("parent_sha256") != parent_digest:
            invalid(f"evolution round {round_no} parent lineage is invalid")
        training = mapping(row.get("training"), label=f"evolution round {round_no} training")
        attempted, completed, training_valid = outcome_calls(
            training, label=f"evolution round {round_no} training"
        )
        target_attempted += attempted
        target_completed += completed
        if not training_valid:
            invalid(f"recorded evolution round {round_no} has invalid training")
        if evaluator_kind == "native-skill-evaluator":
            replay_native_evaluation(
                run_root=run_root,
                evaluator_manifest=evaluator_manifest,
                outcome=training,
                role=str(train_role),
                round_no=round_no,
                expected_skill_sha256=parent_digest,
                feedback_categories=categories,
            )
            nested_native_runs += 1
        proposal = replay_evolution_proposal(
            run_root=run_root,
            round_no=round_no,
            row=row,
            candidate_sha256=candidate_snapshot.sha256,
            parent_digest=parent_digest,
            current_skill_text=current_skill_text,
            categories=categories,
            training=training,
            history=history,
            optimizer_kind=optimizer_kind,
            optimizer_manifest=optimizer_manifest,
        )
        proposal_valid = proposal.valid
        operation = proposal.operation
        optimizer_attempted += proposal.attempted_calls
        optimizer_completed += proposal.completed_calls
        optimizer_evidence_runs += 1
        validation = row.get("validation")
        if proposal_valid and operation != "noop":
            attempted, completed, validation_valid = outcome_calls(
                validation, label=f"evolution round {round_no} validation"
            )
            target_attempted += attempted
            target_completed += completed
            validation_record = mapping(validation, label=f"evolution round {round_no} validation")
            if row.get("candidate_hard") != validation_record.get("hard") or row.get(
                "candidate_soft"
            ) != validation_record.get("soft"):
                invalid(f"evolution round {round_no} candidate scores drifted")
            if evaluator_kind == "native-skill-evaluator":
                replay_native_evaluation(
                    run_root=run_root,
                    evaluator_manifest=evaluator_manifest,
                    outcome=validation_record,
                    role=str(validation_role),
                    round_no=round_no,
                    expected_skill_sha256=candidate_snapshot.sha256,
                )
                nested_native_runs += 1
        else:
            validation_valid = False
            if (
                validation is not None
                or row.get("candidate_hard") is not None
                or row.get("candidate_soft") is not None
            ):
                invalid(f"evolution round {round_no} has unexpected validation")
        if not proposal_valid or (operation != "noop" and (not validation_valid)):
            expected_action = "infrastructure_invalid"
            expected_state = gate_state
        elif operation == "noop":
            decision = gate_decision(
                policy,
                gate_state,
                hard=gate_state.current_hard,
                soft=gate_state.current_soft,
                round_no=round_no,
            )
            expected_action = "flat"
            expected_state = decision.state
        else:
            validation_record = mapping(validation, label=f"evolution round {round_no} validation")
            decision = gate_decision(
                policy,
                gate_state,
                hard=score(validation_record.get("hard"), label="candidate hard"),
                soft=score(validation_record.get("soft"), label="candidate soft"),
                round_no=round_no,
            )
            expected_action = decision.action
            expected_state = decision.state
        if row.get("action") != expected_action:
            invalid(f"evolution round {round_no} gate action does not replay")
        observed_state = gate_state(
            row.get("gate_state"), label=f"evolution round {round_no} gate state"
        )
        if gate_public(observed_state) != gate_public(expected_state):
            invalid(f"evolution round {round_no} gate state does not replay")
        current_digest = (
            candidate_snapshot.sha256
            if expected_action in {"accept", "accept_new_best"}
            else parent_digest
        )
        if expected_action == "accept_new_best":
            best_digest = candidate_snapshot.sha256
        if row.get("current_sha256") != current_digest or row.get("best_sha256") != best_digest:
            invalid(f"evolution round {round_no} accepted lineage is invalid")
        gate_state = expected_state
        history.append(
            EvolutionRound(
                round_no=round_no,
                action=str(row.get("action")),
                operation=str(operation),
                parent_sha256=parent_digest,
                candidate_sha256=candidate_snapshot.sha256,
                current_sha256=current_digest,
                best_sha256=best_digest,
                candidate_hard=row.get("candidate_hard"),
                candidate_soft=row.get("candidate_soft"),
            )
        )
        if expected_action in {"accept", "accept_new_best"}:
            current_skill_text = candidate_text
        parent_digest = current_digest
    return EvolutionRoundReplay(
        gate_state=gate_state,
        parent_digest=parent_digest,
        target_attempted=target_attempted,
        target_completed=target_completed,
        optimizer_attempted=optimizer_attempted,
        optimizer_completed=optimizer_completed,
        nested_native_runs=nested_native_runs,
        optimizer_evidence_runs=optimizer_evidence_runs,
    )


def validate_evolution_evidence(run_root: Path) -> dict[str, object]:
    control = run_root / ".rethinkskill"
    manifest_path = control / "EVOLUTION_MANIFEST.json"
    receipt_path = control / "EVOLUTION_RECEIPT.json"
    manifest, receipt, manifest_snapshot, receipt_snapshot = manifest_receipt_pair(
        manifest_path=manifest_path,
        receipt_path=receipt_path,
        manifest_label="evolution run manifest",
        receipt_label="evolution receipt",
        declaration_field="evolution_receipt_schema_version",
        schema_version=EVOLUTION_RECEIPT_SCHEMA_VERSION,
    )
    if manifest.get("status") != "RETHINKSKILL_EVOLUTION_AUTHORIZED":
        invalid("evolution manifest has an invalid authorization status")
    if receipt.get("status") not in {
        "RETHINKSKILL_EVOLUTION_VALIDATED",
        "RETHINKSKILL_EVOLUTION_INVALID",
    }:
        invalid("evolution receipt has an invalid terminal status")
    benchmark = manifest.get("benchmark")
    if type(benchmark) is not str or not benchmark:
        invalid("evolution manifest benchmark is invalid")
    component_kinds = validate_evolution_components(manifest, benchmark=benchmark)
    evaluator_kind, optimizer_kind = component_kinds
    evaluator_manifest = mapping(manifest.get("evaluator"), label="evolution evaluator")
    nested_native_runs = 0
    ledger_path = run_root / "EVOLUTION_LEDGER.jsonl"
    ledger_ref, ledger_snapshot = artifact_snapshot(
        receipt.get("ledger"),
        path=ledger_path,
        label="evolution ledger",
        keys=frozenset({"path", "sha256", "rows"}),
    )
    rows = ledger_rows(ledger_snapshot, label="evolution ledger")
    if ledger_ref.get("rows") != len(rows):
        invalid("evolution ledger row count does not match its receipt")
    if ledger_ref.get("path") != "EVOLUTION_LEDGER.jsonl":
        invalid("evolution ledger path is not portable")
    run_manifest_reference = mapping(
        receipt.get("run_manifest"), label="evolution run manifest reference"
    )
    if run_manifest_reference.get("path") != ".rethinkskill/EVOLUTION_MANIFEST.json":
        invalid("evolution run manifest path is not portable")
    if [row.get("round_no") for row in rows] != list(range(1, len(rows) + 1)):
        invalid("evolution ledger rounds are not contiguous")
    rounds_requested = integer(
        receipt.get("rounds_requested"), label="requested evolution rounds", minimum=1
    )
    if receipt.get("rounds_recorded") != len(rows) or receipt.get("benchmark") != manifest.get(
        "benchmark"
    ):
        invalid("evolution receipt summary does not match frozen artifacts")
    options = mapping(manifest.get("options"), label="evolution options")
    if options.get("rounds") != rounds_requested or options.get("arm") != receipt.get("arm"):
        invalid("evolution manifest and receipt options do not match")
    train_role = options.get("train_split")
    validation_role = options.get("validation_split")
    feedback_categories = options.get("feedback_categories")
    if (
        type(train_role) is not str
        or type(validation_role) is not str
        or type(feedback_categories) is not list
        or (not feedback_categories)
        or (not all(type(category) is str and category.strip() for category in feedback_categories))
        or (len(feedback_categories) != len(set(feedback_categories)))
    ):
        invalid("evolution component-bound options are invalid")
    if evaluator_kind == "native-skill-evaluator":
        selections = mapping(
            evaluator_manifest.get("selections"), label="native evolution selections"
        )
        if train_role not in selections or validation_role not in selections:
            invalid("native evaluator does not cover evolution split roles")
    elif (run_root / "evaluations").exists():
        invalid("opaque evaluator has unbound native evaluation artifacts")
    skills_root = directory(run_root / "skills", label="evolution skills directory")
    rounds_root = directory(run_root / "rounds", label="evolution rounds directory")
    skill_digests: dict[str, str] = {}
    for name, path in (
        ("final_current", run_root / "final_current_skill.md"),
        ("best", run_root / "best_skill.md"),
    ):
        try:
            skill_snapshot = snapshot_regular_file(path)
            text = skill_snapshot.payload.decode("utf-8")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise ResultValidationError(f"{name} skill is not readable UTF-8") from exc
        record = mapping(receipt.get(name), label=f"{name} skill record")
        digest = skill_hash(text)
        if record.get("sha256") != digest:
            invalid(f"{name} skill SHA-256 does not match its receipt")
        skill_digests[name] = digest
    initial_path = skills_root / "initial.md"
    initial_record = mapping(
        manifest.get("initial_skill"), label="evolution initial skill provenance"
    )
    try:
        initial_snapshot = snapshot_regular_file(initial_path)
    except (OSError, ValueError) as exc:
        raise ResultValidationError("initial skill snapshot could not be read safely") from exc
    initial_digest = initial_snapshot.sha256
    try:
        current_skill_text = initial_snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResultValidationError("initial evolution skill is not readable UTF-8") from exc
    if initial_record.get("sha256") != initial_digest:
        invalid("initial skill snapshot does not match its manifest")
    initial_score = mapping(receipt.get("initial_score"), label="evolution initial score")
    target_attempted, target_completed, _ = outcome_calls(
        initial_score, label="evolution initial score"
    )
    if evaluator_kind == "native-skill-evaluator" and (
        not boundary_reference(
            initial_score, stage="initial_validation", round_no=0, role=str(validation_role)
        )
    ):
        replay_native_evaluation(
            run_root=run_root,
            evaluator_manifest=evaluator_manifest,
            outcome=initial_score,
            role=str(validation_role),
            round_no=0,
            expected_skill_sha256=initial_digest,
        )
        nested_native_runs += 1
    gate_record = mapping(manifest.get("gate"), label="evolution gate")
    if set(gate_record) != {"hard_dead_band", "soft_rescue_delta"}:
        invalid("evolution gate has an invalid schema")
    try:
        policy = GatePolicy(
            hard_dead_band=gate_record["hard_dead_band"],
            soft_rescue_delta=gate_record["soft_rescue_delta"],
        )
        policy.validate()
    except Exception:
        invalid("evolution gate policy is invalid")
    gate_state = GateState(
        current_hard=score(initial_score.get("hard"), label="initial hard"),
        current_soft=score(initial_score.get("soft"), label="initial soft"),
        best_hard=score(initial_score.get("hard"), label="initial hard"),
        best_soft=score(initial_score.get("soft"), label="initial soft"),
        best_round=0,
    )
    optimizer_manifest = mapping(manifest.get("optimizer"), label="evolution optimizer")
    round_replay = replay_evolution_rounds(
        run_root=run_root,
        rows=rows,
        evaluator_kind=evaluator_kind,
        optimizer_kind=optimizer_kind,
        evaluator_manifest=evaluator_manifest,
        optimizer_manifest=optimizer_manifest,
        train_role=train_role,
        validation_role=validation_role,
        feedback_categories=feedback_categories,
        skills_root=skills_root,
        rounds_root=rounds_root,
        policy=policy,
        initial_gate_state=gate_state,
        initial_digest=initial_digest,
        initial_skill_text=current_skill_text,
        target_attempted=target_attempted,
        target_completed=target_completed,
        nested_native_runs=nested_native_runs,
    )
    gate_state = round_replay.gate_state
    parent_digest = round_replay.parent_digest
    target_attempted = round_replay.target_attempted
    target_completed = round_replay.target_completed
    optimizer_attempted = round_replay.optimizer_attempted
    optimizer_completed = round_replay.optimizer_completed
    nested_native_runs = round_replay.nested_native_runs
    optimizer_evidence_runs = round_replay.optimizer_evidence_runs
    terminal = reconcile_evolution_terminal(
        EvolutionTerminalInputs(
            run_root=run_root,
            receipt=receipt,
            rows=tuple(rows),
            skill_digests=skill_digests,
            initial_digest=initial_digest,
            optimizer_manifest=optimizer_manifest,
            parent_digest=parent_digest,
            gate_state=gate_state,
            rounds_requested=rounds_requested,
            target_attempted=target_attempted,
            target_completed=target_completed,
            optimizer_attempted=optimizer_attempted,
            optimizer_completed=optimizer_completed,
            optimizer_evidence_runs=optimizer_evidence_runs,
        )
    )
    return {
        "run_kind": "evolution",
        "terminal_status": terminal.status,
        "artifacts": {
            "manifest_sha256": manifest_snapshot.sha256,
            "receipt_sha256": receipt_snapshot.sha256,
            "ledger_sha256": ledger_snapshot.sha256,
            "ledger_rows": len(rows),
            "initial_skill_sha256": initial_digest,
            "final_current_sha256": skill_digests["final_current"],
            "best_sha256": skill_digests["best"],
            "nested_native_runs": nested_native_runs,
            "optimizer_evidence_runs": terminal.optimizer_evidence_runs,
        },
    }
