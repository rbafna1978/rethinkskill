"""RethinkSkill evolution loop."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.evolution.protocol import GatePolicy, GateState, gate_decision
from rethinkskill.providers.authorization import (
    TerminalCallAccounting,
    initialize_fresh_output_root,
    require_model_call_authorization,
    terminal_failure_boundary,
    validate_frozen_output_scope,
)
from rethinkskill.utils.fs import OutputLock, Repository, utc_now
from rethinkskill.utils.integrity import (
    EVOLUTION_RECEIPT_SCHEMA_VERSION,
    boundary_failure,
    exact_call_counts,
    freeze_manifest,
    freeze_receipt,
    validate_frozen_plan_fields,
    validate_plan_manifest,
    validate_zero_call_preflight,
)
from rethinkskill.utils.release import freeze_component_manifest, regular_file_inventory
from rethinkskill.utils.serde import (
    RegularFileSnapshot,
    atomic_write,
    atomic_write_json,
    atomic_write_json_snapshot,
    atomic_write_snapshot,
    canonical_json_bytes,
    freeze_json_mapping_sequence,
    sha256_bytes,
    snapshot_regular_file,
    thaw_json_mapping,
)

_SPLIT_ROLE = re.compile("^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class EvolutionOptions:
    rounds: int
    arm: str
    feedback_categories: tuple[str, ...]
    train_split: str = "train"
    validation_split: str = "val"
    max_skill_chars: int = 80000

    def validate(self) -> None:
        if type(self.rounds) is not int or self.rounds < 1:
            raise ConfigurationError("evolution rounds must be positive")
        if type(self.arm) is not str or not self.arm.strip():
            raise ConfigurationError("evolution arm must be non-empty")
        if (
            type(self.feedback_categories) is not tuple
            or not self.feedback_categories
            or (
                not all(
                    type(category) is str and category.strip()
                    for category in self.feedback_categories
                )
            )
        ):
            raise ConfigurationError(
                "evolution feedback_categories must be a non-empty tuple of strings"
            )
        if len(self.feedback_categories) != len(set(self.feedback_categories)):
            raise ConfigurationError("evolution feedback_categories contain duplicates")
        if (
            type(self.train_split) is not str
            or _SPLIT_ROLE.fullmatch(self.train_split) is None
            or type(self.validation_split) is not str
            or (_SPLIT_ROLE.fullmatch(self.validation_split) is None)
        ):
            raise ConfigurationError("evolution splits must be safe role identifiers")
        if type(self.max_skill_chars) is not int or self.max_skill_chars < 1:
            raise ConfigurationError("max_skill_chars must be positive")


@dataclass(frozen=True, slots=True)
class EvolutionPlan:
    repository: Repository
    benchmark: str
    initial_skill: Path
    output_root: Path
    gate: GatePolicy
    options: EvolutionOptions
    preflight: Mapping[str, object]


def plan_evolution(
    *,
    repository: Repository,
    benchmark: str,
    initial_skill: Path,
    output_root: Path,
    gate: GatePolicy,
    options: EvolutionOptions,
) -> EvolutionPlan:
    if type(repository) is not Repository:
        raise ConfigurationError("evolution planning requires an exact Repository")
    if type(options) is not EvolutionOptions:
        raise ConfigurationError("evolution planning requires exact EvolutionOptions")
    if type(gate) is not GatePolicy:
        raise ConfigurationError("evolution planning requires an exact GatePolicy")
    options.validate()
    gate.validate()
    if type(benchmark) is not str or not benchmark.strip():
        raise ConfigurationError("evolution benchmark must be non-empty")
    skill_path = initial_skill.expanduser().absolute()
    try:
        skill_snapshot = snapshot_regular_file(skill_path)
        skill_text = skill_snapshot.payload.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ConfigurationError(
            f"initial skill must be a regular, non-symlink UTF-8 file: {skill_path}"
        ) from exc
    if not skill_text.strip():
        raise ConfigurationError("initial skill must be non-empty")
    if len(skill_text) > options.max_skill_chars:
        raise ConfigurationError(f"initial skill exceeds max_skill_chars={options.max_skill_chars}")
    run_root = repository.resolve_relative("runs", must_exist=False)
    safe_output = repository.validate_new_output(output_root, run_root)
    preflight = freeze_manifest(
        {
            "schema_version": 2,
            "status": "RETHINKSKILL_EVOLUTION_PREFLIGHT_PASS",
            "created_at": utc_now(),
            "benchmark": benchmark,
            "initial_skill": {"path": str(skill_path), "sha256": skill_snapshot.sha256},
            "output_root": str(safe_output),
            "gate": {
                "hard_dead_band": gate.hard_dead_band,
                "soft_rescue_delta": gate.soft_rescue_delta,
            },
            "options": {
                "rounds": options.rounds,
                "arm": options.arm,
                "feedback_categories": list(options.feedback_categories),
                "train_split": options.train_split,
                "validation_split": options.validation_split,
                "max_skill_chars": options.max_skill_chars,
            },
            "model_calls": 0,
            "target_calls": 0,
            "optimizer_calls": 0,
        }
    )
    return EvolutionPlan(
        repository=repository,
        benchmark=benchmark,
        initial_skill=skill_path,
        output_root=safe_output,
        gate=gate,
        options=options,
        preflight=preflight,
    )


EDIT_OPERATIONS = frozenset({"add", "delete", "replace", "noop"})


def skill_hash(skill: str) -> str:
    return sha256_bytes(skill.encode("utf-8"))


def validate_call_counts(*, attempted: int, completed: int, label: str) -> None:
    try:
        exact_call_counts(attempted, completed, label=label)
    except ValueError as exc:
        raise ResultValidationError(str(exc)) from exc


def validate_evolution_plan_freeze(plan: EvolutionPlan) -> str:
    """Reject plan or initial-skill drift before output or model calls."""
    from rethinkskill.evolution.protocol import GatePolicy
    from rethinkskill.utils.fs import Repository

    if type(plan) is not EvolutionPlan:
        raise ConfigurationError("evolution execution requires an exact EvolutionPlan")
    if type(plan.repository) is not Repository:
        raise ConfigurationError("evolution plan requires an exact Repository")
    if type(plan.options) is not EvolutionOptions:
        raise ConfigurationError("evolution plan requires exact EvolutionOptions")
    if type(plan.gate) is not GatePolicy:
        raise ConfigurationError("evolution plan requires an exact GatePolicy")
    plan.options.validate()
    plan.gate.validate()
    preflight = validate_plan_manifest(plan.preflight, label="evolution preflight")
    if type(plan.benchmark) is not str or not plan.benchmark.strip():
        raise ConfigurationError("evolution benchmark must be non-empty")
    run_root = plan.repository.resolve_relative("runs", must_exist=False)
    safe_output = validate_frozen_output_scope(
        repository=plan.repository,
        output_root=plan.output_root,
        run_root=run_root,
        label="evolution",
        require_empty=True,
    )
    skill_path = plan.initial_skill.expanduser().absolute()
    try:
        skill_snapshot = snapshot_regular_file(skill_path)
        skill_payload = skill_snapshot.payload
        skill_text = skill_payload.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ConfigurationError(
            f"initial skill must be a regular, non-symlink UTF-8 file: {skill_path}"
        ) from exc
    if not skill_text.strip():
        raise ConfigurationError("initial skill must be non-empty")
    if len(skill_text) > plan.options.max_skill_chars:
        raise ConfigurationError(
            f"initial skill exceeds max_skill_chars={plan.options.max_skill_chars}"
        )
    expected = {
        "benchmark": plan.benchmark,
        "initial_skill": {"path": str(skill_path), "sha256": skill_snapshot.sha256},
        "output_root": str(safe_output),
        "gate": {
            "hard_dead_band": plan.gate.hard_dead_band,
            "soft_rescue_delta": plan.gate.soft_rescue_delta,
        },
        "options": {
            "rounds": plan.options.rounds,
            "arm": plan.options.arm,
            "feedback_categories": list(plan.options.feedback_categories),
            "train_split": plan.options.train_split,
            "validation_split": plan.options.validation_split,
            "max_skill_chars": plan.options.max_skill_chars,
        },
    }
    validate_frozen_plan_fields(preflight, expected, label="evolution")
    validate_zero_call_preflight(
        preflight,
        label="evolution preflight",
        expected_status="RETHINKSKILL_EVOLUTION_PREFLIGHT_PASS",
        call_fields=("model_calls", "target_calls", "optimizer_calls"),
    )
    return skill_text


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """One scorer summary plus feedback from a frozen task selection."""

    valid: bool
    hard: float
    soft: float
    feedback: tuple[Mapping[str, object], ...]
    reference: str
    attempted_calls: int
    completed_calls: int
    failure_class: str | None = None
    failure: str = ""

    def validate(self) -> None:
        validate_call_counts(
            attempted=self.attempted_calls, completed=self.completed_calls, label="evaluation"
        )
        if type(self.valid) is not bool:
            raise ResultValidationError("evaluation valid flag must be boolean")
        if (
            type(self.hard) not in (int, float)
            or not math.isfinite(self.hard)
            or (not 0.0 <= self.hard <= 1.0)
        ):
            raise ResultValidationError("evaluation hard score must be in [0, 1]")
        if (
            type(self.soft) not in (int, float)
            or not math.isfinite(self.soft)
            or (not 0.0 <= self.soft <= 1.0)
        ):
            raise ResultValidationError("evaluation soft score must be in [0, 1]")
        if type(self.feedback) is not tuple or not all(
            isinstance(item, Mapping) for item in self.feedback
        ):
            raise ResultValidationError("evaluation feedback must be a tuple of mappings")
        try:
            canonical_json_bytes([thaw_json_mapping(item) for item in self.feedback])
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise ResultValidationError(
                "evaluation feedback must be finite canonical JSON"
            ) from exc
        if type(self.reference) is not str or not self.reference:
            raise ResultValidationError("evaluation reference must be non-empty")
        if self.failure_class is not None and (
            type(self.failure_class) is not str or not self.failure_class.strip()
        ):
            raise ResultValidationError(
                "evaluation failure_class must be a non-empty string or None"
            )
        if type(self.failure) is not str:
            raise ResultValidationError("evaluation failure must be a string")
        if self.valid and self.failure_class:
            raise ResultValidationError("valid evaluation cannot carry a failure_class")
        if not self.valid and (not self.failure_class):
            raise ResultValidationError("invalid evaluation requires a failure_class")

    def public(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "hard": self.hard,
            "soft": self.soft,
            "feedback": [thaw_json_mapping(item) for item in self.feedback],
            "reference": self.reference,
            "attempted_calls": self.attempted_calls,
            "completed_calls": self.completed_calls,
            "failure_class": self.failure_class,
            "failure": self.failure,
        }


def freeze_evaluation_outcome(value: EvaluationOutcome) -> EvaluationOutcome:
    """Detach one exact evaluator result before gates or optimizer exposure."""
    if type(value) is not EvaluationOutcome:
        raise TypeError("evaluator must return an exact EvaluationOutcome")
    try:
        feedback = freeze_json_mapping_sequence(value.feedback)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ResultValidationError("evaluation feedback must be finite canonical JSON") from exc
    frozen = EvaluationOutcome(
        valid=value.valid,
        hard=value.hard,
        soft=value.soft,
        feedback=feedback,
        reference=value.reference,
        attempted_calls=value.attempted_calls,
        completed_calls=value.completed_calls,
        failure_class=value.failure_class,
        failure=value.failure,
    )
    frozen.validate()
    return frozen


@dataclass(frozen=True, slots=True)
class ProposalOutcome:
    """One bounded skill-edit proposal from an optimizer."""

    valid: bool
    candidate_skill: str
    operation: str
    rationale: str
    attempted_calls: int
    completed_calls: int
    failure_class: str | None = None
    failure: str = ""

    def validate(self, *, current_skill: str, max_skill_chars: int) -> None:
        if type(current_skill) is not str:
            raise ResultValidationError("current skill must be a string")
        if type(max_skill_chars) is not int or max_skill_chars < 1:
            raise ResultValidationError("max_skill_chars must be a positive integer")
        validate_call_counts(
            attempted=self.attempted_calls, completed=self.completed_calls, label="optimizer"
        )
        if type(self.valid) is not bool:
            raise ResultValidationError("proposal valid flag must be boolean")
        if self.failure_class is not None and (
            type(self.failure_class) is not str or not self.failure_class.strip()
        ):
            raise ResultValidationError("proposal failure_class must be a non-empty string or None")
        if type(self.failure) is not str:
            raise ResultValidationError("proposal failure must be a string")
        if self.valid and self.failure_class:
            raise ResultValidationError("valid proposal cannot carry a failure_class")
        if not self.valid and (not self.failure_class):
            raise ResultValidationError("invalid proposal requires a failure_class")
        if type(self.operation) is not str or self.operation not in EDIT_OPERATIONS:
            raise ResultValidationError(f"unsupported skill edit operation: {self.operation!r}")
        if type(self.candidate_skill) is not str or not self.candidate_skill.strip():
            raise ResultValidationError("candidate skill must be non-empty")
        if len(self.candidate_skill) > max_skill_chars:
            raise ResultValidationError(
                f"candidate skill exceeds max_skill_chars={max_skill_chars}"
            )
        if type(self.rationale) is not str:
            raise ResultValidationError("proposal rationale must be a string")
        try:
            self.candidate_skill.encode("utf-8")
            current_skill.encode("utf-8")
            self.rationale.encode("utf-8")
            self.failure.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ResultValidationError("proposal text fields must be valid UTF-8") from exc
        unchanged = self.candidate_skill == current_skill
        if self.operation == "noop" and (not unchanged):
            raise ResultValidationError("noop proposal must preserve the current skill exactly")
        if self.operation != "noop" and unchanged:
            raise ResultValidationError("non-noop proposal must change the current skill")

    def public(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "candidate_sha256": skill_hash(self.candidate_skill),
            "operation": self.operation,
            "rationale": self.rationale,
            "attempted_calls": self.attempted_calls,
            "completed_calls": self.completed_calls,
            "failure_class": self.failure_class,
            "failure": self.failure,
        }


def filter_feedback(
    feedback: Sequence[Mapping[str, object]], allowed: Sequence[str]
) -> tuple[Mapping[str, object], ...]:
    allowed_set = set(allowed)
    return tuple(item for item in feedback if str(item.get("category", "")) in allowed_set)


def with_feedback(
    outcome: EvaluationOutcome, feedback: tuple[Mapping[str, object], ...]
) -> EvaluationOutcome:
    return EvaluationOutcome(
        valid=outcome.valid,
        hard=outcome.hard,
        soft=outcome.soft,
        feedback=feedback,
        reference=outcome.reference,
        attempted_calls=outcome.attempted_calls,
        completed_calls=outcome.completed_calls,
        failure_class=outcome.failure_class,
        failure=outcome.failure,
    )


@dataclass(frozen=True, slots=True)
class EvolutionRound:
    round_no: int
    action: str
    operation: str
    parent_sha256: str
    candidate_sha256: str
    current_sha256: str
    best_sha256: str
    candidate_hard: float | None
    candidate_soft: float | None

    def public(self) -> dict[str, object]:
        return {
            "round_no": self.round_no,
            "action": self.action,
            "operation": self.operation,
            "parent_sha256": self.parent_sha256,
            "candidate_sha256": self.candidate_sha256,
            "current_sha256": self.current_sha256,
            "best_sha256": self.best_sha256,
            "candidate_hard": self.candidate_hard,
            "candidate_soft": self.candidate_soft,
        }


@dataclass(frozen=True, slots=True)
class OptimizationContext:
    round_no: int
    current_skill: str
    current_sha256: str
    feedback_categories: tuple[str, ...]
    training: EvaluationOutcome
    history: tuple[EvolutionRound, ...]


OPTIMIZER_EVIDENCE_FILENAME = "PROPOSAL_EVIDENCE.json"

OPTIMIZER_EVIDENCE_SCHEMA_VERSION = 2


def optimizer_artifact_inventory(run_root: Path, *, round_no: int) -> tuple[dict[str, object], ...]:
    """Snapshot the exact portable artifact set for one optimizer round."""
    round_root = run_root / "optimizer" / f"round_{round_no:04d}"
    evidence_path = round_root / OPTIMIZER_EVIDENCE_FILENAME
    return regular_file_inventory(
        round_root, relative_to=run_root, excluded=(evidence_path,), label="optimizer artifacts"
    )


def write_optimizer_round_evidence(
    run_root: Path,
    *,
    round_no: int,
    parent_sha256: str,
    proposal: ProposalOutcome,
) -> dict[str, object]:
    """Record one proposal and bind the files created by its optimizer."""
    round_root = run_root / "optimizer" / f"round_{round_no:04d}"
    round_root.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema_version": OPTIMIZER_EVIDENCE_SCHEMA_VERSION,
        "status": "RETHINKSKILL_OPTIMIZER_EVIDENCE",
        "round_no": round_no,
        "parent_sha256": parent_sha256,
        "proposal": proposal.public(),
        "artifacts": list(optimizer_artifact_inventory(run_root, round_no=round_no)),
    }
    relative_path = Path("optimizer") / f"round_{round_no:04d}" / OPTIMIZER_EVIDENCE_FILENAME
    snapshot = atomic_write_json_snapshot(run_root / relative_path, evidence)
    return {"path": relative_path.as_posix(), "sha256": snapshot.sha256}


@dataclass(frozen=True, slots=True)
class EvolutionInitialization:
    """Exact initial state and manifest binding published for one run."""

    initial_skill: str
    manifest_snapshot: RegularFileSnapshot


@dataclass(frozen=True, slots=True)
class EvolutionFinalization:
    """Complete state required to freeze one evolution lineage."""

    plan: EvolutionPlan
    manifest_snapshot: RegularFileSnapshot
    initial_score: EvaluationOutcome
    current_skill: str
    best_skill: str
    gate_state: GateState
    rounds: tuple[EvolutionRound, ...]
    round_payloads: tuple[Mapping[str, object], ...]
    target_attempted: int
    target_completed: int
    target_call_accounting_known: bool
    optimizer_attempted: int
    optimizer_completed: int
    optimizer_call_accounting_known: bool
    terminal_optimizer_evidence: Mapping[str, object] | None
    failure_class: str | None
    failure: str
    failure_stage: str | None
    failure_round: int | None

    @property
    def complete(self) -> bool:
        return (
            self.failure_class is None
            and self.target_call_accounting_known
            and self.optimizer_call_accounting_known
            and (len(self.rounds) == self.plan.options.rounds)
            and (self.target_attempted == self.target_completed)
            and (self.optimizer_attempted == self.optimizer_completed)
        )


def initialize_evolution_artifacts(
    plan: EvolutionPlan,
    *,
    initial_skill: str,
    evaluator_manifest: Mapping[str, object],
    optimizer_manifest: Mapping[str, object],
) -> EvolutionInitialization:
    """Freeze authorization metadata and the exact initial skill."""
    control = plan.output_root / ".rethinkskill"
    manifest = thaw_json_mapping(
        freeze_manifest(
            {
                **thaw_json_mapping(plan.preflight),
                "status": "RETHINKSKILL_EVOLUTION_AUTHORIZED",
                "authorized_at": utc_now(),
                "evaluator": thaw_json_mapping(evaluator_manifest),
                "optimizer": thaw_json_mapping(optimizer_manifest),
                "evolution_receipt_schema_version": EVOLUTION_RECEIPT_SCHEMA_VERSION,
            }
        )
    )
    manifest_snapshot = atomic_write_json_snapshot(control / "EVOLUTION_MANIFEST.json", manifest)
    (plan.output_root / "rounds").mkdir(parents=True, exist_ok=True)
    atomic_write(plan.output_root / "skills/initial.md", initial_skill.encode("utf-8"))
    return EvolutionInitialization(initial_skill=initial_skill, manifest_snapshot=manifest_snapshot)


def write_candidate_skill(plan: EvolutionPlan, *, round_no: int, candidate_skill: str) -> None:
    """Freeze one optimizer candidate before validation."""
    atomic_write(
        plan.output_root / f"skills/round_{round_no:04d}_candidate.md",
        candidate_skill.encode("utf-8"),
    )


def build_round_payload(
    round_record: EvolutionRound,
    *,
    training: EvaluationOutcome,
    proposal: ProposalOutcome,
    optimizer_evidence: Mapping[str, object],
    validation: EvaluationOutcome | None,
    gate_state: GateState,
) -> dict[str, object]:
    """Serialize one round without changing evolution state."""
    return {
        **round_record.public(),
        "training": training.public(),
        "proposal": proposal.public(),
        "optimizer_evidence": thaw_json_mapping(optimizer_evidence),
        "validation": validation.public() if validation else None,
        "gate_state": {
            "current_hard": gate_state.current_hard,
            "current_soft": gate_state.current_soft,
            "best_hard": gate_state.best_hard,
            "best_soft": gate_state.best_soft,
            "best_round": gate_state.best_round,
            "noop_streak": gate_state.noop_streak,
            "regression_streak": gate_state.regression_streak,
        },
    }


def write_round_payload(
    plan: EvolutionPlan, *, round_no: int, payload: Mapping[str, object]
) -> None:
    """Freeze one round record using its canonical path."""
    atomic_write_json(plan.output_root / f"rounds/round_{round_no:04d}.json", payload)


def finalize_evolution_artifacts(state: EvolutionFinalization) -> dict[str, object]:
    """Freeze terminal skills, canonical ledger, and receipt."""
    plan = state.plan
    atomic_write(plan.output_root / "final_current_skill.md", state.current_skill.encode("utf-8"))
    atomic_write(plan.output_root / "best_skill.md", state.best_skill.encode("utf-8"))
    ledger = b"".join(
        (
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        for row in state.round_payloads
    )
    ledger_path = plan.output_root / "EVOLUTION_LEDGER.jsonl"
    ledger_snapshot = atomic_write_snapshot(ledger_path, ledger)
    receipt = thaw_json_mapping(
        freeze_receipt(
            {
                "schema_version": EVOLUTION_RECEIPT_SCHEMA_VERSION,
                "status": "RETHINKSKILL_EVOLUTION_VALIDATED"
                if state.complete
                else "RETHINKSKILL_EVOLUTION_INVALID",
                "created_at": utc_now(),
                "benchmark": plan.benchmark,
                "arm": plan.options.arm,
                "rounds_requested": plan.options.rounds,
                "rounds_recorded": len(state.rounds),
                "target_calls": {
                    "attempted": state.target_attempted,
                    "completed": state.target_completed,
                    "accounting_known": state.target_call_accounting_known,
                },
                "optimizer_calls": {
                    "attempted": state.optimizer_attempted,
                    "completed": state.optimizer_completed,
                    "accounting_known": state.optimizer_call_accounting_known,
                },
                "terminal_optimizer_evidence": thaw_json_mapping(state.terminal_optimizer_evidence)
                if state.terminal_optimizer_evidence is not None
                else None,
                "initial_score": state.initial_score.public(),
                "final_current": {
                    "sha256": skill_hash(state.current_skill),
                    "hard": state.gate_state.current_hard,
                    "soft": state.gate_state.current_soft,
                },
                "best": {
                    "sha256": skill_hash(state.best_skill),
                    "hard": state.gate_state.best_hard,
                    "soft": state.gate_state.best_soft,
                    "round": state.gate_state.best_round,
                },
                "ledger": {
                    "path": "EVOLUTION_LEDGER.jsonl",
                    "sha256": ledger_snapshot.sha256,
                    "rows": len(state.round_payloads),
                },
                "run_manifest": {
                    "path": ".rethinkskill/EVOLUTION_MANIFEST.json",
                    "sha256": state.manifest_snapshot.sha256,
                },
                "failure_class": state.failure_class,
                "failure": state.failure,
                "failure_stage": state.failure_stage,
                "failure_round": state.failure_round,
            }
        )
    )
    atomic_write_json(plan.output_root / ".rethinkskill/EVOLUTION_RECEIPT.json", receipt)
    return receipt


@dataclass(slots=True)
class _ComponentCallState:
    """Accumulate trusted counts without interpreting provider behavior."""

    attempted: int = 0
    completed: int = 0
    accounting_known: bool = True

    def add(self, outcome: EvaluationOutcome | ProposalOutcome, *, accounting_known: bool) -> None:
        if type(outcome) not in {EvaluationOutcome, ProposalOutcome}:
            raise ResultValidationError("evolution accounting requires an exact component outcome")
        if type(accounting_known) is not bool:
            raise ResultValidationError("evolution accounting-known flag must be boolean")
        self.attempted += outcome.attempted_calls
        self.completed += outcome.completed_calls
        self.accounting_known = self.accounting_known and accounting_known


@dataclass(slots=True)
class EvolutionExecutionState:
    """Own mutable run bookkeeping but no gate or acceptance policy."""

    initial_score: EvaluationOutcome
    current_skill: str
    best_skill: str
    gate_state: GateState
    target_calls: _ComponentCallState = field(default_factory=_ComponentCallState)
    optimizer_calls: _ComponentCallState = field(default_factory=_ComponentCallState)
    rounds: list[EvolutionRound] = field(default_factory=list)
    round_payloads: list[dict[str, object]] = field(default_factory=list)
    terminal_optimizer_evidence: Mapping[str, object] | None = None
    failure_class: str | None = None
    failure: str = ""
    failure_stage: str | None = None
    failure_round: int | None = None

    @classmethod
    def initialize(
        cls, *, initial_skill: str, initial_score: EvaluationOutcome, accounting_known: bool
    ) -> EvolutionExecutionState:
        state = cls(
            initial_score=initial_score,
            current_skill=initial_skill,
            best_skill=initial_skill,
            gate_state=GateState(
                current_hard=initial_score.hard,
                current_soft=initial_score.soft,
                best_hard=initial_score.hard,
                best_soft=initial_score.soft,
                best_round=0,
            ),
        )
        state.target_calls.add(initial_score, accounting_known=accounting_known)
        if initial_score.failure_class is not None:
            state.fail(
                failure_class=initial_score.failure_class,
                failure=initial_score.failure,
                stage="initial_validation",
                round_no=0,
            )
        return state

    @property
    def failed(self) -> bool:
        return self.failure_class is not None

    def fail(self, *, failure_class: str | None, failure: str, stage: str, round_no: int) -> None:
        if self.failed:
            raise ResultValidationError("evolution failure location is already recorded")
        if type(failure_class) is not str or not failure_class.strip():
            raise ResultValidationError("evolution failure requires a non-empty failure class")
        if type(failure) is not str:
            raise ResultValidationError("evolution failure detail must be a string")
        if type(stage) is not str or stage not in {
            "initial_validation",
            "training",
            "optimization",
            "candidate_validation",
        }:
            raise ResultValidationError(f"invalid evolution failure stage: {stage!r}")
        if type(round_no) is not int or round_no < 0:
            raise ResultValidationError("evolution failure round must be a non-negative integer")
        self.failure_class = failure_class
        self.failure = failure
        self.failure_stage = stage
        self.failure_round = round_no

    def add_round(self, round_record: EvolutionRound, payload: dict[str, object]) -> None:
        self.rounds.append(round_record)
        self.round_payloads.append(payload)


class SkillEvaluator(Protocol):
    """Evaluates one immutable skill on one named frozen split."""

    def public_manifest(self) -> Mapping[str, object]: ...

    def evaluate(self, skill: str, *, split: str, round_no: int) -> EvaluationOutcome: ...


class SkillOptimizer(Protocol):
    """Produces one candidate without owning benchmark evaluation."""

    def public_manifest(self) -> Mapping[str, object]: ...

    def propose(self, context: OptimizationContext) -> ProposalOutcome: ...


def _evaluation_boundary(
    evaluator: SkillEvaluator, skill: str, *, split: str, round_no: int, stage: str
) -> tuple[EvaluationOutcome, bool]:
    """Contain one evaluator call and report whether accounting is trustworthy."""
    try:
        outcome = freeze_evaluation_outcome(
            evaluator.evaluate(skill, split=split, round_no=round_no)
        )
        return (outcome, True)
    except Exception as exc:
        return (
            EvaluationOutcome(
                valid=False,
                hard=0.0,
                soft=0.0,
                feedback=(),
                reference=f"boundary://{stage}/{round_no}/{split}",
                attempted_calls=0,
                completed_calls=0,
                failure_class="native_evaluation_invalid",
                failure=boundary_failure(f"{stage} evaluator boundary", exc),
            ),
            False,
        )


def _optimizer_boundary(
    optimizer: SkillOptimizer, context: OptimizationContext, *, max_skill_chars: int
) -> tuple[ProposalOutcome, bool]:
    """Contain one optimizer call and report whether accounting is trustworthy."""
    try:
        proposal = optimizer.propose(context)
        if type(proposal) is not ProposalOutcome:
            raise TypeError("optimizer must return an exact ProposalOutcome")
        proposal.validate(current_skill=context.current_skill, max_skill_chars=max_skill_chars)
        return (proposal, True)
    except Exception as exc:
        return (
            ProposalOutcome(
                valid=False,
                candidate_skill=context.current_skill,
                operation="noop",
                rationale="optimizer boundary failure",
                attempted_calls=0,
                completed_calls=0,
                failure_class="optimizer_invalid",
                failure=boundary_failure("optimizer boundary", exc),
            ),
            False,
        )


FailureStage = Callable[[str], AbstractContextManager[None]]


def execute_evolution_round(
    plan: EvolutionPlan,
    *,
    state: EvolutionExecutionState,
    evaluator: SkillEvaluator,
    optimizer: SkillOptimizer,
    round_no: int,
    failure_stage: FailureStage,
) -> None:
    """Execute one frozen round and mutate only its core-owned run state."""
    with failure_stage("training"):
        training, training_accounting_known = _evaluation_boundary(
            evaluator,
            state.current_skill,
            split=plan.options.train_split,
            round_no=round_no,
            stage="training",
        )
        state.target_calls.add(training, accounting_known=training_accounting_known)
        if not training.valid:
            state.fail(
                failure_class=training.failure_class,
                failure=training.failure,
                stage="training",
                round_no=round_no,
            )
            return
        filtered = filter_feedback(training.feedback, plan.options.feedback_categories)
        training_for_optimizer = with_feedback(training, filtered)
    with failure_stage("optimization"):
        proposal, proposal_accounting_known = _optimizer_boundary(
            optimizer,
            OptimizationContext(
                round_no=round_no,
                current_skill=state.current_skill,
                current_sha256=skill_hash(state.current_skill),
                feedback_categories=plan.options.feedback_categories,
                training=training_for_optimizer,
                history=tuple(state.rounds),
            ),
            max_skill_chars=plan.options.max_skill_chars,
        )
        state.optimizer_calls.add(proposal, accounting_known=proposal_accounting_known)
        parent_hash = skill_hash(state.current_skill)
        optimizer_evidence = write_optimizer_round_evidence(
            plan.output_root,
            round_no=round_no,
            parent_sha256=parent_hash,
            proposal=proposal,
        )
        if not proposal_accounting_known:
            state.terminal_optimizer_evidence = optimizer_evidence
            state.fail(
                failure_class=proposal.failure_class,
                failure=proposal.failure,
                stage="optimization",
                round_no=round_no,
            )
            return
        candidate_hash = skill_hash(proposal.candidate_skill)
        write_candidate_skill(plan, round_no=round_no, candidate_skill=proposal.candidate_skill)
    candidate_score: EvaluationOutcome | None = None
    if not proposal.valid:
        state.fail(
            failure_class=proposal.failure_class,
            failure=proposal.failure,
            stage="optimization",
            round_no=round_no,
        )
        action = "infrastructure_invalid"
    elif proposal.operation == "noop":
        action = "flat"
        with failure_stage("optimization"):
            decision = gate_decision(
                plan.gate,
                state.gate_state,
                hard=state.gate_state.current_hard,
                soft=state.gate_state.current_soft,
                round_no=round_no,
            )
            state.gate_state = decision.state
    else:
        with failure_stage("candidate_validation"):
            candidate_score, candidate_accounting_known = _evaluation_boundary(
                evaluator,
                proposal.candidate_skill,
                split=plan.options.validation_split,
                round_no=round_no,
                stage="candidate_validation",
            )
            state.target_calls.add(candidate_score, accounting_known=candidate_accounting_known)
            if not candidate_accounting_known:
                state.fail(
                    failure_class=candidate_score.failure_class,
                    failure=candidate_score.failure,
                    stage="candidate_validation",
                    round_no=round_no,
                )
                return
            if not candidate_score.valid:
                state.fail(
                    failure_class=candidate_score.failure_class,
                    failure=candidate_score.failure,
                    stage="candidate_validation",
                    round_no=round_no,
                )
                action = "infrastructure_invalid"
            else:
                decision = gate_decision(
                    plan.gate,
                    state.gate_state,
                    hard=candidate_score.hard,
                    soft=candidate_score.soft,
                    round_no=round_no,
                )
                action = decision.action
                state.gate_state = decision.state
                if action in {"accept", "accept_new_best"}:
                    state.current_skill = proposal.candidate_skill
                if action == "accept_new_best":
                    state.best_skill = proposal.candidate_skill
    with failure_stage("round_publication"):
        round_record = EvolutionRound(
            round_no=round_no,
            action=action,
            operation=proposal.operation,
            parent_sha256=parent_hash,
            candidate_sha256=candidate_hash,
            current_sha256=skill_hash(state.current_skill),
            best_sha256=skill_hash(state.best_skill),
            candidate_hard=candidate_score.hard if candidate_score else None,
            candidate_soft=candidate_score.soft if candidate_score else None,
        )
        round_payload = build_round_payload(
            round_record,
            training=training_for_optimizer,
            proposal=proposal,
            optimizer_evidence=optimizer_evidence,
            validation=candidate_score,
            gate_state=state.gate_state,
        )
        state.add_round(round_record, round_payload)
        write_round_payload(plan, round_no=round_no, payload=round_payload)


def _terminal_calls(
    state: EvolutionExecutionState | None,
    *,
    initial_score: EvaluationOutcome | None,
    initial_accounting_known: bool,
) -> dict[str, TerminalCallAccounting]:
    if state is None:
        target = TerminalCallAccounting(
            attempted=initial_score.attempted_calls if initial_score is not None else 0,
            completed=initial_score.completed_calls if initial_score is not None else 0,
            accounting_known=initial_accounting_known,
        )
        optimizer = TerminalCallAccounting(0, 0, True)
    else:
        target = TerminalCallAccounting(
            attempted=state.target_calls.attempted,
            completed=state.target_calls.completed,
            accounting_known=state.target_calls.accounting_known,
        )
        optimizer = TerminalCallAccounting(
            attempted=state.optimizer_calls.attempted,
            completed=state.optimizer_calls.completed,
            accounting_known=state.optimizer_calls.accounting_known,
        )
    return {"target": target, "optimizer": optimizer}


def execute_evolution(
    plan: EvolutionPlan, *, evaluator: SkillEvaluator, optimizer: SkillOptimizer, authorized: bool
) -> dict[str, object]:
    """Execute exactly one bounded evolution lineage without retries."""
    require_model_call_authorization(authorized, action="skill evolution")
    validate_evolution_plan_freeze(plan)
    evaluator_manifest = freeze_component_manifest(evaluator, label="evolution evaluator")
    optimizer_manifest = freeze_component_manifest(optimizer, label="evolution optimizer")
    with OutputLock(plan.output_root):
        initial_skill = validate_evolution_plan_freeze(plan)
        initialize_fresh_output_root(plan.output_root, label="evolution")
        state: EvolutionExecutionState | None = None
        initial_score: EvaluationOutcome | None = None
        initial_accounting_known = True

        def current_calls() -> dict[str, TerminalCallAccounting]:
            return _terminal_calls(
                state,
                initial_score=initial_score,
                initial_accounting_known=initial_accounting_known,
            )

        def failure_stage(phase: str):
            return terminal_failure_boundary(
                plan.output_root, run_kind="evolution", phase=phase, calls=current_calls
            )

        with failure_stage("artifact_initialization"):
            initialization = initialize_evolution_artifacts(
                plan,
                initial_skill=initial_skill,
                evaluator_manifest=evaluator_manifest,
                optimizer_manifest=optimizer_manifest,
            )
        initial_skill = initialization.initial_skill
        with failure_stage("initial_validation"):
            initial_score, initial_accounting_known = _evaluation_boundary(
                evaluator,
                initial_skill,
                split=plan.options.validation_split,
                round_no=0,
                stage="initial_validation",
            )
            state = EvolutionExecutionState.initialize(
                initial_skill=initial_skill,
                initial_score=initial_score,
                accounting_known=initial_accounting_known,
            )
        assert state is not None
        for round_no in range(1, plan.options.rounds + 1):
            if state.failed:
                break
            execute_evolution_round(
                plan,
                state=state,
                evaluator=evaluator,
                optimizer=optimizer,
                round_no=round_no,
                failure_stage=failure_stage,
            )
        with failure_stage("finalization"):
            receipt = finalize_evolution_artifacts(
                EvolutionFinalization(
                    plan=plan,
                    manifest_snapshot=initialization.manifest_snapshot,
                    initial_score=state.initial_score,
                    current_skill=state.current_skill,
                    best_skill=state.best_skill,
                    gate_state=state.gate_state,
                    rounds=tuple(state.rounds),
                    round_payloads=tuple(state.round_payloads),
                    target_attempted=state.target_calls.attempted,
                    target_completed=state.target_calls.completed,
                    target_call_accounting_known=state.target_calls.accounting_known,
                    optimizer_attempted=state.optimizer_calls.attempted,
                    optimizer_completed=state.optimizer_calls.completed,
                    optimizer_call_accounting_known=state.optimizer_calls.accounting_known,
                    terminal_optimizer_evidence=state.terminal_optimizer_evidence,
                    failure_class=state.failure_class,
                    failure=state.failure,
                    failure_stage=state.failure_stage,
                    failure_round=state.failure_round,
                )
            )
    return receipt
