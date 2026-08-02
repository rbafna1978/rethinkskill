"""RethinkSkill runtime runner."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from rethinkskill.benchmarks.capabilities import BenchmarkCapability
from rethinkskill.benchmarks.core import BenchmarkSpec, EvaluationMode
from rethinkskill.benchmarks.harness_catalog import EvaluationAuthority, NativeHarnessSpec
from rethinkskill.benchmarks.scoring import Verdict, Verification, freeze_verification
from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.evaluation.results import validate_ids, validate_ledger_snapshot
from rethinkskill.integrations.catalog import (
    AdapterDelivery,
    ExecutionFidelity,
    IntegrationSpec,
    IntegrationTier,
)
from rethinkskill.providers.authorization import (
    TerminalCallAccounting,
    initialize_fresh_output_root,
    require_model_call_authorization,
    terminal_failure_boundary,
    validate_frozen_output_scope,
)
from rethinkskill.runtime.tasks import (
    NativeTask,
    RenderedTask,
    bound_dataset_snapshots,
    dataset_tree_files,
    materialize_rendered_workspace,
    materialize_task_assets,
)
from rethinkskill.runtime.types import (
    BenchmarkHarness,
    ModelExecutor,
    ModelOutcome,
    freeze_model_outcome,
)
from rethinkskill.utils.fs import OutputLock, Repository, utc_now
from rethinkskill.utils.integrity import (
    NATIVE_RECEIPT_SCHEMA_VERSION,
    CallCounts,
    bind_evidence_hash,
    bind_manifest_hash,
    bind_receipt_hash,
    boundary_failure,
    exact_call_counts,
    safe_exception_type,
    validate_frozen_plan_fields,
    validate_plan_manifest,
    validate_zero_call_preflight,
)
from rethinkskill.utils.plugins import ExtensionRegistration
from rethinkskill.utils.release import freeze_component_manifest, regular_file_inventory
from rethinkskill.utils.serde import (
    RegularFileSnapshot,
    atomic_write,
    atomic_write_json,
    atomic_write_json_snapshot,
    atomic_write_snapshot,
    canonical_json_bytes,
    freeze_json_mapping,
    sha256_bytes,
    snapshot_regular_file,
    thaw_json_mapping,
    thaw_json_value,
)


@dataclass(frozen=True, slots=True)
class ObservedCallAccounting:
    """Exact returned-call totals plus whether no call escaped observation."""

    attempted: int
    completed: int
    known: bool

    def __post_init__(self) -> None:
        exact_call_counts(self.attempted, self.completed, label="observed native executor")
        if type(self.known) is not bool:
            raise TypeError("observed native accounting known must be boolean")


@dataclass(slots=True)
class ObservedModelExecutor:
    """Validate every returned outcome and count only core-observed calls."""

    _executor: ModelExecutor
    _manifest: Mapping[str, object]
    _attempted: int = field(default=0, init=False, repr=False)
    _completed: int = field(default=0, init=False, repr=False)
    _known: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        self._manifest = freeze_json_mapping(self._manifest)

    def public_manifest(self) -> Mapping[str, object]:
        return thaw_json_mapping(self._manifest)

    def snapshot(self) -> ObservedCallAccounting:
        return ObservedCallAccounting(
            attempted=self._attempted, completed=self._completed, known=self._known
        )

    def delta(self, baseline: ObservedCallAccounting) -> ObservedCallAccounting:
        if type(baseline) is not ObservedCallAccounting:
            raise TypeError("native accounting baseline must be exact ObservedCallAccounting")
        counts: CallCounts = exact_call_counts(
            self._attempted - baseline.attempted,
            self._completed - baseline.completed,
            label="observed native task",
        )
        return ObservedCallAccounting(
            attempted=counts.attempted,
            completed=counts.completed,
            known=baseline.known and self._known,
        )

    def execute(
        self, rendered: RenderedTask, *, workspace: Path, timeout_seconds: int
    ) -> ModelOutcome:
        try:
            outcome = freeze_model_outcome(
                self._executor.execute(
                    rendered, workspace=workspace, timeout_seconds=timeout_seconds
                )
            )
        except Exception:
            self._known = False
            raise
        self._attempted += outcome.attempted_calls
        self._completed += outcome.completed_calls
        return outcome


_CAPABILITY_KEYS = {
    "name",
    "declared",
    "integration",
    "scoring",
    "scoring_registration",
    "scorer_available",
    "native_execution",
    "native_registration",
    "adapter_available",
    "runtime_ready",
    "evaluation_ready",
    "native_runnable",
}

_SCORING_KEYS = {
    "name",
    "family",
    "domain",
    "evaluation_mode",
    "locally_evaluable",
    "metrics",
    "sources",
    "notes",
}

_INTEGRATION_KEYS = {
    "name",
    "tier",
    "execution_fidelity",
    "benchmarks",
    "description",
    "adapter_delivery",
    "adapter_distribution",
    "optional_dependencies",
    "external_requirements",
}

_NATIVE_KEYS = {
    "name",
    "execution_scope",
    "source",
    "native",
    "adapter_available",
    "runtime_ready",
    "evaluation_authority",
    "task_execution",
}


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ConfigurationError(f"{label} must be non-empty text")
    return value


def _text_list(value: object, *, label: str, nonempty: bool) -> list[str]:
    if (
        type(value) is not list
        or (nonempty and (not value))
        or (not all(type(item) is str and item.strip() for item in value))
        or (len(set(value)) != len(value))
    ):
        raise ConfigurationError(f"{label} must be unique text values")
    return value


def _public_mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ConfigurationError(f"{label} must be a JSON object")
    return value


def validate_public_native_capability(
    value: Mapping[str, object], *, benchmark: str | None = None
) -> Mapping[str, object]:
    """Validate and detach one serialized runnable capability binding."""
    try:
        frozen = freeze_json_mapping(value)
        public = thaw_json_mapping(frozen)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ConfigurationError("native capability must be finite canonical JSON") from exc
    if set(public) != _CAPABILITY_KEYS:
        raise ConfigurationError("native capability has an invalid schema")
    name = _text(public.get("name"), label="native capability name")
    if benchmark is not None and name != benchmark:
        raise ConfigurationError("native capability benchmark does not match")
    for key in (
        "declared",
        "scorer_available",
        "adapter_available",
        "runtime_ready",
        "evaluation_ready",
        "native_runnable",
    ):
        if type(public.get(key)) is not bool:
            raise ConfigurationError(f"native capability {key} must be boolean")
    if public["declared"] is not True:
        raise ConfigurationError("native capability must be declared")
    scoring = _public_mapping(public.get("scoring"), label="native scoring declaration")
    if set(scoring) != _SCORING_KEYS or scoring.get("name") != name:
        raise ConfigurationError("native scoring declaration is inconsistent")
    for key in ("family", "domain"):
        _text(scoring.get(key), label=f"native scoring {key}")
    mode = scoring.get("evaluation_mode")
    if mode not in {value.value for value in EvaluationMode}:
        raise ConfigurationError("native scoring evaluation mode is invalid")
    if type(scoring.get("locally_evaluable")) is not bool:
        raise ConfigurationError("native scoring locally_evaluable must be boolean")
    if scoring["locally_evaluable"] is not (mode == EvaluationMode.CASES.value):
        raise ConfigurationError("native scoring mode flags are inconsistent")
    _text_list(scoring.get("metrics"), label="native scoring metrics", nonempty=True)
    if type(scoring.get("notes")) is not str:
        raise ConfigurationError("native scoring notes must be text")
    sources = scoring.get("sources")
    if type(sources) is not list:
        raise ConfigurationError("native scoring sources must be a list")
    for source in sources:
        source_mapping = _public_mapping(source, label="native scoring source")
        if set(source_mapping) != {"relation", "title", "url"}:
            raise ConfigurationError("native scoring source schema is invalid")
        for key in ("relation", "title", "url"):
            _text(source_mapping.get(key), label=f"native scoring source {key}")
    scoring_registration = ExtensionRegistration.from_public(public.get("scoring_registration"))
    if scoring_registration.name != name or scoring_registration.component_kind != "benchmark":
        raise ConfigurationError("native scoring registration does not match its benchmark")
    native = _public_mapping(public.get("native_execution"), label="native execution declaration")
    if not _NATIVE_KEYS.issubset(native) or set(native) - (_NATIVE_KEYS | {"runtime_dependency"}):
        raise ConfigurationError("native execution declaration has an invalid schema")
    if native.get("name") != name:
        raise ConfigurationError("native execution name does not match")
    for key in ("execution_scope", "source"):
        _text(native.get(key), label=f"native execution {key}")
    for key in ("native", "adapter_available", "runtime_ready"):
        if type(native.get(key)) is not bool:
            raise ConfigurationError(f"native execution {key} must be boolean")
    if native["native"] is not True or native["adapter_available"] is not True:
        raise ConfigurationError("native execution flags are inconsistent")
    authority = native.get("evaluation_authority")
    if authority not in {value.value for value in EvaluationAuthority}:
        raise ConfigurationError("native evaluation authority is invalid")
    if native.get("task_execution") not in {"single_model_call", "harness_managed"}:
        raise ConfigurationError("native task execution mode is invalid")
    if "runtime_dependency" in native:
        _public_mapping(native["runtime_dependency"], label="native runtime dependency")
    native_registration = ExtensionRegistration.from_public(public.get("native_registration"))
    if native_registration.name != name or native_registration.component_kind != "native_harness":
        raise ConfigurationError("native harness registration does not match its benchmark")
    integration = _public_mapping(public.get("integration"), label="native integration declaration")
    if set(integration) != _INTEGRATION_KEYS:
        raise ConfigurationError("native integration declaration has an invalid schema")
    _text(integration.get("name"), label="native integration name")
    if integration.get("tier") not in {value.value for value in IntegrationTier}:
        raise ConfigurationError("native integration tier is invalid")
    if integration.get("execution_fidelity") not in {value.value for value in ExecutionFidelity}:
        raise ConfigurationError("native integration fidelity is invalid")
    if integration.get("adapter_delivery") not in {value.value for value in AdapterDelivery}:
        raise ConfigurationError("native integration delivery is invalid")
    benchmarks = _text_list(
        integration.get("benchmarks"), label="native integration benchmarks", nonempty=True
    )
    if name not in benchmarks:
        raise ConfigurationError("native integration does not own the benchmark")
    _text(integration.get("description"), label="native integration description")
    distribution = integration.get("adapter_distribution")
    if distribution is not None:
        _text(distribution, label="native integration adapter distribution")
    delivery = integration["adapter_delivery"]
    if delivery == AdapterDelivery.DECLARED_ONLY.value:
        if distribution is not None:
            raise ConfigurationError("declared-only native integration cannot name a distribution")
    elif distribution is None:
        raise ConfigurationError("delivered native integration requires a distribution")
    _text_list(
        integration.get("optional_dependencies"),
        label="native integration optional dependencies",
        nonempty=False,
    )
    _text_list(
        integration.get("external_requirements"),
        label="native integration external requirements",
        nonempty=False,
    )
    scorer_available = scoring["locally_evaluable"]
    runtime_ready = native["runtime_ready"]
    evaluation_ready = scorer_available or authority == EvaluationAuthority.HARNESS.value
    if (
        public["scorer_available"] is not scorer_available
        or public["adapter_available"] is not True
        or public["runtime_ready"] is not runtime_ready
        or (public["evaluation_ready"] is not evaluation_ready)
        or (public["native_runnable"] is not (runtime_ready and evaluation_ready))
    ):
        raise ConfigurationError("native capability readiness flags are inconsistent")
    return frozen


@dataclass(frozen=True, slots=True)
class NativeRunOptions:
    split: str
    limit: int | None = None
    ids_file: Path | None = None
    seed: int = 42
    asset_root: Path | None = None
    timeout_seconds: int = 900

    def validate(self) -> None:
        if (
            type(self.split) is not str
            or not self.split.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in self.split)
        ):
            raise ConfigurationError("native split must be non-empty")
        if self.limit is not None and (type(self.limit) is not int or self.limit < 1):
            raise ConfigurationError("native limit must be at least one")
        if self.limit is not None and self.ids_file is not None:
            raise ConfigurationError("native limit and ids_file are mutually exclusive")
        if type(self.timeout_seconds) is not int or self.timeout_seconds < 1:
            raise ConfigurationError("native timeout_seconds must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ConfigurationError("native seed must be non-negative")
        if self.ids_file is not None and (
            not isinstance(self.ids_file, Path) or not self.ids_file.is_file()
        ):
            raise ConfigurationError(f"native ids file is absent: {self.ids_file}")
        if self.asset_root is not None and (
            not isinstance(self.asset_root, Path) or not self.asset_root.is_dir()
        ):
            raise ConfigurationError(f"native asset root is absent: {self.asset_root}")


@dataclass(frozen=True, slots=True)
class NativeRunPlan:
    repository: Repository
    capability: BenchmarkCapability
    dataset: Path
    skill: Path
    output_root: Path
    options: NativeRunOptions
    tasks: tuple[NativeTask, ...]
    rendered_tasks: tuple[RenderedTask, ...]
    preflight: Mapping[str, object]

    @property
    def spec(self) -> NativeHarnessSpec:
        native = self.capability.native
        if native is None:
            raise ConfigurationError("native plan capability has no harness")
        return native

    @property
    def registration(self) -> ExtensionRegistration:
        registration = self.capability.native_registration
        if registration is None:
            raise ConfigurationError("native plan capability has no harness registration")
        return registration


class NativeCatalog(Protocol):
    """Minimum selection interface required by native execution."""

    def resolve_capability(self, name: str) -> BenchmarkCapability: ...


RUNTIME_IDENTITY_FILENAME = "RUNTIME_IDENTITY.json"

RUNTIME_IDENTITY_SCHEMA_VERSION = 1


def runtime_identity_sha256(value: Mapping[str, object]) -> str:
    """Return the canonical digest of one detached runtime identity."""
    return sha256_bytes(canonical_json_bytes(thaw_json_mapping(value)))


def write_runtime_identity_evidence(
    control: Path, *, planned: Mapping[str, object], observed: Mapping[str, object]
) -> tuple[dict[str, object], bool]:
    """Persist pre/post runtime identity comparison and return its reference."""
    planned_value = thaw_json_mapping(planned)
    observed_value = thaw_json_mapping(observed)
    planned_sha256 = runtime_identity_sha256(planned_value)
    observed_sha256 = runtime_identity_sha256(observed_value)
    stable = planned_value == observed_value
    evidence = thaw_json_mapping(
        bind_evidence_hash(
            {
                "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
                "status": "RETHINKSKILL_RUNTIME_IDENTITY_STABLE"
                if stable
                else "RETHINKSKILL_RUNTIME_IDENTITY_DRIFTED",
                "planned": planned_value,
                "observed": observed_value,
                "planned_sha256": planned_sha256,
                "observed_sha256": observed_sha256,
                "stable": stable,
                "model_calls": 0,
            },
            hash_field="evidence_sha256",
            label="runtime identity evidence",
        )
    )
    snapshot = atomic_write_json_snapshot(control / RUNTIME_IDENTITY_FILENAME, evidence)
    return (
        {
            "path": f".rethinkskill/{RUNTIME_IDENTITY_FILENAME}",
            "sha256": snapshot.sha256,
            "stable": stable,
        },
        stable,
    )


def freeze_native_tasks(tasks: tuple[NativeTask, ...]) -> tuple[NativeTask, ...]:
    """Detach task payloads into immutable-plan JSON snapshots."""
    if type(tasks) is not tuple:
        raise ConfigurationError("native harness must return a tuple of NativeTask values")
    if not tasks:
        raise ConfigurationError("native harness must return at least one selected task")
    frozen: list[NativeTask] = []
    for task in tasks:
        if type(task) is not NativeTask:
            raise ConfigurationError("native harness must return exact NativeTask values")
        try:
            payload = freeze_json_mapping(task.payload)
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise ConfigurationError("native task payload must be finite canonical JSON") from exc
        frozen_task = NativeTask(task_id=task.task_id, payload=payload, assets=tuple(task.assets))
        frozen_task.validate()
        frozen.append(frozen_task)
    return tuple(frozen)


def native_task_contract(tasks: tuple[NativeTask, ...]) -> list[dict[str, object]]:
    """Serialize the exact normalized task, payload, and asset contract."""
    if type(tasks) is not tuple:
        raise ConfigurationError("native plan tasks must be a tuple")
    rows: list[dict[str, object]] = []
    observed_ids: set[str] = set()
    observed_workspaces: set[str] = set()
    for task in tasks:
        if type(task) is not NativeTask:
            raise ConfigurationError("native plan must contain exact NativeTask values")
        task.validate()
        if task.task_id in observed_ids:
            raise ConfigurationError(f"native plan contains duplicate task ID: {task.task_id}")
        if task.workspace_name in observed_workspaces:
            raise ConfigurationError(
                f"native plan contains colliding task workspaces: {task.workspace_name}"
            )
        observed_ids.add(task.task_id)
        observed_workspaces.add(task.workspace_name)
        rows.append(
            {
                "task_id": task.task_id,
                "workspace_name": task.workspace_name,
                "payload": thaw_json_mapping(task.payload),
                "assets": [asset.public() for asset in task.assets],
            }
        )
    return rows


def native_task_contract_sha256(tasks: tuple[NativeTask, ...]) -> str:
    """Hash the exact normalized task, workspace, payload, and asset contract."""
    return sha256_bytes(canonical_json_bytes(native_task_contract(tasks)))


def freeze_rendered_tasks(
    harness: BenchmarkHarness, tasks: tuple[NativeTask, ...], skill: str
) -> tuple[RenderedTask, ...]:
    """Render and detach the exact executor inputs during zero-call planning."""
    if type(tasks) is not tuple or type(skill) is not str or (not skill.strip()):
        raise ConfigurationError("native rendered-task inputs are invalid")
    frozen: list[RenderedTask] = []
    for task in tasks:
        try:
            candidate = harness.render(task, skill)
            if type(candidate) is not RenderedTask:
                raise TypeError("harness must return an exact RenderedTask")
            rendered = RenderedTask(
                task_markdown=candidate.task_markdown,
                skill_markdown=candidate.skill_markdown,
                invocation=candidate.invocation,
                attachments=tuple(candidate.attachments),
            )
            rendered.validate_for(task)
        except Exception as exc:
            raise ConfigurationError(
                f"native harness could not render task {task.task_id!r}"
            ) from exc
        frozen.append(rendered)
    return tuple(frozen)


def rendered_task_contract(
    tasks: tuple[NativeTask, ...], rendered_tasks: tuple[RenderedTask, ...]
) -> list[dict[str, object]]:
    """Serialize ordered task files, invocation, and exposed attachments."""
    if (
        type(tasks) is not tuple
        or type(rendered_tasks) is not tuple
        or len(tasks) != len(rendered_tasks)
    ):
        raise ConfigurationError("native rendered-task contract is incomplete")
    rows: list[dict[str, object]] = []
    for task, rendered in zip(tasks, rendered_tasks, strict=True):
        if type(task) is not NativeTask or type(rendered) is not RenderedTask:
            raise ConfigurationError("native rendered-task contract requires exact core values")
        rendered.validate_for(task)
        rows.append(
            {
                "task_id": task.task_id,
                "workspace_name": task.workspace_name,
                "task_markdown": rendered.task_markdown,
                "skill_markdown": rendered.skill_markdown,
                "invocation": rendered.invocation,
                "attachments": [attachment.public() for attachment in rendered.attachments],
            }
        )
    return rows


def rendered_task_contract_sha256(
    tasks: tuple[NativeTask, ...], rendered_tasks: tuple[RenderedTask, ...]
) -> str:
    return sha256_bytes(canonical_json_bytes(rendered_task_contract(tasks, rendered_tasks)))


@dataclass(slots=True)
class NativeRunState:
    """Core-owned progress accumulated across the selected task sequence."""

    rows: list[dict[str, object]] = field(default_factory=list)
    attempted_calls: int = 0
    completed_calls: int = 0
    call_accounting_known: bool = True
    aborted_after_task_id: str | None = None
    abort_failure_class: str | None = None
    task_evidence: dict[str, dict[str, object]] = field(default_factory=dict)

    def terminal_calls(self, executor: ObservedModelExecutor) -> dict[str, TerminalCallAccounting]:
        """Return the exact known lower bound for terminal-failure evidence."""
        accounting = executor.snapshot()
        return {
            "target": TerminalCallAccounting(
                attempted=accounting.attempted,
                completed=accounting.completed,
                accounting_known=accounting.known and self.call_accounting_known,
            )
        }


@dataclass(frozen=True, slots=True)
class NativeTaskExecution:
    outcome: ModelOutcome
    abort_after_task: bool
    call_accounting_known: bool


def execute_native_task_boundary(
    *,
    harness: BenchmarkHarness,
    task: NativeTask,
    rendered: RenderedTask,
    executor: ObservedModelExecutor,
    workspace: Path,
    verifier_workspace: Path,
    timeout_seconds: int,
) -> NativeTaskExecution:
    """Run one harness and reconcile its report with observed outcomes."""
    baseline = executor.snapshot()
    custom_executor = getattr(harness, "execute_task", None)
    try:
        candidate = (
            custom_executor(
                task,
                rendered,
                executor,
                workspace=workspace,
                verifier_workspace=verifier_workspace,
                timeout_seconds=timeout_seconds,
            )
            if callable(custom_executor)
            else executor.execute(rendered, workspace=workspace, timeout_seconds=timeout_seconds)
        )
        outcome = freeze_model_outcome(candidate)
    except Exception as exc:
        observed = executor.delta(baseline)
        return NativeTaskExecution(
            outcome=ModelOutcome(
                status="FAILED",
                response="",
                raw="",
                process={
                    "executor_boundary": {
                        "exception_type": safe_exception_type(exc),
                        "call_accounting": "exact" if observed.known else "unknown",
                        "known_attempted_calls": observed.attempted,
                        "known_completed_calls": observed.completed,
                    }
                },
                attempted_calls=observed.attempted,
                completed_calls=observed.completed,
                failure_class="unknown_agent_invalid",
                failure=boundary_failure("executor boundary", exc),
            ),
            abort_after_task=True,
            call_accounting_known=observed.known,
        )
    observed = executor.delta(baseline)
    if not observed.known:
        return NativeTaskExecution(
            outcome=ModelOutcome(
                status="FAILED",
                response="",
                raw=outcome.raw,
                process={
                    **thaw_json_mapping(outcome.process),
                    "executor_boundary": {
                        "call_accounting": "unknown",
                        "known_attempted_calls": observed.attempted,
                        "known_completed_calls": observed.completed,
                    },
                },
                attempted_calls=observed.attempted,
                completed_calls=observed.completed,
                failure_class="unknown_agent_invalid",
                failure="native executor call accounting became unknown inside the benchmark harness",
            ),
            abort_after_task=True,
            call_accounting_known=False,
        )
    if (
        outcome.attempted_calls != observed.attempted
        or outcome.completed_calls != observed.completed
    ):
        return NativeTaskExecution(
            outcome=ModelOutcome(
                status="FAILED",
                response="",
                raw=outcome.raw,
                process={
                    **thaw_json_mapping(outcome.process),
                    "harness_boundary": {
                        "call_accounting": "mismatch",
                        "reported_attempted_calls": outcome.attempted_calls,
                        "reported_completed_calls": outcome.completed_calls,
                        "observed_attempted_calls": observed.attempted,
                        "observed_completed_calls": observed.completed,
                    },
                },
                attempted_calls=observed.attempted,
                completed_calls=observed.completed,
                failure_class="harness_call_accounting_mismatch",
                failure="native harness call accounting does not match core-observed executor outcomes",
            ),
            abort_after_task=True,
            call_accounting_known=True,
        )
    return NativeTaskExecution(outcome=outcome, abort_after_task=False, call_accounting_known=True)


def validate_native_capability(
    capability: BenchmarkCapability, *, benchmark: str
) -> BenchmarkCapability:
    """Validate one exact joined capability selected for native execution."""
    if type(capability) is not BenchmarkCapability:
        raise ConfigurationError("native catalog must return an exact BenchmarkCapability")
    if type(capability.scoring) is not BenchmarkSpec:
        raise ConfigurationError("native capability requires an exact BenchmarkSpec")
    if type(capability.native) is not NativeHarnessSpec:
        raise ConfigurationError("native capability requires an exact NativeHarnessSpec")
    if type(capability.integration) is not IntegrationSpec:
        raise ConfigurationError("native capability requires an exact IntegrationSpec")
    if type(capability.scoring_registration) is not ExtensionRegistration:
        raise ConfigurationError("native capability requires exact scoring registration provenance")
    if type(capability.native_registration) is not ExtensionRegistration:
        raise ConfigurationError("native capability requires exact harness registration provenance")
    capability.scoring.validate()
    capability.native.validate()
    capability.integration.validate()
    capability.scoring_registration.validate()
    capability.native_registration.validate()
    validate_public_native_capability(capability.public(), benchmark=benchmark)
    return capability


def failure_phase(failure_class: str) -> str:
    """Map structured failure classes to a stable execution phase."""
    if failure_class.startswith("environment_"):
        return "environment"
    if failure_class.startswith("verifier_") or failure_class in {"native_evaluation_invalid"}:
        return "verification"
    if failure_class.startswith("optimizer_"):
        return "optimization"
    if failure_class.startswith("artifact_") or failure_class in {
        "generated_code_isolation_violation",
        "missing_artifact",
        "workspace_conflict",
    }:
        return "artifact"
    return "target_execution"


def verification_result_fields(value: Verification) -> dict[str, object]:
    """Shape one frozen verification into canonical persisted score fields."""
    verification = freeze_verification(value)
    metrics = dict(verification.metrics or {})
    soft = next(
        (
            float(metrics[name])
            for name in (
                "f1",
                "case_f1",
                "case_fraction",
                "anls",
                "success",
                "exact_match",
                "accuracy",
                "substring",
            )
            if name in metrics
        ),
        float(verification.verdict is Verdict.PASS),
    )
    return {
        "hard": int(verification.verdict is Verdict.PASS),
        "soft": soft,
        "verdict": verification.verdict.value,
        "reason": verification.reason,
        "answer": verification.answer,
        "metrics": metrics,
        "fail_reason": ""
        if verification.verdict is Verdict.PASS
        else f"deterministic verifier: {verification.reason}",
    }


def result_payload(
    plan: NativeRunPlan,
    task: NativeTask,
    *,
    outcome_status: str,
    response: str,
    attempted_calls: int,
    completed_calls: int,
    failure_class: str | None,
    failure: str,
    process: Mapping[str, object],
    workspace: Path,
    verifier_workspace: Path,
) -> dict[str, object]:
    base = {
        "id": task.task_id,
        "case_id": task.task_id,
        "response": response,
        "agent_ok": outcome_status == "COMPLETED" and (not failure_class),
        "attempted_calls": attempted_calls,
        "completed_calls": completed_calls,
        "execution": thaw_json_mapping(process),
    }
    if failure_class:
        return {
            **base,
            "hard": 0,
            "soft": 0.0,
            "failure_class": failure_class,
            "phase": failure_phase(failure_class),
            "fail_reason": failure,
        }
    workspace_evaluator = getattr(plan.spec.harness, "evaluate_workspace", None)
    verification = (
        workspace_evaluator(
            task, response, workspace=workspace, verifier_workspace=verifier_workspace
        )
        if callable(workspace_evaluator)
        else plan.spec.harness.evaluate(task, response)
    )
    score_fields = verification_result_fields(verification)
    gold = {key: value for key, value in task.payload.items() if key.startswith("gold_")}
    question = task.payload.get("question", "")
    if type(question) is not str:
        raise ResultValidationError("native result question must be text when supplied")
    return {**base, **score_fields, "gold": thaw_json_value(gold), "question": question}


NATIVE_TASK_EVIDENCE_FILENAME = "TASK_EVIDENCE.json"

NATIVE_TASK_EVIDENCE_SCHEMA_VERSION = 1


def native_task_artifact_inventory(
    run_root: Path, *, workspace_name: str
) -> tuple[dict[str, object], ...]:
    task_root = run_root / "tasks" / workspace_name
    evidence_path = task_root / NATIVE_TASK_EVIDENCE_FILENAME
    return regular_file_inventory(
        task_root, relative_to=run_root, excluded=(evidence_path,), label="native task artifacts"
    )


def write_native_task_evidence(
    run_root: Path,
    *,
    task: NativeTask,
    rendered: RenderedTask,
    result: Mapping[str, object],
    executor_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Freeze the task contract, final row, and exact task artifact set."""
    task_record = native_task_contract((task,))[0]
    rendered_record = rendered_task_contract((task,), (rendered,))[0]
    evidence = thaw_json_mapping(
        bind_evidence_hash(
            {
                "schema_version": NATIVE_TASK_EVIDENCE_SCHEMA_VERSION,
                "status": "RETHINKSKILL_NATIVE_TASK_EVIDENCE",
                "task_id": task.task_id,
                "workspace_name": task.workspace_name,
                "task_contract_sha256": sha256_bytes(canonical_json_bytes(task_record)),
                "rendered_contract_sha256": sha256_bytes(canonical_json_bytes(rendered_record)),
                "executor_manifest_sha256": sha256_bytes(
                    canonical_json_bytes(thaw_json_mapping(executor_manifest))
                ),
                "result": thaw_json_mapping(result),
                "artifacts": list(
                    native_task_artifact_inventory(run_root, workspace_name=task.workspace_name)
                ),
            },
            hash_field="evidence_sha256",
            label="native task evidence",
        )
    )
    relative_path = Path("tasks") / task.workspace_name / NATIVE_TASK_EVIDENCE_FILENAME
    snapshot = atomic_write_json_snapshot(run_root / relative_path, evidence)
    return {"path": relative_path.as_posix(), "sha256": snapshot.sha256}


def finalize_native_results(
    plan: NativeRunPlan,
    *,
    state: NativeRunState,
    run_manifest_snapshot: RegularFileSnapshot,
    executor_manifest: Mapping[str, object],
    runtime_identity: Mapping[str, object],
    runtime_identity_stable: bool,
) -> dict[str, object]:
    """Publish the exact ledger bytes and assemble a self-hashed receipt."""
    ledger_payload = b"".join(
        (
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        for row in state.rows
    )
    results_path = plan.output_root / "results.jsonl"
    results_snapshot = atomic_write_snapshot(results_path, ledger_payload)
    validation = validate_ledger_snapshot(
        results_snapshot.payload,
        path=results_path,
        expected_ids=(task.task_id for task in plan.tasks),
    )
    execution_failure_ids = tuple(
        sorted(
            str(row["id"])
            for row in state.rows
            if row.get("failure_class") or row.get("agent_ok") is not True
        )
    )
    valid = (
        validation.status == "STRUCTURALLY_VALID"
        and state.call_accounting_known
        and (state.attempted_calls >= len(plan.tasks))
        and (state.completed_calls == state.attempted_calls)
        and (not execution_failure_ids)
        and runtime_identity_stable
    )
    return thaw_json_mapping(
        bind_receipt_hash(
            {
                "schema_version": NATIVE_RECEIPT_SCHEMA_VERSION,
                "status": "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED"
                if valid
                else "RETHINKSKILL_NATIVE_EXECUTION_INVALID",
                "created_at": utc_now(),
                "benchmark": plan.spec.name,
                "selected_tasks": len(plan.tasks),
                "executed_tasks": len(state.rows),
                "attempted_calls": state.attempted_calls,
                "completed_calls": state.completed_calls,
                "call_accounting_known": state.call_accounting_known,
                "aborted_after_task_id": state.aborted_after_task_id,
                "abort_failure_class": state.abort_failure_class,
                "execution_failure_ids": list(execution_failure_ids),
                "task_evidence": state.task_evidence,
                "results": {"path": "results.jsonl", "sha256": results_snapshot.sha256},
                "run_manifest": {
                    "path": ".rethinkskill/RUN_MANIFEST.json",
                    "sha256": run_manifest_snapshot.sha256,
                },
                "ledger_validation": validation.to_dict(),
                "executor": executor_manifest,
                "runtime_identity": thaw_json_mapping(runtime_identity),
            }
        )
    )


def resolve_native_capability(catalog: NativeCatalog, benchmark: str) -> BenchmarkCapability:
    """Resolve the only selection object accepted by native planning."""
    resolver = getattr(catalog, "resolve_capability", None)
    if not callable(resolver):
        raise ConfigurationError("native planning requires a unified capability catalog")
    capability = validate_native_capability(resolver(benchmark), benchmark=benchmark)
    if not capability.native_runnable:
        raise ConfigurationError(
            f"benchmark is not runtime-ready for native execution: {benchmark}"
        )
    assert capability.scoring_registration is not None
    assert capability.native_registration is not None
    capability.scoring_registration.require_current_distribution_identity()
    capability.native_registration.require_current_distribution_identity()
    return capability


def native_runtime_identity(
    capability: BenchmarkCapability, *, observed: bool = False
) -> dict[str, object]:
    """Combine harness runtime state with selected plugin package identities."""
    if type(observed) is not bool:
        raise ConfigurationError("runtime identity observation flag must be boolean")
    capability = validate_native_capability(capability, benchmark=capability.name)
    assert capability.native is not None
    assert capability.scoring_registration is not None
    assert capability.native_registration is not None
    base = capability.native.runtime_identity()
    sources: dict[str, object] = {}
    for label, registration in (
        ("scoring", capability.scoring_registration),
        ("harness", capability.native_registration),
    ):
        if registration.source != "entry_point":
            continue
        identity = (
            registration.observe_distribution_identity()
            if observed
            else registration.require_current_distribution_identity()
        )
        sources[label] = identity
    if not sources:
        return base
    return {"benchmark_runtime": base, "extension_distributions": sources}


FailureStage = Callable[[str], AbstractContextManager[None]]


def execute_native_tasks(
    plan: NativeRunPlan,
    *,
    executor: ObservedModelExecutor,
    executor_manifest: Mapping[str, object],
    state: NativeRunState,
    failure_stage: FailureStage,
) -> None:
    """Execute the frozen task sequence and update core-owned run state."""
    for task, rendered in zip(plan.tasks, plan.rendered_tasks, strict=True):
        with failure_stage("task_materialization"):
            rendered.validate_for(task)
            task_root = plan.output_root / "tasks" / task.workspace_name
            workspace = task_root / "workspace"
            verifier_workspace = task_root / "verifier"
            materialize_rendered_workspace(rendered, workspace, skill_name="rethinkskill-target")
            atomic_write(task_root / "invocation.txt", rendered.invocation.encode("utf-8"))
            materialize_task_assets(
                task, workspace=workspace, verifier_workspace=verifier_workspace
            )
        with failure_stage("task_execution"):
            execution = execute_native_task_boundary(
                harness=plan.spec.harness,
                task=task,
                rendered=rendered,
                executor=executor,
                workspace=workspace,
                verifier_workspace=verifier_workspace,
                timeout_seconds=plan.options.timeout_seconds,
            )
        outcome = execution.outcome
        state.call_accounting_known = (
            state.call_accounting_known and execution.call_accounting_known
        )
        state.attempted_calls += outcome.attempted_calls
        state.completed_calls += outcome.completed_calls
        with failure_stage("task_result"):
            atomic_write(task_root / "raw.txt", outcome.raw.encode("utf-8"))
            atomic_write_json(task_root / "EXECUTION.json", outcome.public())
            try:
                row = result_payload(
                    plan,
                    task,
                    outcome_status=outcome.status,
                    response=outcome.response,
                    attempted_calls=outcome.attempted_calls,
                    completed_calls=outcome.completed_calls,
                    failure_class=outcome.failure_class,
                    failure=outcome.failure,
                    process=outcome.process,
                    workspace=workspace,
                    verifier_workspace=verifier_workspace,
                )
            except Exception as exc:
                row = result_payload(
                    plan,
                    task,
                    outcome_status="FAILED",
                    response=outcome.response,
                    attempted_calls=outcome.attempted_calls,
                    completed_calls=outcome.completed_calls,
                    failure_class="verifier_presemantic_failure",
                    failure=boundary_failure("verifier boundary", exc),
                    process={
                        **thaw_json_mapping(outcome.process),
                        "verifier_boundary": {"exception_type": safe_exception_type(exc)},
                    },
                    workspace=workspace,
                    verifier_workspace=verifier_workspace,
                )
            atomic_write_json(task_root / "RESULT.json", row)
        with failure_stage("task_evidence"):
            state.task_evidence[task.task_id] = write_native_task_evidence(
                plan.output_root,
                task=task,
                rendered=rendered,
                result=row,
                executor_manifest=executor_manifest,
            )
        state.rows.append(row)
        if execution.abort_after_task:
            state.aborted_after_task_id = task.task_id
            state.abort_failure_class = str(outcome.failure_class)
            break


def _provenance_files(plan: NativeRunPlan) -> tuple[Path, ...]:
    provenance_files = getattr(plan.spec.harness, "provenance_files", None)
    values = (
        tuple(provenance_files(plan.dataset, split=plan.options.split))
        if callable(provenance_files)
        else dataset_tree_files(plan.dataset)
    )
    return tuple(path.expanduser().absolute() for path in values)


def _dataset_records(paths: tuple[Path, ...]) -> list[dict[str, object]]:
    if not paths:
        raise ConfigurationError("native execution has no dataset provenance files")
    records: list[dict[str, object]] = []
    for path in paths:
        try:
            snapshot = snapshot_regular_file(path)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"native dataset provenance file cannot be frozen: {path}"
            ) from exc
        records.append({"path": str(path), "sha256": snapshot.sha256, "size": snapshot.size})
    return records


def _selection_manifest(plan: NativeRunPlan) -> dict[str, object]:
    ids_file = (
        plan.options.ids_file.expanduser().absolute() if plan.options.ids_file is not None else None
    )
    if ids_file is not None:
        try:
            ids_snapshot = snapshot_regular_file(ids_file)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"native ids file cannot be frozen: {ids_file}") from exc
    else:
        ids_snapshot = None
    return {
        "split": plan.options.split,
        "limit": plan.options.limit,
        "ids_file": str(ids_file) if ids_file else None,
        "ids_file_sha256": ids_snapshot.sha256 if ids_snapshot is not None else None,
        "task_ids": [task.task_id for task in plan.tasks],
        "task_workspaces": {task.task_id: task.workspace_name for task in plan.tasks},
        "task_assets": {
            task.task_id: [asset.public() for asset in task.assets] for task in plan.tasks
        },
        "task_contract_sha256": native_task_contract_sha256(plan.tasks),
        "task_contract": native_task_contract(plan.tasks),
        "rendered_contract_sha256": rendered_task_contract_sha256(plan.tasks, plan.rendered_tasks),
        "rendered_contract": rendered_task_contract(plan.tasks, plan.rendered_tasks),
        "count": len(plan.tasks),
        "seed": plan.options.seed,
        "asset_root": str(plan.options.asset_root.expanduser().resolve())
        if plan.options.asset_root is not None
        else None,
    }


def validate_native_plan_freeze(plan: NativeRunPlan) -> str:
    """Reject plan or input drift before any native output or model call."""
    if type(plan) is not NativeRunPlan:
        raise ConfigurationError("native execution requires an exact NativeRunPlan")
    if type(plan.repository) is not Repository:
        raise ConfigurationError("native plan requires an exact Repository")
    if type(plan.options) is not NativeRunOptions:
        raise ConfigurationError("native plan requires exact NativeRunOptions")
    if type(plan.capability) is not BenchmarkCapability:
        raise ConfigurationError("native plan requires an exact BenchmarkCapability")
    validate_native_capability(plan.capability, benchmark=plan.capability.name)
    plan.options.validate()
    preflight = validate_plan_manifest(plan.preflight, label="native preflight")
    if preflight.get("schema_version") != 9:
        raise ConfigurationError("native preflight schema must be 9")
    run_root = plan.repository.resolve_relative("runs", must_exist=False)
    safe_output = validate_frozen_output_scope(
        repository=plan.repository,
        output_root=plan.output_root,
        run_root=run_root,
        label="native",
        require_empty=True,
    )
    skill_path = plan.skill.expanduser().absolute()
    try:
        skill_snapshot = snapshot_regular_file(skill_path)
        skill_payload = skill_snapshot.payload
        skill_text = skill_payload.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ConfigurationError(
            f"native skill must be a regular, non-symlink UTF-8 file: {skill_path}"
        ) from exc
    if not skill_text.strip():
        raise ConfigurationError("native skill must be non-empty")
    dataset_path = plan.dataset.expanduser().absolute()
    if dataset_path.is_symlink():
        raise ConfigurationError(f"native dataset source must not be a symlink: {dataset_path}")
    if not dataset_path.exists():
        raise ConfigurationError(f"native dataset is absent: {dataset_path}")
    expected_dataset = {
        "path": str(dataset_path),
        "files": _dataset_records(_provenance_files(plan)),
    }
    expected_skill = {"path": str(skill_path), "sha256": skill_snapshot.sha256}
    expected = {
        "benchmark": plan.spec.name,
        "capability": plan.capability.public(),
        "runtime_identity": native_runtime_identity(plan.capability),
        "dataset": expected_dataset,
        "skill": expected_skill,
        "selection": _selection_manifest(plan),
        "output_root": str(safe_output),
    }
    validate_frozen_plan_fields(preflight, expected, label="native")
    rerendered = freeze_rendered_tasks(plan.spec.harness, plan.tasks, skill_text)
    if rendered_task_contract(plan.tasks, plan.rendered_tasks) != rendered_task_contract(
        plan.tasks, rerendered
    ):
        raise ConfigurationError("native frozen plan drifted: rendered_tasks")
    validate_zero_call_preflight(
        preflight,
        label="native preflight",
        expected_status="RETHINKSKILL_NATIVE_PREFLIGHT_PASS",
        call_fields=("model_calls",),
    )
    return skill_text


def _requested_ids(payload: bytes | None, *, path: Path | None) -> tuple[str, ...]:
    if payload is None:
        return ()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"native ids file must be readable UTF-8 text: {path}") from exc
    values = tuple(line.strip() for line in text.splitlines() if line.strip())
    selected = validate_ids(values, label="native ids file")
    if not selected:
        raise ConfigurationError("native ids file must contain at least one task ID")
    return selected


def plan_native_run(
    *,
    repository: Repository,
    catalog: NativeCatalog,
    benchmark: str,
    dataset: Path,
    skill: Path,
    output_root: Path,
    options: NativeRunOptions,
) -> NativeRunPlan:
    if type(repository) is not Repository:
        raise ConfigurationError("native planning requires an exact Repository")
    if type(options) is not NativeRunOptions:
        raise ConfigurationError("native planning requires exact NativeRunOptions")
    options.validate()
    frozen_options = replace(
        options,
        ids_file=options.ids_file.expanduser().absolute() if options.ids_file is not None else None,
        asset_root=options.asset_root.expanduser().resolve()
        if options.asset_root is not None
        else None,
    )
    capability = resolve_native_capability(catalog, benchmark)
    spec = capability.native
    assert spec is not None
    resolved_dataset = dataset.expanduser().absolute()
    if resolved_dataset.is_symlink():
        raise ConfigurationError(f"native dataset source must not be a symlink: {resolved_dataset}")
    if not resolved_dataset.exists():
        raise ConfigurationError(f"native dataset is absent: {resolved_dataset}")
    resolved_skill = skill.expanduser().absolute()
    try:
        skill_snapshot = snapshot_regular_file(resolved_skill)
        skill_payload = skill_snapshot.payload
        skill_text = skill_payload.decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ConfigurationError(
            f"native skill must be a regular, non-symlink UTF-8 file: {resolved_skill}"
        ) from exc
    if not skill_text.strip():
        raise ConfigurationError("native skill must be non-empty")
    run_root = repository.resolve_relative("runs", must_exist=False)
    safe_output = repository.validate_new_output(output_root, run_root)
    provenance_files = getattr(spec.harness, "provenance_files", None)
    dataset_files = tuple(
        path.expanduser().absolute()
        for path in (
            tuple(provenance_files(resolved_dataset, split=frozen_options.split))
            if callable(provenance_files)
            else dataset_tree_files(resolved_dataset)
        )
    )
    if not dataset_files:
        raise ConfigurationError(f"native harness reported no provenance files: {benchmark}")
    if len(dataset_files) != len(set(dataset_files)):
        raise ConfigurationError(f"native harness reported duplicate provenance files: {benchmark}")
    ids_snapshot = None
    if frozen_options.ids_file is not None:
        try:
            ids_snapshot = snapshot_regular_file(frozen_options.ids_file)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"native ids file must be a regular, non-symlink file: {frozen_options.ids_file}"
            ) from exc
    requested_ids = _requested_ids(
        ids_snapshot.payload if ids_snapshot is not None else None, path=frozen_options.ids_file
    )
    with bound_dataset_snapshots(dataset_files) as dataset_snapshots:
        tasks = freeze_native_tasks(
            spec.harness.load_tasks(
                resolved_dataset,
                split=frozen_options.split,
                limit=frozen_options.limit,
                requested_ids=requested_ids,
                seed=frozen_options.seed,
                asset_root=frozen_options.asset_root,
            )
        )
    rendered_tasks = freeze_rendered_tasks(spec.harness, tasks, skill_text)
    runtime_identity = native_runtime_identity(capability)
    ids_file = frozen_options.ids_file if frozen_options.ids_file is not None else None
    preflight = bind_manifest_hash(
        {
            "schema_version": 9,
            "status": "RETHINKSKILL_NATIVE_PREFLIGHT_PASS",
            "created_at": utc_now(),
            "benchmark": benchmark,
            "capability": capability.public(),
            "runtime_identity": runtime_identity,
            "dataset": {
                "path": str(resolved_dataset),
                "files": [
                    {
                        "path": str(path),
                        "sha256": dataset_snapshots[path].sha256,
                        "size": dataset_snapshots[path].size,
                    }
                    for path in dataset_files
                ],
            },
            "skill": {"path": str(resolved_skill), "sha256": skill_snapshot.sha256},
            "selection": {
                "split": frozen_options.split,
                "limit": frozen_options.limit,
                "ids_file": str(ids_file) if ids_file else None,
                "ids_file_sha256": ids_snapshot.sha256 if ids_snapshot is not None else None,
                "task_ids": [task.task_id for task in tasks],
                "task_workspaces": {task.task_id: task.workspace_name for task in tasks},
                "task_assets": {
                    task.task_id: [asset.public() for asset in task.assets] for task in tasks
                },
                "task_contract_sha256": native_task_contract_sha256(tasks),
                "task_contract": native_task_contract(tasks),
                "rendered_contract_sha256": rendered_task_contract_sha256(tasks, rendered_tasks),
                "rendered_contract": rendered_task_contract(tasks, rendered_tasks),
                "count": len(tasks),
                "seed": frozen_options.seed,
                "asset_root": str(frozen_options.asset_root)
                if frozen_options.asset_root is not None
                else None,
            },
            "output_root": str(safe_output),
            "model_calls": 0,
        }
    )
    return NativeRunPlan(
        repository=repository,
        capability=capability,
        dataset=resolved_dataset,
        skill=resolved_skill,
        output_root=safe_output,
        options=frozen_options,
        tasks=tasks,
        rendered_tasks=rendered_tasks,
        preflight=preflight,
    )


def execute_native_plan(
    plan: NativeRunPlan, *, executor: ModelExecutor, authorized: bool
) -> dict[str, object]:
    require_model_call_authorization(authorized, action="native execution")
    validate_native_plan_freeze(plan)
    executor_manifest = freeze_component_manifest(executor, label="native executor")
    observed_executor = ObservedModelExecutor(executor, freeze_json_mapping(executor_manifest))
    with OutputLock(plan.output_root):
        validate_native_plan_freeze(plan)
        initialize_fresh_output_root(plan.output_root, label="native")
        control = plan.output_root / ".rethinkskill"
        run_manifest_path = control / "RUN_MANIFEST.json"
        run_manifest = thaw_json_mapping(
            bind_manifest_hash(
                {
                    **thaw_json_mapping(plan.preflight),
                    "status": "RETHINKSKILL_NATIVE_EXECUTION_AUTHORIZED",
                    "authorized_at": utc_now(),
                    "static_preflight_sha256": plan.preflight["manifest_sha256"],
                    "executor": executor_manifest,
                    "process_receipt_schema_version": NATIVE_RECEIPT_SCHEMA_VERSION,
                }
            )
        )
        run_manifest_snapshot = atomic_write_json_snapshot(run_manifest_path, run_manifest)
        state = NativeRunState()

        def failure_stage(phase: str):
            return terminal_failure_boundary(
                plan.output_root,
                run_kind="native",
                phase=phase,
                calls=lambda: state.terminal_calls(observed_executor),
            )

        execute_native_tasks(
            plan,
            executor=observed_executor,
            executor_manifest=executor_manifest,
            state=state,
            failure_stage=failure_stage,
        )
        with failure_stage("ledger_publication"):
            runtime_identity, runtime_identity_stable = write_runtime_identity_evidence(
                control,
                planned=plan.preflight["runtime_identity"],
                observed=native_runtime_identity(plan.capability, observed=True),
            )
            receipt = finalize_native_results(
                plan,
                state=state,
                run_manifest_snapshot=run_manifest_snapshot,
                executor_manifest=executor_manifest,
                runtime_identity=runtime_identity,
                runtime_identity_stable=runtime_identity_stable,
            )
        with failure_stage("receipt_publication"):
            atomic_write_json(control / "PROCESS_RECEIPT.json", receipt)
    return receipt
