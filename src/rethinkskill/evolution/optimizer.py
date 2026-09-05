"""RethinkSkill evolution optimizer."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from rethinkskill.errors import ConfigurationError
from rethinkskill.evolution.loop import OptimizationContext, ProposalOutcome
from rethinkskill.runtime.tasks import RenderedTask, materialize_rendered_workspace
from rethinkskill.runtime.types import ModelExecutor, ModelOutcomeBoundary, freeze_model_outcome
from rethinkskill.utils.integrity import boundary_failure, exact_call_counts
from rethinkskill.utils.plugins import (
    ExtensionRegistration,
    LoadedPluginSpec,
    SealableCatalog,
    load_plugin_spec_records,
    prepare_plugin_spec_batch,
)
from rethinkskill.utils.release import freeze_component_manifest
from rethinkskill.utils.serde import (
    atomic_write,
    atomic_write_json,
    freeze_json_mapping,
    strict_json_loads,
    thaw_json_mapping,
)


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    """Construction and evidence metadata for one optimizer strategy."""

    name: str
    optimizer_type: type
    description: str
    proposal_contract: str
    executor_invocations_per_proposal: int

    def validate(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name != self.name.lower()
            or (self.name != self.name.strip())
        ):
            raise ConfigurationError(f"optimizer strategy name must be lowercase: {self.name!r}")
        if not isinstance(self.optimizer_type, type):
            raise ConfigurationError(f"optimizer implementation must be a type: {self.name}")
        if (
            type(self.description) is not str
            or not self.description.strip()
            or type(self.proposal_contract) is not str
            or (not self.proposal_contract.strip())
        ):
            raise ConfigurationError(
                f"optimizer description and proposal contract are required: {self.name}"
            )
        if (
            type(self.executor_invocations_per_proposal) is not int
            or self.executor_invocations_per_proposal < 0
        ):
            raise ConfigurationError(
                f"optimizer executor invocation bound must be a non-negative integer: {self.name}"
            )

    def public(self) -> dict[str, object]:
        self.validate()
        return {
            "name": self.name,
            "implementation": f"{self.optimizer_type.__module__}:{self.optimizer_type.__qualname__}",
            "description": self.description,
            "proposal_contract": self.proposal_contract,
            "executor_invocations_per_proposal": self.executor_invocations_per_proposal,
            "owns_evaluation": False,
            "owns_acceptance": False,
            "public_entry_points_supported": True,
        }


@dataclass(frozen=True, slots=True)
class ModelSkillOptimizer:
    """Use any ModelExecutor to propose one strict full-skill candidate."""

    executor: ModelExecutor
    output_root: Path
    timeout_seconds: int = 900

    def public_manifest(self) -> Mapping[str, object]:
        return {
            "kind": "model-skill-optimizer",
            "output_contract": "strict_json_full_skill_candidate",
            "executor": freeze_component_manifest(self.executor, label="optimizer executor"),
        }

    def propose(self, context: OptimizationContext) -> ProposalOutcome:
        rendered = render_model_optimizer_task(context)
        workspace = self.output_root / "optimizer" / f"round_{context.round_no:04d}" / "workspace"
        materialize_rendered_workspace(rendered, workspace, skill_name="rethinkskill-optimizer")
        atomic_write(workspace.parent / "invocation.txt", rendered.invocation.encode("utf-8"))
        outcome = freeze_model_outcome(
            self.executor.execute(
                rendered, workspace=workspace, timeout_seconds=self.timeout_seconds
            )
        )
        atomic_write(workspace.parent / "raw.txt", outcome.raw.encode("utf-8"))
        atomic_write_json(workspace.parent / "EXECUTION.json", outcome.public())
        if not outcome.ok:
            return ProposalOutcome(
                valid=False,
                candidate_skill=context.current_skill,
                operation="noop",
                rationale="",
                attempted_calls=outcome.attempted_calls,
                completed_calls=outcome.completed_calls,
                failure_class=outcome.failure_class or "optimizer_invalid",
                failure=outcome.failure,
            )
        try:
            value = strict_json_loads(outcome.response.strip())
        except json.JSONDecodeError as exc:
            return ProposalOutcome(
                valid=False,
                candidate_skill=context.current_skill,
                operation="noop",
                rationale="",
                attempted_calls=outcome.attempted_calls,
                completed_calls=outcome.completed_calls,
                failure_class="optimizer_response_schema_error",
                failure=f"invalid JSON: {exc}",
            )
        if (
            not isinstance(value, dict)
            or set(value) != {"operation", "candidate_skill", "rationale"}
            or (not all(isinstance(item, str) for item in value.values()))
        ):
            return ProposalOutcome(
                valid=False,
                candidate_skill=context.current_skill,
                operation="noop",
                rationale="",
                attempted_calls=outcome.attempted_calls,
                completed_calls=outcome.completed_calls,
                failure_class="optimizer_response_schema_error",
                failure="proposal must contain exactly three string fields: operation, candidate_skill, rationale",
            )
        return ProposalOutcome(
            valid=True,
            candidate_skill=value["candidate_skill"],
            operation=value["operation"],
            rationale=value["rationale"],
            attempted_calls=outcome.attempted_calls,
            completed_calls=outcome.completed_calls,
        )


def render_model_optimizer_task(context: OptimizationContext) -> RenderedTask:
    """Render the deterministic optimizer input contract for replay."""
    history = [item.public() for item in context.history]
    task_markdown = f'# Objective\n\nImprove the current reusable agent skill using only the supplied training feedback. Do not include case-specific answers.\n\n# Current skill\n\n{context.current_skill}\n\n# Allowed feedback categories\n\n{json.dumps(context.feedback_categories, allow_nan=False)}\n\n# Training feedback\n\n{json.dumps(context.training.public(), ensure_ascii=False, allow_nan=False)}\n\n# Accepted/rejected history\n\n{json.dumps(history, ensure_ascii=False, allow_nan=False)}\n\n# Output contract\n\nReturn one JSON object and no markdown. It must contain exactly three string keys: "operation", "candidate_skill", and "rationale". operation must be add, delete, replace, or noop. A noop must copy the current skill byte-for-byte.'
    skill_markdown = '---\nname: "rethinkskill-optimizer"\ndescription: "Bounded optimizer for one external skill artifact."\n---\n\n# Optimizer policy\n\nPreserve useful general procedures, make the smallest justified edit, avoid benchmark-instance leakage, and obey the strict JSON output contract.'
    rendered = RenderedTask(
        task_markdown=task_markdown,
        skill_markdown=skill_markdown,
        invocation="Read `task.md` and `.agents/skills/rethinkskill-optimizer/SKILL.md`. Return only the strict proposal JSON.",
    )
    rendered.validate()
    return rendered


if TYPE_CHECKING:
    from rethinkskill.evolution.loop import SkillOptimizer
    from rethinkskill.runtime.tasks import RenderedTask
    from rethinkskill.runtime.types import ModelExecutor


@dataclass(frozen=True, slots=True)
class OptimizerCallAccounting:
    """Core-observed executor activity for one proposal invocation."""

    invocations: int
    attempted_calls: int
    completed_calls: int
    known: bool
    violation: str | None

    def __post_init__(self) -> None:
        if type(self.invocations) is not int or self.invocations < 0:
            raise ValueError("optimizer executor invocations must be non-negative")
        exact_call_counts(self.attempted_calls, self.completed_calls, label="optimizer executor")
        if type(self.known) is not bool:
            raise ValueError("optimizer executor accounting-known flag must be boolean")
        if self.violation is not None and (
            type(self.violation) is not str or not self.violation.strip()
        ):
            raise ValueError("optimizer executor violation must be non-empty or None")


class OptimizerExecutorBoundary:
    """Count and bound every executor call made by one optimizer proposal."""

    def __init__(
        self, executor: ModelExecutor, manifest: Mapping[str, object], *, invocation_limit: int
    ) -> None:
        if type(invocation_limit) is not int or invocation_limit < 0:
            raise TypeError("optimizer executor invocation limit must be non-negative")
        frozen_manifest = freeze_json_mapping(manifest)
        self._executor = ModelOutcomeBoundary(executor, frozen_manifest)
        self._manifest = frozen_manifest
        self._invocation_limit = invocation_limit
        self._active = False
        self._invocations = 0
        self._attempted_calls = 0
        self._completed_calls = 0
        self._known = True
        self._violation: str | None = None

    def public_manifest(self) -> Mapping[str, object]:
        return thaw_json_mapping(self._manifest)

    def begin_proposal(self) -> None:
        if self._active:
            raise RuntimeError("optimizer proposal accounting is already active")
        self._active = True
        self._invocations = 0
        self._attempted_calls = 0
        self._completed_calls = 0
        self._known = True
        self._violation = None

    def execute(self, rendered: RenderedTask, *, workspace: Path, timeout_seconds: int):
        if not self._active:
            raise RuntimeError("optimizer executor calls are allowed only during propose")
        if self._invocations >= self._invocation_limit:
            self._violation = "executor_invocation_limit_exceeded"
            raise RuntimeError("optimizer executor invocation limit exceeded")
        self._invocations += 1
        try:
            outcome = self._executor.execute(
                rendered, workspace=workspace, timeout_seconds=timeout_seconds
            )
        except Exception:
            self._known = False
            raise
        self._attempted_calls += outcome.attempted_calls
        self._completed_calls += outcome.completed_calls
        return outcome

    def finish_proposal(self) -> OptimizerCallAccounting:
        if not self._active:
            raise RuntimeError("optimizer proposal accounting is not active")
        self._active = False
        return OptimizerCallAccounting(
            invocations=self._invocations,
            attempted_calls=self._attempted_calls,
            completed_calls=self._completed_calls,
            known=self._known,
            violation=self._violation,
        )


@dataclass(frozen=True, slots=True)
class RegisteredSkillOptimizer:
    """Delegate proposals while overriding provenance with catalog evidence."""

    optimizer: SkillOptimizer
    spec: OptimizerSpec
    registration: ExtensionRegistration
    executor_boundary: OptimizerExecutorBoundary

    def public_manifest(self) -> dict[str, object]:
        return {
            **freeze_component_manifest(self.optimizer, label="optimizer strategy"),
            "optimizer_strategy": self.spec.public(),
            "optimizer_registration": self.registration.public(),
        }

    def propose(self, context: OptimizationContext) -> ProposalOutcome:
        self.executor_boundary.begin_proposal()
        try:
            candidate = self.optimizer.propose(context)
        except Exception as exc:
            accounting = self.executor_boundary.finish_proposal()
            if not accounting.known:
                raise RuntimeError("optimizer executor call accounting is unknown") from exc
            return self._invalid_proposal(
                context,
                accounting,
                failure_class="optimizer_invocation_limit_exceeded"
                if accounting.violation is not None
                else "optimizer_plugin_exception",
                failure="optimizer exceeded its declared executor invocation limit"
                if accounting.violation is not None
                else boundary_failure("optimizer strategy", exc),
            )
        accounting = self.executor_boundary.finish_proposal()
        if not accounting.known:
            raise RuntimeError("optimizer executor call accounting is unknown")
        if accounting.violation is not None:
            return self._invalid_proposal(
                context,
                accounting,
                failure_class="optimizer_invocation_limit_exceeded",
                failure="optimizer exceeded its declared executor invocation limit",
            )
        if type(candidate) is not ProposalOutcome:
            return self._invalid_proposal(
                context,
                accounting,
                failure_class="optimizer_response_contract_error",
                failure="optimizer must return an exact ProposalOutcome",
            )
        if (
            candidate.attempted_calls != accounting.attempted_calls
            or candidate.completed_calls != accounting.completed_calls
        ):
            return self._invalid_proposal(
                context,
                accounting,
                failure_class="optimizer_call_accounting_mismatch",
                failure="optimizer-reported call counts do not match core-observed executor outcomes",
            )
        return ProposalOutcome(
            valid=candidate.valid,
            candidate_skill=candidate.candidate_skill,
            operation=candidate.operation,
            rationale=candidate.rationale,
            attempted_calls=accounting.attempted_calls,
            completed_calls=accounting.completed_calls,
            failure_class=candidate.failure_class,
            failure=candidate.failure,
        )

    @staticmethod
    def _invalid_proposal(
        context: OptimizationContext,
        accounting: OptimizerCallAccounting,
        *,
        failure_class: str,
        failure: str,
    ) -> ProposalOutcome:
        return ProposalOutcome(
            valid=False,
            candidate_skill=context.current_skill,
            operation="noop",
            rationale="",
            attempted_calls=accounting.attempted_calls,
            completed_calls=accounting.completed_calls,
            failure_class=failure_class,
            failure=failure,
        )


if TYPE_CHECKING:
    from rethinkskill.evolution.loop import SkillOptimizer
    from rethinkskill.runtime.types import ModelExecutor


class OptimizerCatalog(SealableCatalog):
    """Index built-in and explicitly loaded optimizer strategies."""

    _catalog_label = "optimizer"

    def __init__(self, specs: Iterable[OptimizerSpec] = (), *, source: str = "direct"):
        self._initialize_catalog_storage()
        for spec in specs:
            self.register(spec, source=source)

    def register(
        self,
        spec: OptimizerSpec,
        *,
        source: str = "direct",
        entry_point_group: str | None = None,
        entry_point_name: str | None = None,
        distribution_name: str | None = None,
        distribution_version: str | None = None,
    ) -> None:
        self._ensure_mutable()
        if type(spec) is not OptimizerSpec:
            raise ConfigurationError("optimizer registration requires an exact OptimizerSpec")
        spec.validate()
        registration = ExtensionRegistration(
            name=spec.name,
            component_kind="optimizer_strategy",
            source=source,
            entry_point_group=entry_point_group,
            entry_point_name=entry_point_name,
            distribution_name=distribution_name,
            distribution_version=distribution_version,
        )
        registration.validate()
        self._commit_validated_entries(
            ((spec.name, spec, registration),), duplicate_label="optimizer strategy"
        )

    def resolve(self, name: str) -> OptimizerSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ConfigurationError(
                f"unknown optimizer strategy {name!r}; expected one of: {', '.join(self.names())}"
            ) from exc

    def registration(self, name: str) -> ExtensionRegistration:
        self.resolve(name)
        return self._registrations[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def load_entry_points(self, *, group: str = "rethinkskill.optimizers") -> tuple[str, ...]:
        """Explicitly load third-party optimizer strategy plugins."""
        self._ensure_mutable()
        return load_optimizer_entry_points(self, group=group)

    def selection_manifest(self, name: str) -> dict[str, object]:
        spec = self.resolve(name)
        return {**spec.public(), "registration": self._registrations[name].public()}

    def build(
        self, name: str, *, executor: ModelExecutor, output_root: Path, timeout_seconds: int
    ) -> SkillOptimizer:
        spec = self.resolve(name)
        registration = self._registrations[name]
        if not isinstance(output_root, Path):
            raise ConfigurationError("optimizer output root must be a pathlib.Path")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ConfigurationError("optimizer timeout_seconds must be a positive integer")
        if not callable(getattr(executor, "public_manifest", None)) or not callable(
            getattr(executor, "execute", None)
        ):
            raise ConfigurationError("optimizer executor contract is incomplete")
        executor_boundary = OptimizerExecutorBoundary(
            executor,
            freeze_component_manifest(executor, label="optimizer executor"),
            invocation_limit=spec.executor_invocations_per_proposal,
        )
        try:
            candidate = spec.optimizer_type(
                executor=executor_boundary, output_root=output_root, timeout_seconds=timeout_seconds
            )
            public_manifest = candidate.public_manifest
            propose = candidate.propose
            if not callable(public_manifest) or not callable(propose):
                raise TypeError("optimizer methods are not callable")
        except Exception as exc:
            raise ConfigurationError(
                f"optimizer strategy construction failed: {spec.name}: {type(exc).__name__}"
            ) from exc
        return cast(
            "SkillOptimizer",
            RegisteredSkillOptimizer(
                optimizer=candidate,
                spec=spec,
                registration=registration,
                executor_boundary=executor_boundary,
            ),
        )

    def manifest(self) -> dict[str, object]:
        strategies = [self.selection_manifest(name) for name in self._specs]
        return {
            "schema_version": 3,
            "status": "RETHINKSKILL_OPTIMIZER_CATALOG",
            "sealed": self._sealed,
            "strategies": strategies,
            "count": len(strategies),
            "public_entry_points_supported": True,
            "model_calls": 0,
        }


def builtin_optimizer_specs() -> tuple[OptimizerSpec, ...]:
    """Return reviewed strategy declarations in stable CLI order."""
    return (
        OptimizerSpec(
            name="model-skill",
            optimizer_type=ModelSkillOptimizer,
            description="Propose one complete reusable skill from frozen training feedback through the selected optimizer provider.",
            proposal_contract="strict_json_full_skill_candidate",
            executor_invocations_per_proposal=1,
        ),
    )


def optimizer_catalog(*, load_plugins: bool = False) -> OptimizerCatalog:
    """Assemble built-ins and optionally explicit installed strategies."""
    catalog = OptimizerCatalog(builtin_optimizer_specs(), source="builtin")
    if load_plugins:
        catalog.load_entry_points()
    return catalog.seal()


def _prepare_optimizer_registration(
    record: LoadedPluginSpec[OptimizerSpec], *, group: str
) -> tuple[str, ExtensionRegistration]:
    spec = record.spec
    spec.validate()
    registration = ExtensionRegistration(
        name=spec.name,
        component_kind="optimizer_strategy",
        source="entry_point",
        entry_point_group=group,
        entry_point_name=record.entry_point_name,
        distribution_name=record.distribution_name,
        distribution_version=record.distribution_version,
    )
    registration.validate()
    return (spec.name, registration)


def load_optimizer_entry_points(
    catalog: OptimizerCatalog, *, group: str = "rethinkskill.optimizers"
) -> tuple[str, ...]:
    """Load a validated optimizer batch only after explicit opt-in."""
    records = load_plugin_spec_records(
        group=group, expected_type=OptimizerSpec, label="optimizer strategy"
    )
    prepared = prepare_plugin_spec_batch(
        records,
        existing_names=catalog.names(),
        duplicate_label="optimizer strategy",
        prepare=lambda record: _prepare_optimizer_registration(record, group=group),
    )
    catalog._commit_validated_entries(prepared, duplicate_label="optimizer strategy")
    return tuple((name for name, _, _ in prepared))
