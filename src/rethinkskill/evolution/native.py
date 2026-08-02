"""RethinkSkill evolution native."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from rethinkskill.errors import ResultValidationError
from rethinkskill.evaluation.results import validate_ids
from rethinkskill.evolution.loop import EvaluationOutcome, freeze_evaluation_outcome
from rethinkskill.runtime.runner import (
    NativeCatalog,
    NativeRunOptions,
    NativeRunPlan,
    execute_native_plan,
    plan_native_run,
    resolve_native_capability,
    validate_public_native_capability,
)
from rethinkskill.runtime.types import ModelExecutor
from rethinkskill.utils.fs import Repository
from rethinkskill.utils.integrity import (
    bind_manifest_hash,
    exact_call_counts,
    validate_manifest_hash,
)
from rethinkskill.utils.release import freeze_component_manifest
from rethinkskill.utils.serde import (
    atomic_write,
    freeze_json_mapping,
    sha256_bytes,
    snapshot_regular_file,
    strict_json_loads,
    thaw_json_mapping,
)


@dataclass(frozen=True, slots=True)
class NativeEvaluationSelection:
    split: str
    limit: int | None = None
    ids_file: Path | None = None
    task_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        if type(self.split) is not str or not self.split.strip():
            raise ResultValidationError("native evolution selection split must be non-empty")
        if self.limit is not None and (type(self.limit) is not int or self.limit < 1):
            raise ResultValidationError("native evolution selection limit must be positive")
        if self.limit is not None and self.ids_file is not None:
            raise ResultValidationError(
                "native evolution selection limit and ids_file are mutually exclusive"
            )
        if self.ids_file is not None and (
            not isinstance(self.ids_file, Path) or not self.ids_file.is_file()
        ):
            raise ResultValidationError(
                f"native evolution selection ids_file is absent: {self.ids_file}"
            )
        if type(self.task_ids) is not tuple:
            raise ResultValidationError("native evolution task_ids must be a tuple")
        validate_ids(self.task_ids, label="native evolution task IDs")


@dataclass(frozen=True, slots=True)
class FrozenNativeEvolutionInputs:
    benchmark: str
    capability: Mapping[str, object]
    dataset: Path
    dataset_files: tuple[tuple[str, str], ...]
    selection_dataset_files: Mapping[str, tuple[tuple[str, str], ...]]
    selection_assets: Mapping[str, Mapping[str, tuple[Mapping[str, str], ...]]]
    selections: Mapping[str, NativeEvaluationSelection]
    seed: int
    asset_root: Path | None
    manifest: Mapping[str, object]


def _score(value: object, *, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (not 0.0 <= value <= 1.0):
        raise ResultValidationError(f"native evaluation {label} must be a finite score in [0, 1]")
    return float(value)


def _exact_nonnegative(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ResultValidationError(f"native evaluation {label} must be a non-negative integer")
    return value


def evaluation_outcome(
    *,
    receipt: Mapping[str, object],
    results_path: Path,
    selected_tasks: int,
    reference_path: str | None = None,
) -> EvaluationOutcome:
    if not isinstance(receipt, Mapping):
        raise ResultValidationError("native evaluation receipt must be a mapping")
    if not isinstance(results_path, Path):
        raise ResultValidationError("native evaluation results path must be a pathlib.Path")
    if type(selected_tasks) is not int or selected_tasks < 1:
        raise ResultValidationError("native evaluation selected_tasks must be a positive integer")
    if reference_path is not None and (
        type(reference_path) is not str or not reference_path or "#" in reference_path
    ):
        raise ResultValidationError(
            "native evaluation reference_path must be non-empty text without a fragment"
        )
    try:
        frozen_receipt = freeze_json_mapping(receipt)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ResultValidationError(
            "native evaluation receipt must be finite canonical JSON"
        ) from exc
    receipt_snapshot = thaw_json_mapping(frozen_receipt)
    try:
        counts = exact_call_counts(
            receipt_snapshot.get("attempted_calls"),
            receipt_snapshot.get("completed_calls"),
            label="native evaluation",
        )
    except ValueError as exc:
        raise ResultValidationError(str(exc)) from exc
    attempted = counts.attempted
    completed = counts.completed
    call_accounting_known = receipt_snapshot.get("call_accounting_known")
    if type(call_accounting_known) is not bool:
        raise ResultValidationError(
            "native evaluation receipt call_accounting_known must be boolean"
        )
    if not call_accounting_known:
        raise ResultValidationError("native evaluation call accounting is unknown")
    receipt_selected = _exact_nonnegative(
        receipt_snapshot.get("selected_tasks"), label="receipt selected_tasks"
    )
    receipt_executed = _exact_nonnegative(
        receipt_snapshot.get("executed_tasks"), label="receipt executed_tasks"
    )
    status = receipt_snapshot.get("status")
    if status not in {
        "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED",
        "RETHINKSKILL_NATIVE_EXECUTION_INVALID",
    }:
        raise ResultValidationError("native evaluation receipt has an invalid terminal status")
    ledger_validation = receipt_snapshot.get("ledger_validation")
    if not isinstance(ledger_validation, Mapping):
        raise ResultValidationError("native evaluation receipt has no ledger validation")
    ledger_status = ledger_validation.get("status")
    if type(ledger_status) is not str or not ledger_status:
        raise ResultValidationError("native evaluation ledger status must be non-empty text")
    try:
        results_payload = results_path.read_bytes()
        results_text = results_payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ResultValidationError(
            "native evaluation ledger must be a readable UTF-8 file"
        ) from exc
    rows: list[dict[str, object]] = []
    scored_rows: list[tuple[dict[str, object], float, float]] = []
    observed_case_ids: set[str] = set()
    for line_number, line in enumerate(results_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = strict_json_loads(line)
        except json.JSONDecodeError as exc:
            raise ResultValidationError(
                f"native evaluation ledger line {line_number} is invalid JSON"
            ) from exc
        if type(row) is not dict:
            raise ResultValidationError(
                f"native evaluation ledger line {line_number} must be a JSON object"
            )
        case_id = row.get("case_id")
        if type(case_id) is not str or not case_id:
            raise ResultValidationError(
                f"native evaluation ledger line {line_number} has no exact case_id"
            )
        if case_id in observed_case_ids:
            raise ResultValidationError(
                f"native evaluation ledger has duplicate case_id: {case_id}"
            )
        observed_case_ids.add(case_id)
        failure_class = row.get("failure_class")
        if failure_class is not None and (
            type(failure_class) is not str or not failure_class.strip()
        ):
            raise ResultValidationError(
                f"native evaluation ledger line {line_number} has an invalid failure_class"
            )
        hard = _score(row.get("hard"), label=f"line {line_number} hard")
        soft = _score(row.get("soft"), label=f"line {line_number} soft")
        rows.append(row)
        scored_rows.append((row, hard, soft))
    valid = (
        status == "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED"
        and ledger_status == "STRUCTURALLY_VALID"
        and (receipt_selected == selected_tasks)
        and (receipt_executed == len(rows))
        and (len(rows) == selected_tasks)
        and (attempted >= selected_tasks)
        and (completed == attempted)
        and (not any(row.get("failure_class") for row in rows))
    )
    hard = (
        sum((hard_score for _, hard_score, _ in scored_rows)) / len(scored_rows)
        if scored_rows
        else 0.0
    )
    soft = (
        sum((soft_score for _, _, soft_score in scored_rows)) / len(scored_rows)
        if scored_rows
        else 0.0
    )
    feedback = tuple(
        (
            {
                "category": "infrastructure"
                if row.get("failure_class")
                else "success"
                if hard_score == 1.0
                else "failure",
                "case_id": row["case_id"],
                "answer": row.get("answer"),
                "reason": row.get("reason"),
                "metrics": row.get("metrics", {}),
                "gold": row.get("gold", {}),
                "fail_reason": row.get("fail_reason", ""),
            }
            for row, hard_score, _ in scored_rows
        )
    )
    failure_class = None
    failure = ""
    if not valid:
        invalid_rows = [row for row in rows if row.get("failure_class")]
        failure_class = (
            invalid_rows[0]["failure_class"] if invalid_rows else "native_evaluation_invalid"
        )
        if invalid_rows:
            detail = invalid_rows[0].get("fail_reason", "")
            failure = detail if type(detail) is str else ""
        else:
            failure = ledger_status
        if not failure:
            failure = "native execution receipt summary is invalid"
    outcome = EvaluationOutcome(
        valid=valid,
        hard=hard,
        soft=soft,
        feedback=feedback,
        reference=f"{reference_path or results_path.resolve()}#sha256={sha256_bytes(results_payload)}",
        attempted_calls=attempted,
        completed_calls=completed,
        failure_class=failure_class,
        failure=failure,
    )
    return freeze_evaluation_outcome(outcome)


def freeze_native_evolution_inputs(
    *,
    repository: Repository,
    catalog: NativeCatalog,
    benchmark: str,
    dataset: Path,
    initial_skill: Path,
    output_root: Path,
    selections: Mapping[str, NativeEvaluationSelection],
    seed: int = 42,
    asset_root: Path | None = None,
    timeout_seconds: int = 900,
) -> FrozenNativeEvolutionInputs:
    """Freeze dataset hashes and disjoint exact task IDs with zero model calls."""
    if type(repository) is not Repository:
        raise ResultValidationError("native evolution freezing requires an exact Repository")
    if not isinstance(selections, Mapping):
        raise ResultValidationError("native evolution selections must be a mapping")
    selection_snapshot = dict(selections)
    if len(selection_snapshot) < 2:
        raise ResultValidationError(
            "native evolution requires at least train and validation selections"
        )
    for role, request in selection_snapshot.items():
        if (
            type(role) is not str
            or re.fullmatch("[A-Za-z0-9][A-Za-z0-9._-]*", role) is None
            or type(request) is not NativeEvaluationSelection
        ):
            raise ResultValidationError(f"invalid native evolution selection role: {role!r}")
        request.validate()
    plans = {}
    frozen_selections = {}
    for role, request in selection_snapshot.items():
        if request.task_ids:
            raise ResultValidationError("selection requests cannot supply pre-frozen task_ids")
        plan = plan_native_run(
            repository=repository,
            catalog=catalog,
            benchmark=benchmark,
            dataset=dataset,
            skill=initial_skill,
            output_root=output_root / ".input_preflight" / role,
            options=NativeRunOptions(
                split=request.split,
                limit=request.limit,
                ids_file=request.ids_file,
                seed=seed,
                asset_root=asset_root,
                timeout_seconds=timeout_seconds,
            ),
        )
        plans[role] = plan
        frozen_selections[role] = NativeEvaluationSelection(
            split=request.split,
            limit=request.limit,
            ids_file=request.ids_file,
            task_ids=tuple(task.task_id for task in plan.tasks),
        )
    capabilities = {
        role: freeze_json_mapping(plan.preflight["capability"]) for role, plan in plans.items()
    }
    capability = next(iter(capabilities.values()))
    if any(value != capability for value in capabilities.values()):
        raise ResultValidationError("native evolution capability drifted during input freeze")
    observed: dict[str, str] = {}
    for role, selection in frozen_selections.items():
        for task_id in selection.task_ids:
            previous = observed.get(task_id)
            if previous is not None:
                raise ResultValidationError(
                    f"native evolution train/validation task IDs overlap: {task_id} in {previous} and {role}"
                )
            observed[task_id] = role
    selection_dataset_files = {
        role: tuple(
            (str(item["path"]), str(item["sha256"])) for item in plan.preflight["dataset"]["files"]
        )
        for role, plan in plans.items()
    }
    combined_files: dict[str, str] = {}
    for files in selection_dataset_files.values():
        for path, digest in files:
            previous = combined_files.get(path)
            if previous is not None and previous != digest:
                raise ResultValidationError(
                    f"native evolution dataset hashes drifted during preflight: {path}"
                )
            combined_files[path] = digest
    files = tuple(sorted(combined_files.items()))
    selection_assets = {
        role: {
            str(task_id): tuple(dict(asset) for asset in assets)
            for task_id, assets in plan.preflight["selection"]["task_assets"].items()
        }
        for role, plan in plans.items()
    }
    manifest = bind_manifest_hash(
        {
            "schema_version": 5,
            "status": "RETHINKSKILL_NATIVE_EVOLUTION_INPUTS_FROZEN",
            "benchmark": benchmark,
            "capability": thaw_json_mapping(capability),
            "dataset": {
                "path": str(dataset.expanduser().resolve()),
                "files": [{"path": path, "sha256": digest} for path, digest in files],
            },
            "selections": {
                role: {
                    "split": selection.split,
                    "limit": selection.limit,
                    "ids_file": str(selection.ids_file.expanduser().absolute())
                    if selection.ids_file
                    else None,
                    "ids_file_sha256": plans[role].preflight["selection"]["ids_file_sha256"]
                    if selection.ids_file
                    else None,
                    "task_ids": list(selection.task_ids),
                    "count": len(selection.task_ids),
                    "dataset_files": [
                        {"path": path, "sha256": digest}
                        for path, digest in selection_dataset_files[role]
                    ],
                    "task_assets": {
                        task_id: [dict(asset) for asset in assets]
                        for task_id, assets in selection_assets[role].items()
                    },
                }
                for role, selection in frozen_selections.items()
            },
            "seed": seed,
            "asset_root": str(asset_root.expanduser().resolve())
            if asset_root is not None
            else None,
            "model_calls": 0,
            "target_calls": 0,
            "optimizer_calls": 0,
        }
    )
    return FrozenNativeEvolutionInputs(
        benchmark=benchmark,
        capability=capability,
        dataset=dataset.expanduser().resolve(),
        dataset_files=files,
        selection_dataset_files=MappingProxyType(dict(selection_dataset_files)),
        selection_assets=MappingProxyType(
            {
                role: MappingProxyType(
                    {
                        task_id: tuple(freeze_json_mapping(asset) for asset in assets)
                        for task_id, assets in role_assets.items()
                    }
                )
                for role, role_assets in selection_assets.items()
            }
        ),
        selections=MappingProxyType(dict(frozen_selections)),
        seed=seed,
        asset_root=asset_root.expanduser().resolve() if asset_root is not None else None,
        manifest=manifest,
    )


def validate_frozen_inputs(frozen: FrozenNativeEvolutionInputs) -> None:
    """Validate the detached freeze envelope and current source hashes."""
    if type(frozen) is not FrozenNativeEvolutionInputs:
        raise ResultValidationError("native evolution requires exact FrozenNativeEvolutionInputs")
    try:
        validate_manifest_hash(frozen.manifest, label="native evolution input freeze")
    except Exception as exc:
        raise ResultValidationError(str(exc)) from exc
    if not frozen.dataset.exists():
        raise ResultValidationError(f"native evolution dataset is absent: {frozen.dataset}")
    if (
        not isinstance(frozen.capability, Mapping)
        or not isinstance(frozen.selections, Mapping)
        or (not isinstance(frozen.selection_dataset_files, Mapping))
        or (not isinstance(frozen.selection_assets, Mapping))
        or (set(frozen.selections) != set(frozen.selection_dataset_files))
        or (set(frozen.selections) != set(frozen.selection_assets))
    ):
        raise ResultValidationError("native evolution selection freeze is malformed")
    try:
        validate_public_native_capability(frozen.capability, benchmark=frozen.benchmark)
    except Exception as exc:
        raise ResultValidationError("native evolution capability binding is malformed") from exc
    for role, selection in frozen.selections.items():
        if (
            type(role) is not str
            or re.fullmatch("[A-Za-z0-9][A-Za-z0-9._-]*", role) is None
            or type(selection) is not NativeEvaluationSelection
        ):
            raise ResultValidationError(f"invalid native evolution selection role: {role!r}")
        selection.validate()
    observed_files: list[tuple[str, str]] = []
    for raw_path, expected_hash in frozen.dataset_files:
        path = Path(raw_path)
        try:
            observed = snapshot_regular_file(path).sha256
        except (OSError, ValueError) as exc:
            raise ResultValidationError(
                f"native evolution dataset file cannot be frozen: {path}"
            ) from exc
        if observed != expected_hash:
            raise ResultValidationError("native evolution dataset drifted after input freeze")
        observed_files.append((str(path), observed))
    if tuple(observed_files) != frozen.dataset_files:
        raise ResultValidationError("native evolution dataset freeze is malformed")
    try:
        expected_manifest = {
            "benchmark": frozen.benchmark,
            "capability": thaw_json_mapping(frozen.capability),
            "dataset": {
                "path": str(frozen.dataset),
                "files": [
                    {"path": path, "sha256": digest} for path, digest in frozen.dataset_files
                ],
            },
            "selections": {
                role: {
                    "split": selection.split,
                    "limit": selection.limit,
                    "ids_file": str(selection.ids_file.expanduser().absolute())
                    if selection.ids_file
                    else None,
                    "ids_file_sha256": snapshot_regular_file(
                        selection.ids_file.expanduser().absolute()
                    ).sha256
                    if selection.ids_file
                    else None,
                    "task_ids": list(selection.task_ids),
                    "count": len(selection.task_ids),
                    "dataset_files": [
                        {"path": path, "sha256": digest}
                        for path, digest in frozen.selection_dataset_files[role]
                    ],
                    "task_assets": {
                        task_id: [dict(asset) for asset in assets]
                        for task_id, assets in frozen.selection_assets[role].items()
                    },
                }
                for role, selection in frozen.selections.items()
            },
            "seed": frozen.seed,
            "asset_root": str(frozen.asset_root) if frozen.asset_root is not None else None,
        }
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ResultValidationError(
            "native evolution input freeze is malformed or drifted"
        ) from exc
    manifest = thaw_json_mapping(frozen.manifest)
    for field, value in expected_manifest.items():
        if manifest.get(field) != value:
            raise ResultValidationError(f"native evolution input freeze drifted: {field}")
    if (
        manifest.get("status") != "RETHINKSKILL_NATIVE_EVOLUTION_INPUTS_FROZEN"
        or manifest.get("schema_version") != 5
        or manifest.get("model_calls") != 0
        or (manifest.get("target_calls") != 0)
        or (manifest.get("optimizer_calls") != 0)
    ):
        raise ResultValidationError(
            "native evolution input freeze schema or call accounting is invalid"
        )


def validate_frozen_plan(
    frozen: FrozenNativeEvolutionInputs,
    *,
    role: str,
    selection: NativeEvaluationSelection,
    plan: NativeRunPlan,
) -> None:
    validate_frozen_inputs(frozen)
    if type(selection) is not NativeEvaluationSelection:
        raise ResultValidationError("native evolution requires an exact NativeEvaluationSelection")
    if type(plan) is not NativeRunPlan:
        raise ResultValidationError("native evolution requires an exact NativeRunPlan")
    if role not in frozen.selections or frozen.selections[role] != selection:
        raise ResultValidationError(f"native evolution selection role drifted: {role!r}")
    observed_files = tuple(
        (str(item["path"]), str(item["sha256"])) for item in plan.preflight["dataset"]["files"]
    )
    if observed_files != frozen.selection_dataset_files[role]:
        raise ResultValidationError("native evolution dataset drifted after input freeze")
    observed_ids = tuple(task.task_id for task in plan.tasks)
    if observed_ids != selection.task_ids:
        raise ResultValidationError("native evolution task selection drifted after input freeze")
    observed_assets = {
        str(task_id): tuple(dict(asset) for asset in assets)
        for task_id, assets in plan.preflight["selection"]["task_assets"].items()
    }
    if observed_assets != frozen.selection_assets[role]:
        raise ResultValidationError("native evolution task assets drifted after input freeze")
    if plan.preflight["capability"] != frozen.capability:
        raise ResultValidationError("native evolution capability drifted after input freeze")


@dataclass(frozen=True, slots=True)
class NativeSkillEvaluator:
    """Evaluate skill text through one native harness and target executor."""

    repository: Repository
    catalog: NativeCatalog
    frozen: FrozenNativeEvolutionInputs
    output_root: Path
    executor: ModelExecutor
    timeout_seconds: int = 900

    def _validate_capability_binding(self) -> None:
        capability = resolve_native_capability(self.catalog, self.frozen.benchmark)
        if freeze_json_mapping(capability.public()) != self.frozen.capability:
            raise ResultValidationError("native evaluator capability drifted after input freeze")

    def public_manifest(self) -> Mapping[str, object]:
        validate_frozen_inputs(self.frozen)
        self._validate_capability_binding()
        return {
            "kind": "native-skill-evaluator",
            "benchmark": self.frozen.benchmark,
            "capability": thaw_json_mapping(self.frozen.capability),
            "dataset": str(self.frozen.dataset),
            "selections": {
                role: {"split": selection.split, "task_ids": list(selection.task_ids)}
                for role, selection in self.frozen.selections.items()
            },
            "dataset_files": [
                {"path": path, "sha256": digest} for path, digest in self.frozen.dataset_files
            ],
            "seed": self.frozen.seed,
            "asset_root": str(self.frozen.asset_root)
            if self.frozen.asset_root is not None
            else None,
            "executor": freeze_component_manifest(self.executor, label="native evaluator executor"),
        }

    def evaluate(self, skill: str, *, split: str, round_no: int) -> EvaluationOutcome:
        validate_frozen_inputs(self.frozen)
        self._validate_capability_binding()
        try:
            selection = self.frozen.selections[split]
        except KeyError as exc:
            raise ResultValidationError(
                f"native evaluator has no selection role {split!r}"
            ) from exc
        label = f"{split}_round_{round_no:04d}"
        skill_path = self.output_root / "evaluation_skills" / f"{label}.md"
        atomic_write(skill_path, skill.encode("utf-8"))
        ids_path = self.output_root / "selection_ids" / f"{split}.txt"
        atomic_write(
            ids_path, "".join(f"{task_id}\n" for task_id in selection.task_ids).encode("utf-8")
        )
        evaluation_root = self.output_root / "evaluations" / label
        plan = plan_native_run(
            repository=self.repository,
            catalog=self.catalog,
            benchmark=self.frozen.benchmark,
            dataset=self.frozen.dataset,
            skill=skill_path,
            output_root=evaluation_root,
            options=NativeRunOptions(
                split=selection.split,
                ids_file=ids_path,
                seed=self.frozen.seed,
                asset_root=self.frozen.asset_root,
                timeout_seconds=self.timeout_seconds,
            ),
        )
        validate_frozen_plan(self.frozen, role=split, selection=selection, plan=plan)
        receipt = execute_native_plan(plan, executor=self.executor, authorized=True)
        return evaluation_outcome(
            receipt=receipt,
            results_path=evaluation_root / "results.jsonl",
            selected_tasks=len(plan.tasks),
            reference_path=(
                evaluation_root.relative_to(self.output_root) / "results.jsonl"
            ).as_posix(),
        )
