"""Native SpreadsheetBench artifact harness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rethinkskill.benchmarks.scoring import Verdict, Verification, verify_spreadsheet
from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.runtime.tasks import (
    NativeAsset,
    NativeTask,
    RenderedTask,
    _dataset_files,
    _load_json_records,
    _select_tasks,
    _text,
)
from rethinkskill.runtime.types import (
    ModelExecutor,
    ModelOutcome,
    numeric_version,
    probe_distribution,
)
from rethinkskill.utils.integrity import boundary_failure, safe_exception_type
from rethinkskill.utils.serde import atomic_write, thaw_json_mapping
from rethinkskill_spreadsheetbench.runtime import (
    extract_python_code,
    preview_workbook_bytes,
    run_generated_code,
)


def _artifact_failure_outcome(
    provider_outcome: ModelOutcome,
    execution_rows: Sequence[Mapping[str, object]],
    exc: BaseException,
) -> ModelOutcome:
    return ModelOutcome(
        status="FAILED",
        response=provider_outcome.response,
        raw=provider_outcome.raw,
        process={
            **thaw_json_mapping(provider_outcome.process),
            "artifact_runner": {
                "kind": "offline_python_codegen",
                "exception_type": safe_exception_type(exc),
                "cases": list(execution_rows),
            },
        },
        attempted_calls=provider_outcome.attempted_calls,
        completed_calls=provider_outcome.completed_calls,
        failure_class="artifact_runner_error",
        failure=boundary_failure(
            "SpreadsheetBench artifact runner",
            exc,
        ),
    )


def _spreadsheet_cases(task_directory: Path) -> tuple[tuple[str, Path, Path], ...]:
    """Discover the three filename layouts used by SpreadsheetBench."""

    cases: list[tuple[str, Path, Path]] = []
    for input_path in sorted(task_directory.glob("*_input.xlsx")):
        case_no = input_path.name.split("_", 1)[0]
        golden = Path(str(input_path).replace("_input.xlsx", "_answer.xlsx"))
        if golden.is_file():
            cases.append((case_no, input_path, golden))
    for input_path in sorted(task_directory.glob("*_init.xlsx")):
        case_no = input_path.name.split("_", 1)[0]
        golden = Path(str(input_path).replace("_init.xlsx", "_golden.xlsx"))
        if golden.is_file():
            cases.append((case_no, input_path, golden))
    if not cases:
        initial = task_directory / "initial.xlsx"
        golden = task_directory / "golden.xlsx"
        if initial.is_file() and golden.is_file():
            cases.append(("1", initial, golden))
    return tuple(cases)


def _spreadsheet_task_directory(
    root: Path,
    record: Mapping[str, Any],
    task_id: str,
) -> Path:
    value = record.get("spreadsheet_path", f"spreadsheet/{task_id}")
    if type(value) is not str:
        raise ResultValidationError(f"spreadsheet_path must be text: {task_id}")
    raw = value.strip()
    if not raw:
        raise ResultValidationError(f"spreadsheet task has an empty spreadsheet_path: {task_id}")
    candidate = Path(raw)
    resolved = (
        candidate.expanduser().resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    if not resolved.is_relative_to(root):
        raise ResultValidationError(
            f"spreadsheet task directory escapes --asset-root: {task_id}: {resolved}"
        )
    if not resolved.is_dir():
        raise ResultValidationError(f"spreadsheet task directory is absent: {task_id}: {resolved}")
    return resolved


@dataclass(frozen=True, slots=True)
class SpreadsheetBenchHarness:
    """Official codegen-style SpreadsheetBench artifact execution."""

    def dependency_manifest(self) -> Mapping[str, object]:
        dependency = probe_distribution(
            distribution="openpyxl",
            import_name="openpyxl",
            requirement=">=3.1,<4",
            accepts=lambda value: (
                (parsed := numeric_version(value)) is not None
                and parsed >= (3, 1)
                and parsed < (4,)
            ),
        )
        return {
            "kind": "offline_python_codegen",
            **dependency,
            "openpyxl_available": dependency["module_available"],
        }

    def provenance_files(
        self,
        source: Path,
        *,
        split: str,
    ) -> tuple[Path, ...]:
        return _dataset_files(source, split)

    def load_tasks(
        self,
        source: Path,
        *,
        split: str,
        limit: int | None,
        requested_ids: Sequence[str],
        seed: int,
        asset_root: Path | None,
    ) -> tuple[NativeTask, ...]:
        del seed
        dependency = self.dependency_manifest()
        if not dependency["ready"]:
            raise ConfigurationError(
                "SpreadsheetBench native execution requires "
                "openpyxl>=3.1,<4; observed "
                f"{dependency.get('version') or 'not installed'}"
            )
        if asset_root is None:
            raise ConfigurationError(
                "SpreadsheetBench native execution requires --asset-root "
                "pointing to the frozen workbook corpus"
            )
        root = asset_root.expanduser().resolve()
        tasks: list[NativeTask] = []
        first_input_payloads: list[bytes] = []
        for path in _dataset_files(source, split):
            for record in _load_json_records(path):
                task_id = _text(record, "id")
                instruction = _text(record, "instruction")
                instruction_type = _text(
                    record,
                    "instruction_type",
                    default="",
                )
                answer_position = _text(
                    record,
                    "answer_position",
                    default="",
                ).strip()
                answer_sheet = _text(
                    record,
                    "answer_sheet",
                    default="",
                ).strip()
                if answer_position and answer_sheet and "!" not in answer_position:
                    answer_position = f"{answer_sheet}!{answer_position}"
                if not answer_position:
                    raise ResultValidationError(
                        f"spreadsheet task has no answer_position: {task_id}"
                    )
                task_directory = _spreadsheet_task_directory(
                    root,
                    record,
                    task_id,
                )
                discovered = _spreadsheet_cases(task_directory)
                if not discovered:
                    raise ResultValidationError(
                        f"spreadsheet task has no input/golden cases: {task_id}"
                    )
                assets: list[NativeAsset] = []
                cases: list[dict[str, str]] = []
                first_input_payload: bytes | None = None
                for index, (case_no, input_path, golden_path) in enumerate(discovered):
                    for artifact in (input_path, golden_path):
                        resolved_artifact = artifact.resolve()
                        if not resolved_artifact.is_relative_to(root):
                            raise ResultValidationError(
                                f"spreadsheet workbook escapes --asset-root: {resolved_artifact}"
                            )
                    target = f"cases/{index:03d}"
                    input_target = f"{target}/input.xlsx"
                    golden_target = f"{target}/golden.xlsx"
                    input_asset, input_snapshot = NativeAsset.freeze_with_snapshot(
                        input_path,
                        target=input_target,
                        media_type=(
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        ),
                        role="input_workbook",
                    )
                    assets.append(input_asset)
                    if first_input_payload is None:
                        first_input_payload = input_snapshot.payload
                    assets.append(
                        NativeAsset.freeze(
                            golden_path,
                            target=golden_target,
                            media_type=(
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            ),
                            role="reference_workbook",
                            visibility="verifier",
                        )
                    )
                    cases.append(
                        {
                            "case_no": case_no,
                            "input_target": input_target,
                            "golden_target": golden_target,
                            "output_target": f"{target}/output.xlsx",
                        }
                    )
                tasks.append(
                    NativeTask(
                        task_id=task_id,
                        payload={
                            "instruction": instruction,
                            "instruction_type": instruction_type,
                            "answer_position": answer_position,
                            "cases": tuple(cases),
                            "execution_mode": "single_codegen",
                        },
                        assets=tuple(assets),
                    )
                )
                if first_input_payload is None:
                    raise ResultValidationError(f"spreadsheet task has no frozen input: {task_id}")
                first_input_payloads.append(first_input_payload)
        selected = _select_tasks(
            tasks,
            limit=limit,
            requested_ids=requested_ids,
        )
        payload_by_id = {
            task.task_id: payload for task, payload in zip(tasks, first_input_payloads, strict=True)
        }
        normalized: list[NativeTask] = []
        for task in selected:
            preview = preview_workbook_bytes(payload_by_id[task.task_id])
            normalized.append(
                NativeTask(
                    task_id=task.task_id,
                    payload={
                        **dict(task.payload),
                        "first_input_preview": preview,
                    },
                    assets=task.assets,
                )
            )
        return tuple(normalized)

    def render(self, task: NativeTask, skill: str) -> RenderedTask:
        task.validate()
        inputs = tuple(
            asset
            for asset in task.assets
            if asset.role == "input_workbook" and asset.visibility == "workspace"
        )
        if not inputs:
            raise ResultValidationError(
                f"spreadsheet task has no visible input workbook: {task.task_id}"
            )
        preview = task.payload["first_input_preview"]
        task_markdown = (
            "# SpreadsheetBench Task\n\n"
            f"Task ID: `{task.task_id}`\n\n"
            f"Instruction type: {task.payload['instruction_type']}\n\n"
            f"Answer position: `{task.payload['answer_position']}`\n\n"
            "## Instruction\n\n"
            f"{task.payload['instruction']}\n\n"
            "## First input workbook preview\n\n"
            f"```text\n{preview}\n```\n\n"
            "## Output contract\n\n"
            "Return exactly one complete Python program in a `python` code "
            "fence. The program receives `INPUT_PATH` and `OUTPUT_PATH`, "
            "loads the input workbook, applies the instruction, and saves the "
            "result to `OUTPUT_PATH`. Do not hardcode previewed values or row "
            "counts. Preserve unrelated workbook content. Do not access the "
            "network, subprocesses, or files other than the supplied workbook."
        )
        rendered = RenderedTask(
            task_markdown=task_markdown,
            skill_markdown=(
                "---\n"
                'name: "rethinkskill-target"\n'
                'description: "Dynamic skill for native SpreadsheetBench '
                'code generation."\n'
                "---\n\n"
                f"{skill.strip()}\n"
            ),
            invocation=(
                "Read `.agents/skills/rethinkskill-target/SKILL.md` and "
                "`task.md`. Inspect the task-local input workbook only if "
                "useful. Return exactly one self-contained Python code fence "
                "that transforms `INPUT_PATH` into `OUTPUT_PATH`; do not run it."
            ),
            attachments=inputs,
        )
        rendered.validate()
        return rendered

    def execute_task(
        self,
        task: NativeTask,
        rendered: RenderedTask,
        executor: ModelExecutor,
        *,
        workspace: Path,
        verifier_workspace: Path,
        timeout_seconds: int,
    ) -> ModelOutcome:
        del verifier_workspace
        provider_outcome = executor.execute(
            rendered,
            workspace=workspace,
            timeout_seconds=timeout_seconds,
        )
        if not provider_outcome.ok:
            return provider_outcome
        execution_rows: list[dict[str, object]] = []
        try:
            code = extract_python_code(provider_outcome.response)
            atomic_write(
                workspace / "generated_solution.py",
                code.encode("utf-8"),
            )
            extraction_error = ""
        except ValueError:
            code = ""
            extraction_error = "spreadsheet_python_contract_error"
        except Exception as exc:  # noqa: BLE001
            return _artifact_failure_outcome(
                provider_outcome,
                execution_rows,
                exc,
            )
        if code:
            try:
                for case in task.payload["cases"]:
                    input_path = workspace / str(case["input_target"])
                    output_path = workspace / str(case["output_target"])
                    execution = run_generated_code(
                        code,
                        input_path=input_path,
                        output_path=output_path,
                        timeout_seconds=min(timeout_seconds, 120),
                    )
                    execution_rows.append(
                        {
                            "case_no": str(case["case_no"]),
                            **execution,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                return _artifact_failure_outcome(
                    provider_outcome,
                    execution_rows,
                    exc,
                )
        return ModelOutcome(
            status=provider_outcome.status,
            response=provider_outcome.response,
            raw=provider_outcome.raw,
            process={
                **thaw_json_mapping(provider_outcome.process),
                "artifact_runner": {
                    "kind": "offline_python_codegen",
                    "extraction_error": extraction_error,
                    "cases": execution_rows,
                },
            },
            attempted_calls=provider_outcome.attempted_calls,
            completed_calls=provider_outcome.completed_calls,
            failure_class=provider_outcome.failure_class,
            failure=provider_outcome.failure,
        )

    def evaluate_workspace(
        self,
        task: NativeTask,
        response: str,
        *,
        workspace: Path,
        verifier_workspace: Path,
    ) -> Verification:
        del response
        results: list[Verification] = []
        for case in task.payload["cases"]:
            results.append(
                verify_spreadsheet(
                    workspace / str(case["output_target"]),
                    verifier_workspace / str(case["golden_target"]),
                    task.payload["answer_position"],
                )
            )
        passed = sum(result.verdict is Verdict.PASS for result in results)
        total = len(results)
        fraction = passed / total if total else 0.0
        if total and passed == total:
            return Verification(
                Verdict.PASS,
                "all_spreadsheet_cases_pass",
                checked=total,
                metrics={"case_fraction": fraction},
            )
        first_failure = next(
            (result.reason for result in results if result.verdict is not Verdict.PASS),
            "no_spreadsheet_cases",
        )
        return Verification(
            Verdict.FAIL if total else Verdict.ABSTAIN,
            f"spreadsheet_case_failure:{first_failure}",
            checked=total,
            metrics={"case_fraction": fraction},
        )

    def evaluate(self, task: NativeTask, response: str) -> Verification:
        del task, response
        return Verification(
            Verdict.ABSTAIN,
            "spreadsheet_artifact_workspace_required",
        )
