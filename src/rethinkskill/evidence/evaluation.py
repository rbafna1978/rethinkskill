"""RethinkSkill evidence evaluation."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from rethinkskill.benchmarks.capabilities import capability_catalog
from rethinkskill.benchmarks.scoring import freeze_verification
from rethinkskill.evaluation.core import (
    EvaluationCatalog,
    load_cases_bytes,
    resolve_deterministic_scoring_binding,
    verdict_payload,
)
from rethinkskill.evidence.common import (
    artifact_snapshot,
    integer,
    invalid,
    json_object_snapshot,
    mapping,
)
from rethinkskill.utils.serde import RegularFileSnapshot, thaw_json_mapping


def _portable_path(value: object, *, label: str) -> PurePosixPath:
    if type(value) is not str:
        invalid(f"{label} path is invalid")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ("\\" in value)
    ):
        invalid(f"{label} path is not portable")
    return relative


def _reject_symlink_components(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            invalid(f"{label} path must not use symlinks")
    return path


def _portable_artifact(
    root: Path, reference: object, *, label: str, rows: bool
) -> tuple[dict[str, object], RegularFileSnapshot, PurePosixPath]:
    value = mapping(reference, label=f"{label} reference")
    relative = _portable_path(value.get("path"), label=label)
    path = _reject_symlink_components(root, relative, label=label)
    keys = {"path", "sha256", "size"}
    if rows:
        keys.add("rows")
    reference_value, snapshot = artifact_snapshot(
        value, path=path, label=label, keys=frozenset(keys)
    )
    if integer(reference_value.get("size"), label=f"{label} size") != snapshot.size:
        invalid(f"{label} size does not match its manifest")
    return (reference_value, snapshot, relative)


def _expected_asset_paths(
    cases: dict[str, dict[str, object]], artifact_fields: tuple[str, ...]
) -> list[str]:
    return sorted(
        {
            str(record["path"])
            for row in cases.values()
            for field in artifact_fields
            if isinstance((record := row.get(field)), dict) and type(record.get("path")) is str
        }
    )


def _validate_assets(
    *, root: Path, assets: object, expected_logical_paths: list[str], frozen_base: PurePosixPath
) -> int:
    if type(assets) is not list:
        invalid("evaluation asset evidence is invalid")
    if len(assets) != len(expected_logical_paths):
        invalid("evaluation asset evidence is not canonical")
    for index, (raw, logical_path) in enumerate(zip(assets, expected_logical_paths, strict=True)):
        record = mapping(raw, label=f"evaluation asset {index}")
        if set(record) != {"logical_path", "path", "sha256", "size"}:
            invalid("evaluation asset schema is invalid")
        if record.get("logical_path") != logical_path:
            invalid("evaluation asset evidence is not canonical")
        logical = _portable_path(logical_path, label=f"evaluation asset {index} logical")
        _, _, observed_path = _portable_artifact(
            root,
            {
                "path": record.get("path"),
                "sha256": record.get("sha256"),
                "size": record.get("size"),
            },
            label=f"evaluation asset {index}",
            rows=False,
        )
        if observed_path != frozen_base / logical:
            invalid("evaluation asset path is not bound to its frozen input")
    return len(assets)


def _replay(
    *,
    manifest: dict[str, object],
    benchmark: str,
    cases: dict[str, dict[str, object]],
    catalog: EvaluationCatalog,
    artifact_root: Path,
    base: Path,
    output_snapshot: RegularFileSnapshot,
    output_rows: int,
) -> int:
    binding = resolve_deterministic_scoring_binding(catalog, benchmark)
    if thaw_json_mapping(binding.manifest) != manifest.get("scoring_capability"):
        invalid("evaluation scoring capability does not replay")
    spec = binding.spec
    assert spec.adapter is not None
    asset_rows = 0
    if manifest.get("artifact_fields") != list(spec.artifact_fields):
        invalid("evaluation artifact-field declaration does not replay")
    input_reference = mapping(manifest.get("input"), label="evaluation input reference")
    frozen_base = _portable_path(input_reference.get("path"), label="evaluation input").parent
    asset_rows = _validate_assets(
        root=artifact_root,
        assets=manifest.get("assets"),
        expected_logical_paths=_expected_asset_paths(cases, spec.artifact_fields),
        frozen_base=frozen_base,
    )
    verdicts: list[dict[str, object]] = []
    verifications = []
    for case_id, row in cases.items():
        try:
            verification = freeze_verification(spec.adapter(row, base))
        except Exception:
            invalid(f"evaluation scorer replay failed: {benchmark}:{case_id}")
        verifications.append(verification)
        verdicts.append({"case_id": case_id, **verification.to_dict()})
    verdicts.sort(key=lambda row: str(row["case_id"]))
    if verdict_payload(verdicts) != output_snapshot.payload:
        invalid("evaluation verdict rows do not replay")
    if output_rows != len(verdicts):
        invalid("evaluation output row count does not replay")
    counts = {
        verdict: sum(row["verdict"] == verdict for row in verdicts)
        for verdict in ("PASS", "FAIL", "ABSTAIN")
    }
    if manifest.get("counts") != counts or manifest.get("metrics") != spec.summarize(verifications):
        invalid("evaluation aggregate metrics do not replay")
    return asset_rows


def validate_evaluation_evidence(
    manifest_path: Path, *, catalog: EvaluationCatalog | None = None
) -> dict[str, object]:
    """Replay current deterministic evaluation evidence without model calls."""
    path = manifest_path.expanduser().absolute()
    manifest, manifest_snapshot = json_object_snapshot(path, label="evaluation manifest")
    schema_version = manifest.get("schema_version")
    if (
        schema_version != 4
        or manifest.get("status") != "RETHINKSKILL_DETERMINISTIC_VERDICTS_FROZEN"
        or manifest.get("model_calls") != 0
        or (manifest.get("target_calls") != 0)
        or (manifest.get("optimizer_calls") != 0)
    ):
        invalid("evaluation manifest schema or zero-call status is invalid")
    benchmark = manifest.get("benchmark")
    if type(benchmark) is not str or not benchmark:
        invalid("evaluation benchmark is invalid")
    root = path.parent
    input_reference, input_snapshot, input_relative = _portable_artifact(
        root, manifest.get("input"), label="evaluation input", rows=True
    )
    output_reference, output_snapshot, output_relative = _portable_artifact(
        root, manifest.get("output"), label="evaluation output", rows=True
    )
    if input_relative == output_relative:
        invalid("evaluation input and output paths must differ")
    input_path = root.joinpath(*input_relative.parts)
    base = input_path.parent
    cases = load_cases_bytes(input_snapshot.payload, source=input_path)
    if integer(input_reference.get("rows"), label="evaluation input rows") != len(cases):
        invalid("evaluation input row count does not replay")
    asset_rows = _replay(
        manifest=manifest,
        benchmark=benchmark,
        cases=cases,
        catalog=catalog or capability_catalog(),
        artifact_root=path.parent,
        base=base,
        output_snapshot=output_snapshot,
        output_rows=integer(output_reference.get("rows"), label="evaluation output rows"),
    )
    return {
        "run_kind": "deterministic_evaluation",
        "schema_version": schema_version,
        "terminal_status": manifest["status"],
        "artifacts": {
            "manifest_sha256": manifest_snapshot.sha256,
            "input_sha256": input_snapshot.sha256,
            "output_sha256": output_snapshot.sha256,
            "result_rows": integer(output_reference.get("rows"), label="evaluation output rows"),
            "asset_rows": asset_rows,
        },
        "model_calls": 0,
    }
