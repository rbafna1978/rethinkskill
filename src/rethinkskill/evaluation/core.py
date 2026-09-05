"""RethinkSkill evaluation core."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from rethinkskill.benchmarks.capabilities import BenchmarkCapability, capability_catalog
from rethinkskill.benchmarks.core import BenchmarkSpec, artifact_path
from rethinkskill.benchmarks.scoring import Verdict, Verification, freeze_verification
from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.integrations.catalog import IntegrationSpec
from rethinkskill.utils.fs import OutputLock, utc_now
from rethinkskill.utils.integrity import freeze_manifest
from rethinkskill.utils.plugins import ExtensionRegistration
from rethinkskill.utils.serde import (
    RegularFileSnapshot,
    atomic_write,
    atomic_write_json,
    atomic_write_snapshot,
    freeze_json_mapping,
    sha256_bytes,
    snapshot_regular_file,
    strict_json_loads,
)


def verdict_payload(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def evaluation_manifest(
    *,
    benchmark: str,
    spec: BenchmarkSpec,
    scoring_capability: Mapping[str, object],
    input_path: Path,
    input_sha256: str,
    input_size: int,
    assets: Sequence[Mapping[str, object]],
    output_path: Path,
    case_count: int,
    verdicts: Sequence[Mapping[str, object]],
    verifications: Sequence[Verification],
    payload: bytes,
    created_at: str,
) -> Mapping[str, object]:
    counts = {
        verdict.value: sum(row["verdict"] == verdict.value for row in verdicts)
        for verdict in Verdict
    }
    return freeze_manifest(
        {
            "schema_version": 4,
            "status": "RETHINKSKILL_DETERMINISTIC_VERDICTS_FROZEN",
            "created_at": created_at,
            "benchmark": benchmark,
            "input": {
                "path": str(input_path),
                "sha256": input_sha256,
                "rows": case_count,
                "size": input_size,
            },
            "assets": [dict(record) for record in assets],
            "artifact_fields": list(spec.artifact_fields),
            "output": {
                "path": str(output_path),
                "sha256": sha256_bytes(payload),
                "rows": len(verdicts),
                "size": len(payload),
            },
            "counts": counts,
            "metrics": spec.summarize(verifications),
            "scoring_capability": scoring_capability,
            "model_calls": 0,
            "target_calls": 0,
            "optimizer_calls": 0,
        }
    )


class ScoringCatalog(Protocol):
    """Selection interface required by model-free benchmark scoring."""

    def resolve_scorer(self, name: str) -> BenchmarkSpec: ...


class EvaluationCatalog(Protocol):
    """Joined selection interface required by persisted evaluations."""

    def resolve_capability(self, name: str) -> BenchmarkCapability: ...


def materialize_evaluation_inputs(
    *,
    spec: BenchmarkSpec,
    cases: Mapping[str, Mapping[str, Any]],
    source_path: Path,
    source_snapshot: RegularFileSnapshot,
    staging_root: Path,
    evidence_name: str,
) -> tuple[Path, list[dict[str, object]]]:
    """Copy the exact scorer read-set into a relocation-safe evidence tree."""
    frozen_base = staging_root / "input"
    frozen_input = frozen_base / source_path.name
    atomic_write_snapshot(frozen_input, source_snapshot.payload)
    observed: dict[str, RegularFileSnapshot] = {}
    for case_id, row in cases.items():
        for field in spec.artifact_fields:
            raw_record = row.get(field)
            if raw_record is None:
                continue
            if not isinstance(raw_record, Mapping):
                continue
            raw_path = raw_record.get("path")
            if type(raw_path) is not str:
                continue
            path = artifact_path(source_path.parent, raw_record, f"case {case_id!r} {field}")
            try:
                snapshot = snapshot_regular_file(path)
            except (OSError, ValueError) as exc:
                raise ResultValidationError(
                    f"case {case_id!r} {field} is not a regular file"
                ) from exc
            prior = observed.get(raw_path)
            if prior is not None and prior != snapshot:
                raise ResultValidationError(f"declared evaluation asset drifted: {raw_path}")
            observed[raw_path] = snapshot
    records: list[dict[str, object]] = []
    for logical_path, snapshot in sorted(observed.items()):
        relative = PurePosixPath(logical_path)
        target = frozen_base.joinpath(*relative.parts)
        atomic_write_snapshot(target, snapshot.payload)
        records.append(
            {
                "logical_path": logical_path,
                "path": f"{evidence_name}/input/{relative.as_posix()}",
                "sha256": snapshot.sha256,
                "size": snapshot.size,
            }
        )
    return (frozen_input, records)


def load_cases_bytes(payload: bytes, *, source: Path) -> dict[str, dict[str, Any]]:
    """Parse one immutable byte snapshot of a JSONL case file."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResultValidationError(f"{source}: case input must be UTF-8") from exc
    cases: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = strict_json_loads(line)
        except json.JSONDecodeError as exc:
            raise ResultValidationError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ResultValidationError(f"{source}:{line_number}: expected object")
        case_id = value.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ResultValidationError(f"{source}:{line_number}: missing or invalid case_id")
        if case_id in cases:
            raise ResultValidationError(f"{source}:{line_number}: duplicate case_id={case_id}")
        cases[case_id] = value
    if not cases:
        raise ResultValidationError(f"no cases found in {source}")
    return cases


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    return load_cases_bytes(path.read_bytes(), source=path)


@dataclass(frozen=True, slots=True)
class DeterministicScoringBinding:
    """One exact scorer plus catalog-owned registration and ownership."""

    spec: BenchmarkSpec
    registration: ExtensionRegistration
    manifest: Mapping[str, object]


def resolve_deterministic_scoring_binding(
    catalog: object, benchmark: str
) -> DeterministicScoringBinding:
    """Resolve one joined capability without probing unrelated runtimes."""
    resolver = getattr(catalog, "resolve_capability", None)
    if not callable(resolver):
        raise ConfigurationError("deterministic evaluation requires a unified capability catalog")
    capability = resolver(benchmark)
    if type(capability) is not BenchmarkCapability:
        raise ConfigurationError("evaluation catalog must return an exact BenchmarkCapability")
    spec = capability.scoring
    registration = capability.scoring_registration
    integration = capability.integration
    if type(spec) is not BenchmarkSpec:
        raise ConfigurationError("evaluation capability requires an exact BenchmarkSpec")
    if type(registration) is not ExtensionRegistration:
        raise ConfigurationError(
            "evaluation capability requires exact scorer registration provenance"
        )
    if type(integration) is not IntegrationSpec:
        raise ConfigurationError("evaluation capability requires an exact IntegrationSpec")
    spec.validate()
    registration.validate()
    integration.validate()
    if (
        spec.name != benchmark
        or registration.name != benchmark
        or registration.component_kind != "benchmark"
        or (benchmark not in integration.benchmarks)
    ):
        raise ConfigurationError("deterministic evaluation capability binding is inconsistent")
    if not spec.locally_evaluable or spec.adapter is None:
        raise ConfigurationError(f"benchmark requires its external harness: {benchmark}")
    manifest = freeze_json_mapping(
        {
            "name": benchmark,
            "scoring": spec.public(),
            "scoring_registration": registration.public(),
            "integration": integration.public(),
        }
    )
    return DeterministicScoringBinding(spec=spec, registration=registration, manifest=manifest)


def evaluate_case(
    benchmark: str, row: dict[str, Any], base: Path, *, catalog: ScoringCatalog | None = None
) -> Verification:
    spec = (catalog or capability_catalog()).resolve_scorer(benchmark)
    if not spec.locally_evaluable or spec.adapter is None:
        raise ConfigurationError(f"benchmark requires its external harness: {benchmark}")
    return spec.adapter(row, base)


def evaluate_cases(
    *, benchmark: str, input_path: Path, output_path: Path, catalog: EvaluationCatalog | None = None
) -> Mapping[str, object]:
    active_catalog = catalog or capability_catalog()
    binding = resolve_deterministic_scoring_binding(active_catalog, benchmark)
    spec = binding.spec
    assert spec.adapter is not None
    source_path = input_path.expanduser().absolute()
    try:
        source_snapshot = snapshot_regular_file(source_path)
    except (OSError, ValueError) as exc:
        raise ResultValidationError(
            f"case input must be a regular non-symlink file: {source_path}"
        ) from exc
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    evidence_root = output_path.with_name(output_path.name + ".evidence")
    with OutputLock(output_path):
        if output_path.exists() or output_path.is_symlink():
            raise ResultValidationError(f"refusing to replace existing verdicts: {output_path}")
        if manifest_path.exists() or manifest_path.is_symlink():
            raise ResultValidationError(f"refusing to replace existing manifest: {manifest_path}")
        if evidence_root.exists() or evidence_root.is_symlink():
            raise ResultValidationError(f"refusing to replace existing evidence: {evidence_root}")
        input_payload = source_snapshot.payload
        cases = load_cases_bytes(input_payload, source=source_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_path.name}.evidence-", dir=output_path.parent
        ) as temporary:
            staging_root = Path(temporary)
            frozen_input, assets = materialize_evaluation_inputs(
                spec=spec,
                cases=cases,
                source_path=source_path,
                source_snapshot=source_snapshot,
                staging_root=staging_root,
                evidence_name=evidence_root.name,
            )
            verdicts: list[dict[str, object]] = []
            verifications: list[Verification] = []
            for case_id, row in cases.items():
                observed = spec.adapter(row, frozen_input.parent)
                try:
                    verification = freeze_verification(observed)
                except ResultValidationError as exc:
                    raise ResultValidationError(
                        f"deterministic scorer must return a valid exact Verification: {benchmark}:{case_id}"
                    ) from exc
                verifications.append(verification)
                verdicts.append({"case_id": case_id, **verification.to_dict()})
            verdicts.sort(key=lambda row: str(row["case_id"]))
            payload = verdict_payload(verdicts)
            manifest = evaluation_manifest(
                benchmark=benchmark,
                spec=spec,
                scoring_capability=binding.manifest,
                input_path=Path(evidence_root.name, "input", source_path.name),
                input_sha256=source_snapshot.sha256,
                input_size=source_snapshot.size,
                assets=assets,
                output_path=Path(output_path.name),
                case_count=len(cases),
                verdicts=verdicts,
                verifications=verifications,
                payload=payload,
                created_at=utc_now(),
            )
            os.replace(staging_root, evidence_root)
            atomic_write(output_path, payload)
            atomic_write_json(manifest_path, manifest)
    return manifest
