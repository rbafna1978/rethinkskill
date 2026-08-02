"""RethinkSkill evidence runtime."""

from __future__ import annotations

import json
from pathlib import Path

from rethinkskill.evidence.common import artifact_snapshot, invalid, mapping, sha256_digest
from rethinkskill.runtime.runner import RUNTIME_IDENTITY_SCHEMA_VERSION, runtime_identity_sha256
from rethinkskill.utils.integrity import validate_evidence_hash
from rethinkskill.utils.serde import strict_json_loads

_EVIDENCE_KEYS = {
    "schema_version",
    "status",
    "planned",
    "observed",
    "planned_sha256",
    "observed_sha256",
    "stable",
    "model_calls",
    "evidence_sha256",
}


def replay_runtime_identity_evidence(
    *, run_root: Path, manifest_identity: object, reference: object
) -> tuple[bool, str]:
    """Replay one pre/post runtime identity comparison."""
    artifact_reference, snapshot = artifact_snapshot(
        reference,
        path=run_root / ".rethinkskill/RUNTIME_IDENTITY.json",
        label="native runtime identity evidence",
        keys=frozenset({"path", "sha256", "stable"}),
    )
    if artifact_reference.get("path") != ".rethinkskill/RUNTIME_IDENTITY.json":
        invalid("native runtime identity path is not portable")
    try:
        evidence = strict_json_loads(snapshot.payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        invalid("native runtime identity evidence is not strict JSON")
    if type(evidence) is not dict or set(evidence) != _EVIDENCE_KEYS:
        invalid("native runtime identity evidence schema is invalid")
    try:
        validate_evidence_hash(
            evidence, hash_field="evidence_sha256", label="native runtime identity evidence"
        )
    except Exception:
        invalid("native runtime identity evidence integrity check failed")
    planned = mapping(evidence.get("planned"), label="planned native runtime identity")
    observed = mapping(evidence.get("observed"), label="observed native runtime identity")
    manifest = mapping(manifest_identity, label="native manifest runtime identity")
    planned_sha256 = sha256_digest(
        evidence.get("planned_sha256"), label="planned native runtime identity SHA-256"
    )
    observed_sha256 = sha256_digest(
        evidence.get("observed_sha256"), label="observed native runtime identity SHA-256"
    )
    stable = evidence.get("stable")
    if type(stable) is not bool or type(artifact_reference.get("stable")) is not bool:
        invalid("native runtime identity stability must be boolean")
    expected_stable = planned == observed
    expected_status = (
        "RETHINKSKILL_RUNTIME_IDENTITY_STABLE"
        if expected_stable
        else "RETHINKSKILL_RUNTIME_IDENTITY_DRIFTED"
    )
    if (
        evidence.get("schema_version") != RUNTIME_IDENTITY_SCHEMA_VERSION
        or evidence.get("model_calls") != 0
        or planned != manifest
        or (planned_sha256 != runtime_identity_sha256(planned))
        or (observed_sha256 != runtime_identity_sha256(observed))
        or (stable != expected_stable)
        or (artifact_reference.get("stable") != expected_stable)
        or (evidence.get("status") != expected_status)
    ):
        invalid("native runtime identity evidence does not replay")
    return (expected_stable, snapshot.sha256)
