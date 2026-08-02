"""Parent-side validation for the external WebShop Python environment."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from rethinkskill.errors import ConfigurationError
from rethinkskill.utils.serde import (
    canonical_json_bytes,
    sha256_bytes,
    snapshot_regular_file,
    strict_json_loads,
)

_ERROR_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NORMALIZE_DISTRIBUTION = re.compile(r"[-_.]+")
_PROBE_TIMEOUT_SECONDS = 180
_MAX_PROBE_OUTPUT_BYTES = 8 * 1024 * 1024


def build_child_environment(allowlist: Sequence[str]) -> dict[str, str]:
    """Build the credential-minimal environment shared by probe and bridge."""

    environment = {key: os.environ[key] for key in allowlist if key in os.environ}
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _bounded_identity_text(value: object) -> bool:
    return bool(
        type(value) is str
        and value
        and len(value) <= 512
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
    )


def validate_python_environment_identity(
    value: object,
) -> dict[str, object]:
    """Validate and recompute one external probe result."""

    if type(value) is not dict or set(value) != {
        "schema_version",
        "kind",
        "python",
        "sha256",
        "distributions",
        "distribution_count",
        "files",
        "bytes",
    }:
        raise ConfigurationError("WebShop Python environment identity schema is invalid")
    if value.get("schema_version") != 1 or value.get("kind") != "python_distribution_environment":
        raise ConfigurationError("WebShop Python environment identity header is invalid")
    python = value.get("python")
    if type(python) is not dict or set(python) != {
        "implementation",
        "version",
        "cache_tag",
    }:
        raise ConfigurationError("WebShop Python implementation identity is invalid")
    version = python.get("version")
    if (
        not _bounded_identity_text(python.get("implementation"))
        or (
            python.get("cache_tag") is not None
            and not _bounded_identity_text(python.get("cache_tag"))
        )
        or type(version) is not list
        or len(version) != 3
        or any(type(part) is not int or part < 0 for part in version)
    ):
        raise ConfigurationError("WebShop Python implementation identity is invalid")
    distributions = value.get("distributions")
    if type(distributions) is not list or not distributions:
        raise ConfigurationError("WebShop Python distribution inventory is empty")
    normalized_names: list[str] = []
    files = 0
    size = 0
    for row in distributions:
        if type(row) is not dict or set(row) != {
            "distribution",
            "normalized_name",
            "version",
            "sha256",
            "files",
            "bytes",
        }:
            raise ConfigurationError("WebShop Python distribution identity schema is invalid")
        name = row.get("distribution")
        normalized = row.get("normalized_name")
        digest = row.get("sha256")
        row_files = row.get("files")
        row_size = row.get("bytes")
        if (
            not _bounded_identity_text(name)
            or not _bounded_identity_text(normalized)
            or normalized != _NORMALIZE_DISTRIBUTION.sub("-", name).lower()
            or not _bounded_identity_text(row.get("version"))
            or type(digest) is not str
            or _SHA256.fullmatch(digest) is None
            or type(row_files) is not int
            or row_files < 1
            or type(row_size) is not int
            or row_size < 0
        ):
            raise ConfigurationError("WebShop Python distribution identity is invalid")
        normalized_names.append(normalized)
        files += row_files
        size += row_size
    if normalized_names != sorted(normalized_names):
        raise ConfigurationError("WebShop Python distribution inventory is not sorted")
    if len(normalized_names) != len(set(normalized_names)):
        raise ConfigurationError("WebShop Python distribution inventory contains duplicates")
    digest = value.get("sha256")
    if (
        type(digest) is not str
        or _SHA256.fullmatch(digest) is None
        or digest != sha256_bytes(canonical_json_bytes(distributions))
        or value.get("distribution_count") != len(distributions)
        or value.get("files") != files
        or value.get("bytes") != size
    ):
        raise ConfigurationError("WebShop Python environment identity summary is invalid")
    return value


def _executable_identity(
    path: Path,
    *,
    label: str,
) -> dict[str, object]:
    """Freeze one configured launcher without losing symlink semantics."""

    configured = path.expanduser().absolute()
    try:
        before = configured.lstat()
        link_target = os.readlink(configured) if stat.S_ISLNK(before.st_mode) else None
        resolved = configured.resolve(strict=True)
        snapshot = snapshot_regular_file(resolved)
        after = configured.lstat()
        final_link_target = os.readlink(configured) if stat.S_ISLNK(after.st_mode) else None
    except (OSError, ValueError) as exc:
        raise ConfigurationError(
            f"WebShop {label} executable identity is unavailable: {type(exc).__name__}"
        ) from exc
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        link_target,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        final_link_target,
    )
    if before_identity != after_identity:
        raise ConfigurationError(f"WebShop {label} executable changed while reading")
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ConfigurationError(f"WebShop {label} executable target is not a regular file")
    return {
        "configured_path": str(configured),
        "configured_is_symlink": stat.S_ISLNK(before.st_mode),
        "configured_link_target": link_target,
        "resolved_path": str(resolved),
        "sha256": snapshot.sha256,
        "size": snapshot.size,
    }


def python_executable_identity(path: Path) -> dict[str, object]:
    """Freeze the configured Python launcher and resolved executable."""

    return _executable_identity(path, label="Python")


def java_executable_identity(path: Path) -> dict[str, object]:
    """Freeze the configured Java launcher and resolved executable."""

    return _executable_identity(path, label="Java")


def probe_python_environment_identity(
    *,
    python: Path,
    checkout_root: Path,
    probe: Path,
    environment: Mapping[str, str],
) -> dict[str, object]:
    """Run the external probe and independently validate its bounded output."""

    argv = [
        str(python.expanduser().absolute()),
        "-B",
        "-I",
        str(probe.resolve(strict=True)),
    ]
    try:
        completed = subprocess.run(
            argv,
            cwd=checkout_root.resolve(strict=True),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigurationError(
            f"WebShop Python environment probe failed: {type(exc).__name__}"
        ) from exc
    payload = completed.stdout
    if type(payload) is not bytes or not payload or len(payload) > _MAX_PROBE_OUTPUT_BYTES:
        raise ConfigurationError("WebShop Python environment probe returned invalid output")
    try:
        response = strict_json_loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("WebShop Python environment probe returned invalid JSON") from exc
    if completed.returncode != 0:
        if (
            type(response) is dict
            and set(response) == {"status", "error_code", "error_type"}
            and response.get("status") == "error"
            and response.get("error_code") == "environment_identity_unavailable"
            and type(response.get("error_type")) is str
            and _ERROR_TYPE.fullmatch(response["error_type"]) is not None
        ):
            raise ConfigurationError(
                "WebShop Python environment probe reported environment_identity_unavailable"
            )
        raise ConfigurationError("WebShop Python environment probe failed")
    if (
        type(response) is not dict
        or set(response) != {"status", "identity"}
        or response.get("status") != "ok"
    ):
        raise ConfigurationError("WebShop Python environment probe response is invalid")
    return validate_python_environment_identity(response.get("identity"))


__all__ = [
    "build_child_environment",
    "java_executable_identity",
    "probe_python_environment_identity",
    "python_executable_identity",
    "validate_python_environment_identity",
]
