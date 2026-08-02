"""RethinkSkill utils release."""

from __future__ import annotations

import json
import os
import re
import site
import stat
import sysconfig
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.utils.serde import (
    canonical_json_bytes,
    freeze_json_mapping,
    sha256_bytes,
    snapshot_regular_file,
    thaw_json_mapping,
)


def regular_file_inventory(
    root: Path, *, relative_to: Path, excluded: Iterable[Path] = (), label: str
) -> tuple[dict[str, object], ...]:
    """Snapshot a tree without following links or accepting special files."""
    excluded_paths = frozenset(path.absolute() for path in excluded)
    if root.is_symlink():
        raise ResultValidationError(f"{label} root must not be a symlink: {root}")
    if not root.is_dir():
        raise ResultValidationError(f"{label} root must be a directory: {root}")
    files: list[Path] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ResultValidationError(f"{label} directory is unreadable: {directory}") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ResultValidationError(f"{label} entry cannot be inspected: {path}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ResultValidationError(f"{label} must not contain symlinks: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                visit(path)
            elif stat.S_ISREG(metadata.st_mode):
                files.append(path)
            else:
                raise ResultValidationError(f"{label} must contain only regular files: {path}")

    visit(root)
    records: list[dict[str, object]] = []
    for path in files:
        if path.absolute() in excluded_paths:
            continue
        try:
            snapshot = snapshot_regular_file(path)
            relative = path.relative_to(relative_to).as_posix()
        except (OSError, ValueError) as exc:
            raise ResultValidationError(f"{label} entry could not be frozen: {path}") from exc
        records.append({"path": relative, "sha256": snapshot.sha256, "size": snapshot.size})
    return tuple(records)


_SHA256 = re.compile("^[0-9a-f]{64}$")

_IDENTITY_KEYS = {"schema_version", "distribution", "version", "sha256", "files", "bytes"}


def canonical_distribution_name(value: str | None) -> str | None:
    """Normalize a Python distribution name using the packaging rule."""
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        raise ConfigurationError("distribution name must be non-empty text")
    return re.sub("[-_.]+", "-", value.strip()).lower()


def _identity_text(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ConfigurationError(f"invalid {label}: {value!r}")
    return value


def _installed_environment_roots() -> tuple[Path, ...]:
    """Return existing interpreter-owned install directories."""
    candidates = list(sysconfig.get_paths().values())
    try:
        candidates.extend(site.getsitepackages())
    except (AttributeError, OSError):
        pass
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if type(candidate) is not str or not candidate:
            continue
        try:
            resolved = Path(candidate).expanduser().resolve(strict=True)
        except OSError:
            continue
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)
    if not roots:
        raise ConfigurationError("active Python environment roots are unavailable")
    return tuple(roots)


def _allowed_environment_roots(allowed_roots: tuple[Path, ...] | None) -> tuple[Path, ...]:
    values = _installed_environment_roots() if allowed_roots is None else allowed_roots
    if type(values) is not tuple or not values:
        raise ConfigurationError("distribution identity requires non-empty allowed roots")
    roots: list[Path] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Path):
            raise ConfigurationError("distribution identity roots must be pathlib.Path values")
        try:
            root = value.expanduser().resolve(strict=True)
        except OSError as exc:
            raise ConfigurationError("distribution identity root is unavailable") from exc
        if not root.is_dir():
            raise ConfigurationError("distribution identity root must be a directory")
        key = str(root)
        if key not in seen:
            seen.add(key)
            roots.append(root)
    return tuple(roots)


def _is_within_allowed_root(path: Path, allowed_roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve(strict=True)
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def distribution_file_identity_from_metadata(
    distribution: object, *, name: str, version: str, allowed_roots: tuple[Path, ...] | None = None
) -> dict[str, object]:
    """Hash every installed file declared by one distribution metadata object."""
    name = _identity_text(name, label="distribution name")
    version = _identity_text(version, label="distribution version")
    files = getattr(distribution, "files", None)
    locate_file = getattr(distribution, "locate_file", None)
    if files is None or not callable(locate_file):
        raise ConfigurationError(f"runtime distribution has no installed-file inventory: {name}")
    parent_traversal = any(".." in Path(str(item).replace("\\", "/")).parts for item in files)
    roots = (
        _allowed_environment_roots(allowed_roots)
        if allowed_roots is not None or parent_traversal
        else None
    )
    if distribution_inventory_uses_indirection(distribution, allowed_roots=roots):
        raise ConfigurationError(f"runtime distribution source is indirect or editable: {name}")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in sorted(files, key=lambda value: str(value)):
        relative = str(item).replace("\\", "/")
        relative_path = Path(relative)
        if not relative or "\x00" in relative or relative in seen or relative_path.is_absolute():
            raise ConfigurationError(f"runtime distribution file inventory is invalid: {name}")
        seen.add(relative)
        path = Path(locate_file(item)).expanduser().absolute()
        if path.is_symlink():
            raise ConfigurationError("runtime distribution file must not be a symlink")
        if roots is not None:
            try:
                within_environment = _is_within_allowed_root(path, roots)
            except OSError as exc:
                raise ConfigurationError(
                    f"runtime distribution file identity is unavailable: {type(exc).__name__}"
                ) from exc
            if not within_environment:
                raise ConfigurationError(
                    "runtime distribution file is outside the active Python environment"
                )
        try:
            snapshot = snapshot_regular_file(path)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"runtime distribution file identity is unavailable: {type(exc).__name__}"
            ) from exc
        rows.append({"path": relative, "sha256": snapshot.sha256, "size": snapshot.size})
    if not rows:
        raise ConfigurationError(f"runtime distribution file inventory is empty: {name}")
    return {
        "schema_version": 1,
        "distribution": name,
        "version": version,
        "sha256": sha256_bytes(canonical_json_bytes(rows)),
        "files": len(rows),
        "bytes": sum(int(row["size"]) for row in rows),
    }


def distribution_inventory_uses_indirection(
    distribution: object, *, allowed_roots: tuple[Path, ...] | None = None
) -> bool:
    """Detect installed inventories that do not bind their imported source."""
    files = getattr(distribution, "files", None)
    locate_file = getattr(distribution, "locate_file", None)
    if files is None or not callable(locate_file):
        return True
    if allowed_roots is None:
        roots = None
    else:
        try:
            roots = _allowed_environment_roots(allowed_roots)
        except ConfigurationError:
            return True
    for item in files:
        relative = str(item).replace("\\", "/")
        lowered = relative.lower()
        if lowered.endswith((".pth", ".egg-link")):
            return True
        if lowered.endswith("direct_url.json"):
            try:
                path = Path(locate_file(item)).expanduser().absolute()
                if path.is_symlink() or (
                    roots is not None and (not _is_within_allowed_root(path, roots))
                ):
                    return True
                snapshot = snapshot_regular_file(path)
                direct_url = json.loads(snapshot.payload)
            except (OSError, TypeError, ValueError, json.JSONDecodeError, UnicodeError):
                return True
            if not isinstance(direct_url, dict):
                return True
            directory = direct_url.get("dir_info")
            if isinstance(directory, dict) and directory.get("editable") is True:
                return True
    return False


def validate_distribution_file_identity(
    value: object, *, name: str | None = None, version: str | None = None
) -> Mapping[str, object]:
    """Validate and deeply freeze one exact public distribution identity."""
    if not isinstance(value, Mapping):
        raise ConfigurationError("distribution identity must be a mapping")
    try:
        frozen = freeze_json_mapping(value)
        public = thaw_json_mapping(frozen)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConfigurationError("distribution identity must be finite canonical JSON") from exc
    if set(public) != _IDENTITY_KEYS:
        raise ConfigurationError("distribution identity schema is invalid")
    observed_name = _identity_text(public.get("distribution"), label="distribution identity name")
    observed_version = _identity_text(public.get("version"), label="distribution identity version")
    if name is not None and observed_name != name:
        raise ConfigurationError("distribution identity name does not match")
    if version is not None and observed_version != version:
        raise ConfigurationError("distribution identity version does not match")
    if public.get("schema_version") != 1:
        raise ConfigurationError("distribution identity schema version is invalid")
    digest = public.get("sha256")
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        raise ConfigurationError("distribution identity SHA-256 is invalid")
    if type(public.get("files")) is not int or public["files"] < 1:
        raise ConfigurationError("distribution identity file count is invalid")
    if type(public.get("bytes")) is not int or public["bytes"] < 0:
        raise ConfigurationError("distribution identity byte count is invalid")
    return frozen


def observe_distribution_file_identity(
    distribution: object | None, *, name: str | None, version: str | None
) -> dict[str, object]:
    """Return current identity or a portable unavailable observation."""
    if distribution is None or name is None or version is None:
        return {
            "status": "RETHINKSKILL_DISTRIBUTION_IDENTITY_UNAVAILABLE",
            "distribution": name,
            "version": version,
        }
    try:
        metadata = getattr(distribution, "metadata", None)
        current_name = metadata.get("Name") if metadata is not None else None
        current_version = getattr(distribution, "version", None)
        if type(current_name) is not str or type(current_version) is not str:
            raise ConfigurationError("runtime distribution name or version is unavailable")
        return distribution_file_identity_from_metadata(
            distribution, name=current_name, version=current_version
        )
    except Exception as exc:
        return {
            "status": "RETHINKSKILL_DISTRIBUTION_IDENTITY_UNAVAILABLE",
            "distribution": name,
            "version": version,
            "failure_type": type(exc).__name__,
        }


def freeze_mapping_manifest(component: Any, *, method_name: str, label: str) -> dict[str, object]:
    """Call one manifest method once and detach one JSON-safe snapshot."""
    try:
        method = getattr(component, method_name)
        if not callable(method):
            raise TypeError(f"{method_name} is not callable")
        value = method()
        if not isinstance(value, Mapping):
            raise TypeError(f"{method_name} did not return a mapping")
        return thaw_json_mapping(freeze_json_mapping(value))
    except Exception as exc:
        raise ConfigurationError(f"{label} manifest is unavailable: {type(exc).__name__}") from exc


def freeze_component_manifest(component: Any, *, label: str) -> dict[str, object]:
    """Call ``public_manifest`` once and detach one JSON-safe snapshot."""
    try:
        return freeze_mapping_manifest(
            component, method_name="public_manifest", label=f"{label} public"
        )
    except ConfigurationError as exc:
        raise ConfigurationError(
            f"{label} public manifest is unavailable: {(type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__)}"
        ) from exc
