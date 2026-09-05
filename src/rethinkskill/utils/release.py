"""Small helpers for release metadata and artifact inventories."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.utils.serde import freeze_json_mapping, snapshot_regular_file, thaw_json_mapping


def regular_file_inventory(
    root: Path, *, relative_to: Path, excluded: Iterable[Path] = (), label: str
) -> tuple[dict[str, object], ...]:
    """List regular artifacts and bind each external file by content hash."""
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


def canonical_distribution_name(value: str | None) -> str | None:
    """Normalize a Python distribution name using the packaging rule."""
    if value is None:
        return None
    if type(value) is not str or not value.strip():
        raise ConfigurationError("distribution name must be non-empty text")
    return re.sub("[-_.]+", "-", value.strip()).lower()


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
        cause = type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__
        raise ConfigurationError(f"{label} public manifest is unavailable: {cause}") from exc
