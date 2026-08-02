"""RethinkSkill runtime tasks."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.utils.fs import is_within
from rethinkskill.utils.serde import (
    RegularFileSnapshot,
    atomic_copy,
    atomic_write,
    canonical_json_bytes,
    snapshot_regular_file,
    strict_json_loads,
    thaw_json_mapping,
)


def resolve_asset_below_root(root: Path, value: str, *, label: str) -> Path:
    """Resolve an explicit path or basename fallback without escaping root."""
    if not isinstance(root, Path):
        raise ResultValidationError("asset root must be a pathlib.Path")
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ResultValidationError(f"{label} path must be non-empty NUL-free text")
    if type(label) is not str or not label.strip():
        raise ResultValidationError("asset label must be non-empty text")
    resolved_root = root.expanduser().resolve()
    requested = Path(value)
    candidates = (
        requested if requested.is_absolute() else resolved_root / requested,
        resolved_root / requested.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if not resolved.is_relative_to(resolved_root):
                raise ResultValidationError(f"{label} escapes asset root: {value}")
            return resolved
    raise ResultValidationError(f"{label} is absent below asset root: {value}")


_SPLIT_ALIASES = {
    "train": "train",
    "valid_seen": "val",
    "selection": "val",
    "val": "val",
    "valid_unseen": "test",
    "test": "test",
}


def _absolute(path: Path) -> Path:
    """Normalize spelling without erasing a final symlink boundary."""
    return path.expanduser().absolute()


def _is_dataset_file(path: Path) -> bool:
    return (
        path.is_file()
        and (not path.name.startswith("."))
        and (path.suffix.lower() in {".json", ".jsonl"})
    )


def _split_roots(source: Path, split: str) -> tuple[Path, ...]:
    resolved = _absolute(source)
    if resolved.is_symlink():
        raise ResultValidationError(f"native dataset source must not be a symlink: {resolved}")
    if resolved.is_file():
        return (resolved.parent,)
    if not resolved.is_dir():
        raise ResultValidationError(f"native dataset source is absent: {resolved}")
    if split == "all":
        return tuple(
            resolved / name for name in ("train", "val", "test") if (resolved / name).is_dir()
        ) or (resolved,)
    directory = _SPLIT_ALIASES.get(split)
    if directory is None:
        raise ConfigurationError(f"unsupported native dataset split: {split}")
    candidate = resolved / directory
    return (candidate,) if candidate.is_dir() else (resolved,)


def _dataset_files(source: Path, split: str) -> tuple[Path, ...]:
    resolved = _absolute(source)
    if resolved.is_symlink():
        raise ResultValidationError(f"native dataset source must not be a symlink: {resolved}")
    if resolved.is_file():
        if not _is_dataset_file(resolved):
            raise ResultValidationError(
                f"native dataset file must be visible JSON/JSONL: {resolved}"
            )
        return (resolved,)
    roots = _split_roots(resolved, split)
    files = tuple(
        sorted(path for root in roots for path in root.iterdir() if _is_dataset_file(path))
    )
    if not files:
        raise ResultValidationError(
            f"native dataset contains no JSON/JSONL files: {resolved} split={split}"
        )
    return files


def dataset_tree_files(source: Path) -> tuple[Path, ...]:
    """Return all visible JSON shards that can affect native selection."""
    resolved = _absolute(source)
    if resolved.is_symlink():
        raise ResultValidationError(f"native dataset source must not be a symlink: {resolved}")
    if resolved.is_file():
        if not _is_dataset_file(resolved):
            raise ResultValidationError(
                f"native dataset file must be visible JSON/JSONL: {resolved}"
            )
        return (resolved,)
    if not resolved.is_dir():
        raise ResultValidationError(f"native dataset source is absent: {resolved}")
    files = tuple(
        sorted(
            path
            for path in resolved.rglob("*")
            if _is_dataset_file(path)
            and (not any(part.startswith(".") for part in path.relative_to(resolved).parts))
        )
    )
    if not files:
        raise ResultValidationError(
            f"native dataset contains no visible JSON/JSONL files: {resolved}"
        )
    return files


def dataset_files(source: Path, split: str) -> tuple[Path, ...]:
    """Public extension API for deterministic dataset-shard discovery."""
    return _dataset_files(source, split)


_RESERVED_WORKSPACE_FILES = {"codex_last_message.txt", "task.md"}

_RESERVED_WORKSPACE_ROOTS = {
    ".agents",
    ".claude",
    ".codex",
    ".gemini",
    ".git",
    ".rethinkskill",
    "steps",
}


def portable_asset_parts(target: str) -> tuple[str, ...]:
    """Normalize a portable target for collision checks."""
    return tuple(
        unicodedata.normalize("NFC", part).casefold() for part in PurePosixPath(target).parts
    )


@dataclass(frozen=True, slots=True)
class NativeAsset:
    """One hash-bound immutable input materialized into a task workspace."""

    source: Path
    target: str
    media_type: str
    role: str
    sha256: str
    visibility: str = "workspace"

    @classmethod
    def freeze(
        cls, source: Path, *, target: str, media_type: str, role: str, visibility: str = "workspace"
    ) -> NativeAsset:
        asset, _ = cls.freeze_with_snapshot(
            source, target=target, media_type=media_type, role=role, visibility=visibility
        )
        return asset

    @classmethod
    def freeze_with_snapshot(
        cls, source: Path, *, target: str, media_type: str, role: str, visibility: str = "workspace"
    ) -> tuple[NativeAsset, RegularFileSnapshot]:
        """Freeze one asset and return the exact bytes behind its digest."""
        resolved = source.expanduser().absolute()
        try:
            snapshot = snapshot_regular_file(resolved)
        except (OSError, ValueError) as exc:
            raise ResultValidationError(
                f"native task asset must be a regular non-symlink file: {resolved}"
            ) from exc
        asset = cls(
            source=resolved,
            target=target,
            media_type=media_type,
            role=role,
            sha256=snapshot.sha256,
            visibility=visibility,
        )
        asset._validate_fields()
        return (asset, snapshot)

    def _validate_fields(self) -> None:
        if not isinstance(self.source, Path):
            raise ResultValidationError(f"native task asset source must be a path: {self.source}")
        if type(self.target) is not str:
            raise ResultValidationError(f"native task asset target is unsafe: {self.target!r}")
        target = PurePosixPath(self.target)
        try:
            target_bytes = self.target.encode("utf-8")
            part_lengths = tuple(len(part.encode("utf-8")) for part in target.parts)
        except UnicodeEncodeError as exc:
            raise ResultValidationError(
                f"native task asset target is unsafe: {self.target!r}"
            ) from exc
        if (
            target.is_absolute()
            or not target.parts
            or any(part in {"", ".", ".."} for part in target.parts)
            or ("\\" in self.target)
            or any(ord(character) < 32 or ord(character) == 127 for character in self.target)
            or (len(target_bytes) > 1024)
            or any(length > 240 for length in part_lengths)
        ):
            raise ResultValidationError(f"native task asset target is unsafe: {self.target!r}")
        normalized = portable_asset_parts(self.target)
        if self.visibility == "workspace" and (
            normalized[0] in _RESERVED_WORKSPACE_ROOTS
            or "/".join(normalized) in _RESERVED_WORKSPACE_FILES
        ):
            raise ResultValidationError(
                f"native task asset target collides with executor control files: {self.target!r}"
            )
        if (
            type(self.media_type) is not str
            or not self.media_type
            or type(self.role) is not str
            or (not self.role)
        ):
            raise ResultValidationError("native task asset media_type and role are required")
        if type(self.visibility) is not str or self.visibility not in {"workspace", "verifier"}:
            raise ResultValidationError(
                "native task asset visibility must be workspace or verifier"
            )
        if type(self.sha256) is not str:
            raise ResultValidationError("native task asset sha256 must be text")

    def validate(self) -> None:
        self._validate_fields()
        try:
            observed = snapshot_regular_file(self.source).sha256
        except (OSError, ValueError) as exc:
            raise ResultValidationError(
                f"native task asset cannot be frozen: {self.source}"
            ) from exc
        if observed != self.sha256:
            raise ResultValidationError(f"native task asset hash drift: {self.source}")

    def public(self) -> dict[str, str]:
        return {
            "source": str(self.source),
            "target": self.target,
            "media_type": self.media_type,
            "role": self.role,
            "sha256": self.sha256,
            "visibility": self.visibility,
        }


_MAX_CONTEXT_CHARS = 6000


def _text(record: Mapping[str, Any], key: str, *, default: str | None = None) -> str:
    value = record.get(key, default)
    if type(value) is not str:
        raise ResultValidationError(f"dataset field {key!r} must be text")
    return value


def _aliases(record: Mapping[str, Any]) -> tuple[str, ...]:
    value = record.get("answers", record.get("gold_aliases"))
    if type(value) is str:
        return (value,)
    if type(value) is list and all(type(item) is str for item in value):
        return tuple(value)
    raise ResultValidationError("dataset field 'answers' must be text or a text list")


def _truncate_context(context: str) -> str:
    if len(context) <= _MAX_CONTEXT_CHARS:
        return context
    chunks = context.split("[DOC]")
    selected = ""
    for chunk in chunks:
        candidate = selected + "[DOC]" + chunk if selected else chunk
        if len(candidate) > _MAX_CONTEXT_CHARS:
            break
        selected = candidate
    return selected or context[:_MAX_CONTEXT_CHARS] + "\n...[truncated]"


def _first_field(record: Mapping[str, Any], keys: Sequence[str], *, field_role: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    raise ResultValidationError(
        f"dataset record is missing {field_role}; expected one of {tuple(keys)}"
    )


_ACTIVE_DATASET_SNAPSHOTS: ContextVar[Mapping[Path, RegularFileSnapshot] | None] = ContextVar(
    "rethinkskill_dataset_snapshots", default=None
)


def freeze_dataset_files(paths: tuple[Path, ...]) -> Mapping[Path, RegularFileSnapshot]:
    """Read each dataset shard once and retain matching bytes and digests."""
    if type(paths) is not tuple or not paths:
        raise ResultValidationError("dataset snapshot requires a non-empty tuple of paths")
    snapshots: dict[Path, RegularFileSnapshot] = {}
    for raw_path in paths:
        if not isinstance(raw_path, Path):
            raise ResultValidationError("dataset snapshot paths must be exact pathlib.Path values")
        path = _absolute(raw_path)
        if path in snapshots:
            raise ResultValidationError(f"duplicate dataset snapshot path: {path}")
        try:
            snapshots[path] = snapshot_regular_file(path)
        except (OSError, ValueError) as exc:
            raise ResultValidationError(
                f"dataset shard cannot be frozen as a regular file: {path}"
            ) from exc
    return MappingProxyType(snapshots)


@contextmanager
def bound_dataset_snapshots(
    paths: tuple[Path, ...],
) -> Iterator[Mapping[Path, RegularFileSnapshot]]:
    """Bind normalized loaders to one immutable set of dataset reads."""
    snapshots = freeze_dataset_files(paths)
    token = _ACTIVE_DATASET_SNAPSHOTS.set(snapshots)
    try:
        yield snapshots
    finally:
        _ACTIVE_DATASET_SNAPSHOTS.reset(token)


def _dataset_snapshot(path: Path) -> RegularFileSnapshot:
    normalized = _absolute(path)
    active = _ACTIVE_DATASET_SNAPSHOTS.get()
    if active is not None:
        try:
            return active[normalized]
        except KeyError as exc:
            raise ResultValidationError(
                f"dataset loader accessed a shard outside the bound provenance snapshot: {normalized}"
            ) from exc
    try:
        return snapshot_regular_file(normalized)
    except (OSError, ValueError) as exc:
        raise ResultValidationError(
            f"dataset shard cannot be read as a regular file: {normalized}"
        ) from exc


def load_dataset_text(path: Path) -> str:
    """Decode one bound dataset snapshot as UTF-8 text."""
    snapshot = _dataset_snapshot(path)
    try:
        return snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResultValidationError(
            f"dataset shard must be valid UTF-8: {_absolute(path)}"
        ) from exc


_SAFE_WORKSPACE_ID = re.compile("^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class NativeTask:
    """One immutable benchmark case after dataset normalization."""

    task_id: str
    payload: Mapping[str, Any]
    assets: tuple[NativeAsset, ...] = ()

    def validate(self) -> None:
        if (
            type(self.task_id) is not str
            or not self.task_id
            or len(self.task_id) > 512
            or any(ord(character) < 32 or ord(character) == 127 for character in self.task_id)
        ):
            raise ResultValidationError(f"invalid native task id: {self.task_id!r}")
        try:
            self.task_id.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ResultValidationError("native task id must be valid UTF-8 text") from exc
        if not isinstance(self.payload, Mapping):
            raise ResultValidationError("native task payload must be a mapping")
        try:
            canonical_json_bytes(thaw_json_mapping(self.payload))
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise ResultValidationError(
                "native task payload must be finite canonical JSON"
            ) from exc
        if type(self.assets) is not tuple or not all(
            type(asset) is NativeAsset for asset in self.assets
        ):
            raise ResultValidationError(
                "native task assets must be a tuple of exact NativeAsset values"
            )
        targets: list[tuple[str, tuple[str, ...]]] = []
        for asset in self.assets:
            asset.validate()
            parts = portable_asset_parts(asset.target)
            for visibility, observed in targets:
                if visibility != asset.visibility:
                    continue
                shared = min(len(parts), len(observed))
                if parts[:shared] == observed[:shared]:
                    raise ResultValidationError(
                        f"colliding native task asset targets: {asset.visibility}:{asset.target}"
                    )
            targets.append((asset.visibility, parts))

    @property
    def workspace_name(self) -> str:
        """Return a portable task directory without changing the evidence ID."""
        if (
            _SAFE_WORKSPACE_ID.fullmatch(self.task_id) is not None
            and ".." not in self.task_id
            and (len(self.task_id.encode("utf-8")) <= 120)
        ):
            return self.task_id
        prefix = re.sub("[^A-Za-z0-9._-]+", "-", self.task_id).strip(".-")[:48] or "task"
        digest = sha256(self.task_id.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}-{digest}"


def _load_json_records(path: Path) -> list[Mapping[str, Any]]:
    text = load_dataset_text(path).strip()
    if not text:
        return []
    try:
        value = strict_json_loads(text)
    except json.JSONDecodeError:
        value = None
    if type(value) is list:
        records = value
    elif type(value) is dict:
        nested = value.get("data", value.get("items"))
        if type(nested) is list:
            records = nested
        elif "id" in value or "case_id" in value:
            records = [value]
        else:
            records = list(value.values())
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(strict_json_loads(line))
            except json.JSONDecodeError as exc:
                raise ResultValidationError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
    if not all(type(record) is dict for record in records):
        raise ResultValidationError(f"dataset records must be JSON objects: {path}")
    return records


def _dataset_records(source: Path, split: str) -> list[Mapping[str, Any]]:
    """Load every normalized record from the selected JSON shards."""
    return [record for path in _dataset_files(source, split) for record in _load_json_records(path)]


def load_json_records(path: Path) -> list[Mapping[str, Any]]:
    """Public extension API for normalized JSON/JSONL record loading."""
    return _load_json_records(path)


@dataclass(frozen=True, slots=True)
class RenderedTask:
    """Task-local files and the short invocation sent to an executor."""

    task_markdown: str
    skill_markdown: str
    invocation: str
    attachments: tuple[NativeAsset, ...] = ()

    def validate(self) -> None:
        if type(self.task_markdown) is not str or not self.task_markdown.strip():
            raise ResultValidationError("rendered task.md is empty")
        if type(self.skill_markdown) is not str or not self.skill_markdown.strip():
            raise ResultValidationError("rendered SKILL.md is empty")
        if type(self.invocation) is not str or not self.invocation.strip():
            raise ResultValidationError("native executor invocation is empty")
        if type(self.attachments) is not tuple or not all(
            type(attachment) is NativeAsset for attachment in self.attachments
        ):
            raise ResultValidationError(
                "rendered attachments must be a tuple of exact NativeAsset values"
            )
        targets: list[tuple[str, ...]] = []
        for attachment in self.attachments:
            attachment.validate()
            if attachment.visibility != "workspace":
                raise ResultValidationError("rendered attachments must be workspace-visible")
            target = portable_asset_parts(attachment.target)
            for observed in targets:
                shared = min(len(target), len(observed))
                if target[:shared] == observed[:shared]:
                    raise ResultValidationError(
                        f"colliding rendered attachment target: {attachment.target}"
                    )
            targets.append(target)

    def validate_for(self, task: NativeTask) -> None:
        """Validate that every exposed attachment belongs to this task."""
        task.validate()
        self.validate()
        available = frozenset(asset for asset in task.assets if asset.visibility == "workspace")
        for attachment in self.attachments:
            if attachment not in available:
                raise ResultValidationError(
                    f"rendered attachment is not a frozen workspace asset for task {task.task_id!r}: {attachment.target!r}"
                )


def _select_tasks(
    tasks: Sequence[NativeTask], *, limit: int | None, requested_ids: Sequence[str]
) -> tuple[NativeTask, ...]:
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ConfigurationError("native task limit must be a positive integer")
    if limit is not None and requested_ids:
        raise ConfigurationError("native task limit and requested IDs are mutually exclusive")
    observed: set[str] = set()
    workspaces: dict[str, str] = {}
    for task in tasks:
        if type(task) is not NativeTask:
            raise ResultValidationError("native task selection requires exact NativeTask values")
        task.validate()
        if task.task_id in observed:
            raise ResultValidationError(f"duplicate native task id: {task.task_id}")
        observed.add(task.task_id)
        workspace = task.workspace_name.casefold()
        if workspace in workspaces:
            raise ResultValidationError(
                f"native task workspace collision: {workspaces[workspace]!r}, {task.task_id!r}"
            )
        workspaces[workspace] = task.task_id
    selected = list(tasks)
    if requested_ids:
        if len(requested_ids) != len(set(requested_ids)):
            raise ResultValidationError("duplicate requested native task ids")
        by_id = {task.task_id: task for task in tasks}
        missing = [task_id for task_id in requested_ids if task_id not in by_id]
        if missing:
            raise ResultValidationError(f"requested native task ids are absent: {missing[:20]}")
        selected = [by_id[task_id] for task_id in requested_ids]
    elif limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ResultValidationError("native task selection is empty")
    return tuple(selected)


def select_tasks(
    tasks: Sequence[NativeTask], *, limit: int | None, requested_ids: Sequence[str]
) -> tuple[NativeTask, ...]:
    """Public extension API for fail-closed task-ID selection."""
    return _select_tasks(tasks, limit=limit, requested_ids=requested_ids)


_SKILL_NAME = re.compile("^[a-z0-9][a-z0-9-]{0,63}$")


def _new_destination(root: Path, relative: PurePosixPath) -> Path:
    if root.is_symlink():
        raise ResultValidationError(f"workspace root cannot be a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    destination = root.joinpath(*relative.parts)
    if destination.exists() or destination.is_symlink():
        raise ResultValidationError(f"workspace destination already exists: {destination}")
    resolved_destination = destination.resolve()
    if not is_within(resolved_destination, resolved_root):
        raise ResultValidationError(f"workspace destination escapes its root: {destination}")
    for parent in destination.parents:
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ResultValidationError(f"workspace destination traverses a symlink: {destination}")
        if parent == root:
            break
    return destination


def materialize_rendered_workspace(
    rendered: RenderedTask, workspace: Path, *, skill_name: str
) -> None:
    """Write executor control files exactly once."""
    if type(rendered) is not RenderedTask:
        raise ResultValidationError("workspace materialization requires an exact RenderedTask")
    rendered.validate()
    if _SKILL_NAME.fullmatch(skill_name) is None:
        raise ResultValidationError(f"invalid workspace skill name: {skill_name!r}")
    task_path = _new_destination(workspace, PurePosixPath("task.md"))
    skill_path = _new_destination(workspace, PurePosixPath(f".agents/skills/{skill_name}/SKILL.md"))
    atomic_write(task_path, rendered.task_markdown.encode("utf-8"))
    atomic_write(skill_path, rendered.skill_markdown.encode("utf-8"))


def materialize_task_assets(task: NativeTask, *, workspace: Path, verifier_workspace: Path) -> None:
    """Copy every hash-bound task asset into its isolated namespace."""
    if type(task) is not NativeTask:
        raise ResultValidationError("asset materialization requires an exact NativeTask")
    task.validate()
    for asset in task.assets:
        root = workspace if asset.visibility == "workspace" else verifier_workspace
        destination = _new_destination(root, PurePosixPath(asset.target))
        try:
            atomic_copy(asset.source, destination, expected_sha256=asset.sha256)
        except ValueError as exc:
            raise ResultValidationError(f"native task asset hash drift: {asset.source}") from exc
