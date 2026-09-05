"""RethinkSkill utils serde."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any


def _validate_json_keys(value: Any) -> None:
    """Reject mapping keys that JSON would otherwise coerce to text."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if type(key) is not str:
                raise TypeError("canonical JSON object keys must be strings")
            _validate_json_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _validate_json_keys(nested)


def canonical_json_bytes(value: Any) -> bytes:
    _validate_json_keys(value)
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RegularFileSnapshot:
    """One immutable read of a regular file and its matching digest."""

    payload: bytes
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes:
            raise TypeError("regular-file snapshot payload must be exact bytes")
        if type(self.size) is not int or self.size != len(self.payload):
            raise ValueError("regular-file snapshot size does not match payload")
        if type(self.sha256) is not str or self.sha256 != sha256_bytes(self.payload):
            raise ValueError("regular-file snapshot digest does not match payload")


def snapshot_regular_file(path: Path) -> RegularFileSnapshot:
    """Read one regular file once without following a final symlink.

    Hashing and parsing evidence through the returned payload prevents a path
    replacement between separate read and digest operations from producing a
    mixed provenance claim.
    """
    source = path.expanduser().absolute()
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow and source.is_symlink():
        raise ValueError(f"snapshot source must not be a symlink: {source}")
    descriptor = os.open(source, os.O_RDONLY | nofollow)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"snapshot source must be a regular file: {source}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    final_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if identity != final_identity or len(payload) != after.st_size:
        raise ValueError(f"snapshot source changed while reading: {source}")
    return RegularFileSnapshot(payload=payload, sha256=sha256_bytes(payload), size=len(payload))


def _materialize_json(value: Any, active: set[int], memo: dict[int, Any]) -> Any:
    """Read every untrusted JSON container once into built-in containers."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise ValueError("circular JSON mapping")
        if marker in memo:
            return memo[marker]
        active.add(marker)
        try:
            try:
                items = tuple(value.items())
            except Exception as exc:
                raise TypeError(f"JSON mapping read failed: {type(exc).__name__}") from exc
            detached: dict[str, Any] = {}
            memo[marker] = detached
            for item in items:
                try:
                    key, nested = item
                except Exception as exc:
                    raise TypeError(f"JSON mapping item read failed: {type(exc).__name__}") from exc
                if type(key) is not str:
                    raise TypeError("JSON object keys must be exact strings")
                if key in detached:
                    raise TypeError(f"duplicate JSON object key: {key!r}")
                detached[key] = _materialize_json(nested, active, memo)
            return detached
        finally:
            active.remove(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise ValueError("circular JSON sequence")
        if marker in memo:
            return memo[marker]
        active.add(marker)
        try:
            try:
                items = tuple(value)
            except Exception as exc:
                raise TypeError(f"JSON sequence read failed: {type(exc).__name__}") from exc
            detached_list: list[Any] = []
            memo[marker] = detached_list
            detached_list.extend(_materialize_json(nested, active, memo) for nested in items)
            return detached_list
        finally:
            active.remove(marker)
    raise TypeError("JSON values must use an exact JSON scalar or a mapping/list/tuple")


def _freeze_detached(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_detached(nested) for key, nested in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_detached(nested) for nested in value)
    return value


def freeze_json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    """Read one mapping once, validate canonical JSON, and detach it deeply."""
    if not isinstance(value, Mapping):
        raise TypeError("JSON snapshot source must be a mapping")
    materialized = _materialize_json(value, set(), {})
    detached = json.loads(canonical_json_bytes(materialized).decode("utf-8"))
    if not isinstance(detached, dict):
        raise TypeError("JSON mapping snapshot must be an object")
    return _freeze_detached(detached)


def freeze_json_mapping_sequence(
    values: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Detach a sequence of mappings with one read of every source mapping."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError("JSON mapping snapshot source must be a sequence")
    materialized = _materialize_json(values, set(), {})
    if not isinstance(materialized, list) or not all(
        isinstance(value, dict) for value in materialized
    ):
        raise TypeError("JSON mapping sequence snapshot must contain objects")
    detached = json.loads(canonical_json_bytes(materialized).decode("utf-8"))
    if not isinstance(detached, list) or not all(isinstance(value, dict) for value in detached):
        raise TypeError("JSON mapping sequence snapshot must contain objects")
    return tuple(_freeze_detached(value) for value in detached)


def thaw_json_value(value: Any) -> Any:
    """Return a fresh mutable JSON value from an immutable snapshot."""
    if isinstance(value, Mapping):
        return {key: thaw_json_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(nested) for nested in value]
    if isinstance(value, list):
        return [thaw_json_value(nested) for nested in value]
    return value


def thaw_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    thawed = thaw_json_value(value)
    if not isinstance(thawed, dict):
        raise TypeError("frozen JSON mapping did not thaw to an object")
    return thawed


def _reject_json_constant(value: str) -> None:
    raise json.JSONDecodeError(f"non-standard JSON constant is forbidden: {value}", value, 0)


def _strict_json_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise json.JSONDecodeError("duplicate JSON object key is forbidden", "", 0)
        value[key] = nested
    return value


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Parse unambiguous standards-compliant JSON."""
    parsed = json.loads(
        value, parse_constant=_reject_json_constant, object_pairs_hook=_strict_json_object
    )
    try:
        canonical_json_bytes(parsed)
    except (TypeError, ValueError) as exc:
        raise json.JSONDecodeError(
            "parsed JSON contains a non-canonical value", str(value), 0
        ) from exc
    return parsed


def read_json(path: Path) -> Any:
    return strict_json_loads(path.read_text(encoding="utf-8"))


def read_json_snapshot(path: Path) -> tuple[RegularFileSnapshot, Any]:
    """Parse strict JSON from the same regular-file bytes used for hashing."""
    snapshot = snapshot_regular_file(path)
    return (snapshot, strict_json_loads(snapshot.payload))


def _fsync_directory(path: Path) -> None:
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory = os.open(path, directory_flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_write(path: Path, payload: bytes) -> None:
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def atomic_write_snapshot(path: Path, payload: bytes) -> RegularFileSnapshot:
    """Atomically publish exact bytes and return their provenance binding."""
    if type(payload) is not bytes:
        raise TypeError("atomic snapshot payload must be exact bytes")
    atomic_write(path, payload)
    return RegularFileSnapshot(payload=payload, sha256=sha256_bytes(payload), size=len(payload))


def atomic_copy(source: Path, destination: Path, *, expected_sha256: str) -> None:
    """Copy one frozen file through an exclusive, fsynced temporary."""
    source_path = source.expanduser().absolute()
    if source_path.is_symlink():
        raise ValueError(f"copy source must not be a symlink: {source_path}")
    target = destination.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp.", dir=target.parent)
    temporary = Path(temporary_name)
    observed = sha256()
    source_descriptor = -1
    try:
        source_descriptor = os.open(source_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        if not stat.S_ISREG(os.fstat(source_descriptor).st_mode):
            raise ValueError(f"copy source must be a regular file: {source_path}")
        with os.fdopen(source_descriptor, "rb") as reader, os.fdopen(descriptor, "wb") as writer:
            source_descriptor = -1
            descriptor = -1
            while chunk := reader.read(1024 * 1024):
                observed.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        digest = observed.hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"source content does not match frozen SHA-256: {source_path}")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(thaw_json_value(value)))


def atomic_write_json_snapshot(path: Path, value: Any) -> RegularFileSnapshot:
    """Canonically publish JSON and bind the receipt to those exact bytes."""
    payload = canonical_json_bytes(thaw_json_value(value))
    return atomic_write_snapshot(path, payload)
