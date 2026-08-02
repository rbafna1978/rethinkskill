"""Python 3.8-compatible installed-distribution identity probe.

This file is executed by the independently installed WebShop interpreter.  It
intentionally uses only the standard library and emits one bounded JSON value.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import re
import site
import stat
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

_BUILTIN_EXCEPTION_NAMES = {
    value: value.__name__
    for value in vars(builtins).values()
    if isinstance(value, type) and issubclass(value, BaseException)
}
_NORMALIZE_NAME = re.compile(r"[-_.]+")


def _safe_exception_type(exc: BaseException) -> str:
    for exception_type in type(exc).__mro__:
        name = _BUILTIN_EXCEPTION_NAMES.get(exception_type)
        if name is not None:
            return name
    return "Exception"


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _snapshot_regular_file(path: Path) -> tuple[str, int, bytes]:
    source = path.expanduser().absolute()
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow and source.is_symlink():
        raise ValueError("distribution file must not be a symlink")
    descriptor = os.open(str(source), os.O_RDONLY | nofollow)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("distribution path must be a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if before_identity != after_identity or len(payload) != after.st_size:
        raise ValueError("distribution file changed while reading")
    return _sha256_bytes(payload), len(payload), payload


def _identity_text(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("invalid " + label)
    return value


def _is_within_allowed_root(path: Path, allowed_roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve(strict=True)
    for root in allowed_roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _validate_pth_paths(
    path: Path,
    payload: bytes,
    allowed_roots: tuple[Path, ...],
) -> None:
    try:
        value = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("distribution .pth file is not UTF-8") from exc
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith(("import ", "import\t")):
            continue
        target = Path(line).expanduser()
        if not target.is_absolute():
            target = path.parent / target
        try:
            within_environment = _is_within_allowed_root(
                target,
                allowed_roots,
            )
        except OSError as exc:
            raise ValueError("distribution .pth source is indirect") from exc
        if not within_environment:
            raise ValueError("distribution .pth source is indirect")


def _distribution_identity(
    distribution: object,
    *,
    allowed_roots: tuple[Path, ...],
) -> dict[str, object]:
    distribution_metadata = getattr(distribution, "metadata", None)
    raw_name = distribution_metadata.get("Name") if distribution_metadata is not None else None
    name = _identity_text(raw_name, label="distribution name")
    normalized_name = _NORMALIZE_NAME.sub("-", name).lower()
    version = _identity_text(
        getattr(distribution, "version", None),
        label="distribution version",
    )
    files = getattr(distribution, "files", None)
    locate_file = getattr(distribution, "locate_file", None)
    if files is None or not callable(locate_file):
        raise ValueError("distribution has no installed-file inventory")

    rows_by_path = {}
    resolved_locations = {}
    for item in sorted(files, key=lambda value: str(value)):
        relative = str(item).replace("\\", "/")
        relative_path = Path(relative)
        if not relative or "\x00" in relative or relative_path.is_absolute():
            raise ValueError("distribution file inventory is invalid")
        lowered = relative.lower()
        if lowered.endswith(".egg-link"):
            raise ValueError("distribution source is indirect")
        path = Path(locate_file(item)).expanduser().absolute()
        if path.is_symlink():
            raise ValueError("distribution file must not be a symlink")
        if not _is_within_allowed_root(path, allowed_roots):
            raise ValueError("distribution file is outside the Python environment")
        digest, size, payload = _snapshot_regular_file(path)
        if lowered.endswith(".pth"):
            _validate_pth_paths(path, payload, allowed_roots)
        if lowered.endswith("direct_url.json"):
            try:
                direct_url = json.loads(payload.decode("utf-8"))
            except (UnicodeError, ValueError) as exc:
                raise ValueError("distribution direct URL is invalid") from exc
            if not isinstance(direct_url, dict):
                raise ValueError("distribution direct URL is invalid")
            directory = direct_url.get("dir_info")
            if isinstance(directory, dict) and directory.get("editable") is True:
                raise ValueError("distribution source is editable")
        row = {
            "path": relative,
            "sha256": digest,
            "size": size,
        }
        resolved_location = str(path.resolve(strict=True))
        previous = rows_by_path.get(relative)
        if previous is not None:
            if previous != row or resolved_locations[relative] != resolved_location:
                raise ValueError("duplicate distribution file inventory is inconsistent")
            continue
        rows_by_path[relative] = row
        resolved_locations[relative] = resolved_location
    rows = list(rows_by_path.values())
    if not rows:
        raise ValueError("distribution file inventory is empty")
    return {
        "distribution": name,
        "normalized_name": normalized_name,
        "version": version,
        "sha256": _sha256_bytes(_canonical_json_bytes(rows)),
        "files": len(rows),
        "bytes": sum(row["size"] for row in rows),
    }


def _installed_environment_roots() -> tuple[Path, ...]:
    candidates = list(sysconfig.get_paths().values())
    candidates.extend(site.getsitepackages())
    roots = []
    seen = set()
    for candidate in candidates:
        if type(candidate) is not str or not candidate:
            continue
        path = Path(candidate).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            continue
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)
    if not roots:
        raise ValueError("Python environment roots are unavailable")
    return tuple(roots)


def _environment_identity(
    distributions: object = None,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
) -> dict[str, object]:
    installed = metadata.distributions() if distributions is None else distributions
    roots = (
        _installed_environment_roots()
        if allowed_roots is None
        else tuple(path.resolve(strict=True) for path in allowed_roots)
    )
    if not roots:
        raise ValueError("Python environment roots are unavailable")
    rows = [_distribution_identity(distribution, allowed_roots=roots) for distribution in installed]
    rows.sort(
        key=lambda row: (
            row["normalized_name"],
            row["version"],
            row["distribution"],
        )
    )
    if not rows:
        raise ValueError("Python environment has no installed distributions")
    normalized_names = [row["normalized_name"] for row in rows]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("Python environment has duplicate distributions")
    implementation = getattr(sys, "implementation", None)
    implementation_name = getattr(implementation, "name", None)
    cache_tag = getattr(implementation, "cache_tag", None)
    if type(implementation_name) is not str or not implementation_name:
        raise ValueError("Python implementation identity is unavailable")
    if cache_tag is not None and type(cache_tag) is not str:
        raise ValueError("Python cache tag is invalid")
    return {
        "schema_version": 1,
        "kind": "python_distribution_environment",
        "python": {
            "implementation": implementation_name,
            "version": [
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            ],
            "cache_tag": cache_tag,
        },
        "sha256": _sha256_bytes(_canonical_json_bytes(rows)),
        "distributions": rows,
        "distribution_count": len(rows),
        "files": sum(row["files"] for row in rows),
        "bytes": sum(row["bytes"] for row in rows),
    }


def main() -> int:
    try:
        identity = _environment_identity()
        value = {"status": "ok", "identity": identity}
        code = 0
    except Exception as exc:  # noqa: BLE001
        value = {
            "status": "error",
            "error_code": "environment_identity_unavailable",
            "error_type": _safe_exception_type(exc),
        }
        code = 1
    sys.stdout.write(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    sys.stdout.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
