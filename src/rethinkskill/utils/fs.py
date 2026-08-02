"""RethinkSkill utils fs."""

from __future__ import annotations

import fcntl
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from rethinkskill.errors import ConfigurationError, ProtectedPathError


class OutputLock:
    def __init__(self, output_root: Path):
        resolved = output_root.expanduser().resolve()
        self.path = resolved.parent / f".{resolved.name}.rethinkskill.lock"
        self.handle: TextIO | None = None

    def __enter__(self) -> OutputLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = -1
        try:
            if self.path.is_symlink():
                raise ConfigurationError(f"output lock cannot be a symlink: {self.path}")
            flags = (
                os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(self.path, flags, 384)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ConfigurationError(f"output lock must be one regular file: {self.path}")
            os.fchmod(descriptor, 384)
            self.handle = os.fdopen(descriptor, "r+", encoding="utf-8")
            descriptor = -1
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._close()
            raise ConfigurationError(f"another runner owns {self.path}") from exc
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            self._close()
            raise
        try:
            self.handle.seek(0)
            self.handle.truncate()
            started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            self.handle.write(f"pid={os.getpid()} started_at={started_at}\n")
            self.handle.flush()
            os.fsync(self.handle.fileno())
        except Exception:
            self._close()
            raise
        return self

    def _close(self) -> None:
        handle = self.handle
        self.handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __exit__(self, *_: object) -> None:
        self._close()


REPOSITORY_MARKERS = ("pyproject.toml",)

PROTECTED_TOP_LEVEL = ("final_results", "manuscript", "archive", "research", ".local_secrets")


def is_within(path: Path, root: Path) -> bool:
    resolved_path = path.expanduser().resolve()
    resolved_root = root.expanduser().resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class Repository:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> Repository:
        candidate = (start or Path.cwd()).resolve()
        for current in (candidate, *candidate.parents):
            if all((current / marker).is_file() for marker in REPOSITORY_MARKERS):
                return cls(current)
        raise ConfigurationError(
            f"unable to find rethinkskill repository from {candidate}; required markers={REPOSITORY_MARKERS}"
        )

    def resolve_relative(self, value: str, *, must_exist: bool = True) -> Path:
        relative = Path(value)
        if relative.is_absolute():
            raise ConfigurationError(f"registry path must be relative: {value}")
        path = (self.root / relative).resolve()
        if not is_within(path, self.root):
            raise ConfigurationError(f"registry path escapes repository: {value}")
        if must_exist and (not path.exists()):
            raise ConfigurationError(f"registry path does not exist: {value}")
        return path

    def validate_new_output(self, value: Path, run_root: Path) -> Path:
        repository_root = self.root.expanduser().resolve()
        raw_run_root = run_root.expanduser()
        resolved_run_root = (
            raw_run_root if raw_run_root.is_absolute() else repository_root / raw_run_root
        ).resolve()
        if resolved_run_root == repository_root or not is_within(
            resolved_run_root, repository_root
        ):
            raise ProtectedPathError(
                f"configured run root must be a dedicated repository subtree: {resolved_run_root}"
            )
        raw_output = value.expanduser()
        path = (raw_output if raw_output.is_absolute() else repository_root / raw_output).resolve()
        if path in {repository_root, Path("/"), resolved_run_root}:
            raise ProtectedPathError(f"unsafe output root: {path}")
        for name in PROTECTED_TOP_LEVEL:
            protected = (repository_root / name).resolve()
            if is_within(resolved_run_root, protected):
                raise ProtectedPathError(
                    f"configured run root enters protected tree {protected}: {resolved_run_root}"
                )
            if is_within(path, protected):
                raise ProtectedPathError(
                    f"new executions cannot write into protected tree {protected}: {path}"
                )
        if not is_within(path, resolved_run_root):
            raise ProtectedPathError(
                f"new output must be under configured run root {resolved_run_root}: {path}"
            )
        return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
