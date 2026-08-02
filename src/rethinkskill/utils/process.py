"""RethinkSkill utils process."""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

_PROCESS_IDENTIFIER = re.compile("^[A-Za-z_][A-Za-z0-9_]*$")


def is_process_identifier(value: object) -> bool:
    """Return whether ``value`` is one exact process-safe identifier."""
    return type(value) is str and _PROCESS_IDENTIFIER.fullmatch(value) is not None


def _snapshot_argv(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("process argv must be a sequence")
    try:
        argv = tuple(value)
    except Exception as exc:
        raise ValueError(f"process argv read failed: {type(exc).__name__}") from exc
    if not argv or any(type(item) is not str or "\x00" in item for item in argv):
        raise ValueError("process argv must contain exact NUL-free strings")
    return argv


def snapshot_process_environment(value: object) -> Mapping[str, str]:
    """Read one environment mapping once into a detached read-only copy."""
    if not isinstance(value, Mapping):
        raise ValueError("process environment must be a mapping")
    try:
        items = tuple(value.items())
    except Exception as exc:
        raise ValueError(f"process environment read failed: {type(exc).__name__}") from exc
    environment: dict[str, str] = {}
    for item in items:
        try:
            name, nested = item
        except Exception as exc:
            raise ValueError(f"process environment item is invalid: {type(exc).__name__}") from exc
        if not is_process_identifier(name) or name in environment:
            raise ValueError("process environment names must be unique exact identifiers")
        if type(nested) is not str or "\x00" in nested:
            raise ValueError("process environment values must be exact NUL-free strings")
        environment[name] = nested
    return MappingProxyType(environment)


@dataclass(frozen=True, slots=True)
class ProcessInvocation:
    """The exact immutable process request passed to ``subprocess.Popen``."""

    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str] = field(repr=False)
    timeout_seconds: int

    def __post_init__(self) -> None:
        argv = _snapshot_argv(self.argv)
        if not isinstance(self.cwd, Path):
            raise ValueError("process cwd must be a pathlib.Path")
        try:
            cwd = self.cwd.expanduser().resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"process cwd resolution failed: {type(exc).__name__}") from exc
        if not cwd.is_dir():
            raise ValueError("process cwd must be an existing directory")
        environment = snapshot_process_environment(self.environment)
        if type(self.timeout_seconds) is not int or self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be a positive integer")
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "environment", environment)

    def public(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "environment_names": sorted(self.environment),
            "environment_count": len(self.environment),
            "timeout_seconds": self.timeout_seconds,
        }


_SECRET_ENVIRONMENT_MARKERS = (
    "API_KEY",
    "AUTH",
    "PROXY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
)


def secret_values(environment: Mapping[str, str]) -> tuple[str, ...]:
    """Return non-trivial secret-like values for receipt redaction."""
    frozen = snapshot_process_environment(environment)
    values = {
        value
        for key, value in frozen.items()
        if any(marker in key.upper() for marker in _SECRET_ENVIRONMENT_MARKERS) and len(value) >= 8
    }
    return tuple(sorted(values, key=len, reverse=True))


def combined_secret_values(*environments: Mapping[str, str]) -> tuple[str, ...]:
    """Return unique redactions across child and ambient environments."""
    values = {value for environment in environments for value in secret_values(environment)}
    return tuple(sorted(values, key=len, reverse=True))


def snapshot_redactions(value: object) -> tuple[str, ...]:
    """Read and validate one redaction sequence exactly once."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("redactions must be a sequence")
    try:
        redactions = tuple(value)
    except Exception as exc:
        raise ValueError(f"redactions read failed: {type(exc).__name__}") from exc
    if any(type(secret) is not str or not secret for secret in redactions):
        raise ValueError("redactions must contain exact non-empty strings")
    return redactions


def redact_frozen_text(value: str, redactions: tuple[str, ...]) -> str:
    """Apply an already frozen redaction sequence to one string."""
    for secret in redactions:
        value = value.replace(secret, "[REDACTED]")
    return value


def redact_text(value: str, redactions: Sequence[str]) -> str:
    if type(value) is not str:
        raise ValueError("redacted value must be an exact string")
    return redact_frozen_text(value, snapshot_redactions(redactions))


def snapshot_launch_failure(exc: object) -> tuple[str, int | None]:
    """Reduce one launch exception to non-sensitive stable evidence."""
    if not isinstance(exc, OSError):
        raise ValueError("launch failure requires an OSError")
    failure_type = type(exc).__name__
    if not is_process_identifier(failure_type):
        failure_type = "OSError"
    failure_errno = exc.errno if type(exc.errno) is int else None
    return (failure_type, failure_errno)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int | None
    timed_out: bool
    elapsed_seconds: float
    stdout: str = field(repr=False)
    stderr: str = field(repr=False)
    invocation: ProcessInvocation | None = None
    launch_failure_type: str | None = None
    launch_failure_errno: int | None = None

    def __post_init__(self) -> None:
        if self.returncode is not None and type(self.returncode) is not int:
            raise ValueError("process returncode must be an exact integer or None")
        if type(self.timed_out) is not bool:
            raise ValueError("process timed_out must be an exact boolean")
        if (
            type(self.elapsed_seconds) is not float
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("process elapsed_seconds must be a finite non-negative float")
        if type(self.stdout) is not str or type(self.stderr) is not str:
            raise ValueError("process streams must be exact strings")
        if self.invocation is not None and type(self.invocation) is not ProcessInvocation:
            raise ValueError("process invocation must be an exact ProcessInvocation or None")
        if self.launch_failure_type is None:
            if self.returncode is None or self.launch_failure_errno is not None:
                raise ValueError("a non-launched process requires launch-failure evidence")
        else:
            if not is_process_identifier(self.launch_failure_type):
                raise ValueError("process launch failure type must be an exact identifier")
            if (
                self.returncode is not None
                or self.invocation is None
                or self.timed_out
                or self.stdout
                or self.stderr
            ):
                raise ValueError("launch failure requires an unstarted empty-stream result")
            if self.launch_failure_errno is not None and type(self.launch_failure_errno) is not int:
                raise ValueError("process launch failure errno must be an exact integer or None")

    @classmethod
    def from_launch_failure(
        cls, invocation: ProcessInvocation, exc: OSError, *, elapsed_seconds: float
    ) -> ProcessResult:
        """Create sanitized evidence for a child that never started."""
        if type(invocation) is not ProcessInvocation:
            raise ValueError("launch failure requires an exact ProcessInvocation")
        failure_type, failure_errno = snapshot_launch_failure(exc)
        return cls(
            returncode=None,
            timed_out=False,
            elapsed_seconds=elapsed_seconds,
            stdout="",
            stderr="",
            invocation=invocation,
            launch_failure_type=failure_type,
            launch_failure_errno=failure_errno,
        )

    @property
    def launched(self) -> bool:
        return self.returncode is not None

    @property
    def launch_failed(self) -> bool:
        return self.launch_failure_type is not None

    def to_public_dict(
        self, tail_chars: int = 20000, *, redactions: Sequence[str] = ()
    ) -> dict[str, object]:
        if type(tail_chars) is not int or tail_chars < 1:
            raise ValueError("tail_chars must be a positive integer")
        frozen_redactions = snapshot_redactions(redactions)
        value = {
            "launched": self.launched,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "stdout_tail": redact_frozen_text(self.stdout[-tail_chars:], frozen_redactions),
            "stderr_tail": redact_frozen_text(self.stderr[-tail_chars:], frozen_redactions),
            "launch_failure": {"type": self.launch_failure_type, "errno": self.launch_failure_errno}
            if self.launch_failed
            else None,
        }
        if self.invocation is not None:
            value["invocation"] = self.invocation.public()
        return value


def launch_failure_summary(result: ProcessResult) -> str:
    """Return one sanitized diagnostic for a child that never started."""
    if type(result) is not ProcessResult:
        raise ValueError("launch failure summary requires an exact ProcessResult")
    if not result.launch_failed:
        return ""
    summary = f"{result.launch_failure_type}"
    if result.launch_failure_errno is not None:
        summary += f"; errno={result.launch_failure_errno}"
    return summary


def _signal_process_group(process: subprocess.Popen[str], signal_number: int) -> None:
    """Best-effort signal delivery for timeout cleanup races."""
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass


def execute(
    argv: Sequence[str], *, cwd: Path, environment: Mapping[str, str], timeout_seconds: int
) -> ProcessResult:
    invocation = ProcessInvocation(
        argv=argv, cwd=cwd, environment=environment, timeout_seconds=timeout_seconds
    )
    return execute_invocation(invocation)


def execute_invocation(invocation: ProcessInvocation) -> ProcessResult:
    """Execute one exact invocation and bind it to the returned result."""
    if type(invocation) is not ProcessInvocation:
        raise ValueError("execution requires an exact ProcessInvocation")
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(invocation.argv),
            cwd=invocation.cwd,
            env=dict(invocation.environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        return ProcessResult.from_launch_failure(
            invocation, exc, elapsed_seconds=time.monotonic() - started
        )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=invocation.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _signal_process_group(process, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL)
            stdout, stderr = process.communicate()
    return ProcessResult(
        returncode=process.returncode,
        timed_out=timed_out,
        elapsed_seconds=time.monotonic() - started,
        stdout=stdout,
        stderr=stderr,
        invocation=invocation,
    )
