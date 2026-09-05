"""RethinkSkill providers authorization."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from rethinkskill.errors import (
    AuthorizationRequiredError,
    ConfigurationError,
    ResultValidationError,
)
from rethinkskill.utils.fs import Repository, utc_now
from rethinkskill.utils.integrity import (
    exact_call_counts,
    safe_exception_type,
)
from rethinkskill.utils.release import regular_file_inventory
from rethinkskill.utils.serde import (
    atomic_write_json,
    snapshot_regular_file,
    strict_json_loads,
)


def require_model_call_authorization(authorized: bool, *, action: str) -> None:
    """Require an exact explicit authorization decision before live work."""
    if type(authorized) is not bool:
        raise ConfigurationError("authorized must be boolean")
    if type(action) is not str or not action.strip():
        raise ConfigurationError("authorization action must be non-empty text")
    if not authorized:
        raise AuthorizationRequiredError(f"{action} requires explicit model-call authorization")


def validate_output_root(output_root: Path, *, label: str, require_empty: bool) -> None:
    """Validate a run root without creating or otherwise mutating it."""
    if not isinstance(output_root, Path):
        raise ConfigurationError("output root must be a pathlib.Path")
    if type(label) is not str or not label.strip():
        raise ConfigurationError("output root label must be non-empty text")
    if type(require_empty) is not bool:
        raise ConfigurationError("require_empty must be boolean")
    if output_root.exists() and (not output_root.is_dir()):
        raise ConfigurationError(f"{label} output root is not a directory: {output_root}")
    if require_empty and output_root.exists() and any(output_root.iterdir()):
        raise ConfigurationError(f"{label} output root is non-empty: {output_root}")


def validate_frozen_output_scope(
    *, repository: Repository, output_root: Path, run_root: Path, label: str, require_empty: bool
) -> Path:
    """Re-resolve a frozen output path and enforce its directory policy."""
    if type(repository) is not Repository:
        raise ConfigurationError("output scope requires an exact Repository")
    if not isinstance(output_root, Path):
        raise ConfigurationError("output root must be a pathlib.Path")
    if not isinstance(run_root, Path):
        raise ConfigurationError("run root must be a pathlib.Path")
    safe_output = repository.validate_new_output(output_root, run_root)
    if safe_output != output_root:
        raise ConfigurationError(f"{label} output root no longer matches its frozen plan")
    validate_output_root(safe_output, label=label, require_empty=require_empty)
    return safe_output


def initialize_fresh_output_root(output_root: Path, *, label: str) -> None:
    """Validate and create an empty root while the caller holds its lock."""
    validate_output_root(output_root, label=label, require_empty=True)
    output_root.mkdir(parents=True, exist_ok=True)


TERMINAL_FAILURE_FILENAME = "TERMINAL_FAILURE.json"

TERMINAL_FAILURE_SCHEMA_VERSION = 2

_RUN_CONTRACTS = MappingProxyType(
    {
        "native": MappingProxyType(
            {
                "manifest": ".rethinkskill/RUN_MANIFEST.json",
                "receipt": ".rethinkskill/PROCESS_RECEIPT.json",
                "components": ("target",),
                "phases": frozenset(
                    {
                        "task_materialization",
                        "task_execution",
                        "task_result",
                        "task_evidence",
                        "ledger_publication",
                        "receipt_publication",
                    }
                ),
            }
        ),
        "evolution": MappingProxyType(
            {
                "manifest": ".rethinkskill/EVOLUTION_MANIFEST.json",
                "receipt": ".rethinkskill/EVOLUTION_RECEIPT.json",
                "components": ("target", "optimizer"),
                "phases": frozenset(
                    {
                        "artifact_initialization",
                        "initial_validation",
                        "training",
                        "optimization",
                        "candidate_validation",
                        "round_publication",
                        "finalization",
                    }
                ),
            }
        ),
    }
)


@dataclass(frozen=True, slots=True)
class TerminalCallAccounting:
    """Known call lower bound and whether it is exact."""

    attempted: int
    completed: int
    accounting_known: bool

    def __post_init__(self) -> None:
        exact_call_counts(self.attempted, self.completed, label="terminal failure calls")
        if type(self.accounting_known) is not bool:
            raise TypeError("terminal failure accounting-known must be boolean")

    def public(self) -> dict[str, object]:
        return {
            "attempted": self.attempted,
            "completed": self.completed,
            "accounting_known": self.accounting_known,
        }


def terminal_failure_contract(run_kind: str) -> Mapping[str, object]:
    """Return the immutable internal contract for one supported run kind."""
    try:
        return _RUN_CONTRACTS[run_kind]
    except (KeyError, TypeError) as exc:
        raise ResultValidationError(f"unsupported terminal failure run kind: {run_kind!r}") from exc


def write_terminal_failure(
    run_root: Path,
    *,
    run_kind: str,
    phase: str,
    exception: BaseException,
    calls: Mapping[str, TerminalCallAccounting],
) -> dict[str, object]:
    """Bind one framework exception without persisting its message."""
    contract = terminal_failure_contract(run_kind)
    phases = contract["phases"]
    if type(phase) is not str or phase not in phases:
        raise ResultValidationError(f"invalid {run_kind} terminal failure phase: {phase!r}")
    if not isinstance(exception, BaseException):
        raise TypeError("terminal failure requires an exception")
    components = contract["components"]
    if (
        type(calls) is not dict
        or set(calls) != set(components)
        or any(type(value) is not TerminalCallAccounting for value in calls.values())
    ):
        raise ResultValidationError(f"invalid {run_kind} terminal failure call accounting")
    root = run_root.expanduser().absolute()
    failure_path = root / ".rethinkskill" / TERMINAL_FAILURE_FILENAME
    receipt_path = root / str(contract["receipt"])
    if failure_path.exists() or failure_path.is_symlink():
        raise ResultValidationError(
            f"refusing to replace terminal failure evidence: {failure_path}"
        )
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ResultValidationError(
            "terminal failure evidence cannot coexist with a normal receipt"
        )
    manifest_relative = str(contract["manifest"])
    manifest_path = root / manifest_relative
    manifest_snapshot = snapshot_regular_file(manifest_path)
    manifest = strict_json_loads(manifest_snapshot.payload)
    if type(manifest) is not dict:
        raise ResultValidationError("terminal failure manifest is not an object")
    artifacts = regular_file_inventory(
        root, relative_to=root, excluded=(failure_path,), label=f"{run_kind} partial run artifacts"
    )
    evidence = {
        "schema_version": TERMINAL_FAILURE_SCHEMA_VERSION,
        "status": "RETHINKSKILL_FRAMEWORK_FAILURE",
        "created_at": utc_now(),
        "run_kind": run_kind,
        "phase": phase,
        "exception_type": safe_exception_type(exception),
        "calls": {name: calls[name].public() for name in components},
        "run_manifest": {"path": manifest_relative, "sha256": manifest_snapshot.sha256},
        "artifacts": list(artifacts),
        "scientific_completion_claimed": False,
    }
    atomic_write_json(failure_path, evidence)
    return evidence


@contextmanager
def terminal_failure_boundary(
    run_root: Path,
    *,
    run_kind: str,
    phase: str,
    calls: Callable[[], dict[str, TerminalCallAccounting]],
) -> Iterator[None]:
    """Persist redacted partial-run evidence, then preserve the exception."""
    if not callable(calls):
        raise TypeError("terminal failure call snapshot must be callable")
    try:
        yield
    except Exception as exc:
        try:
            write_terminal_failure(
                run_root, run_kind=run_kind, phase=phase, exception=exc, calls=calls()
            )
        except Exception:
            pass
        raise
