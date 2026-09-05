"""RethinkSkill runtime types."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata, util
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rethinkskill.benchmarks.scoring import Verification
from rethinkskill.runtime.tasks import NativeTask, RenderedTask
from rethinkskill.utils.integrity import exact_call_counts
from rethinkskill.utils.serde import (
    canonical_json_bytes,
    freeze_json_mapping,
    thaw_json_mapping,
)


def numeric_version(value: str) -> tuple[int, ...] | None:
    """Parse a numeric release prefix without importing packaging libraries."""
    match = re.match("^(\\d+(?:\\.\\d+)*)", value.strip())
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def probe_distribution(
    *, distribution: str, import_name: str, requirement: str, accepts: Callable[[str], bool]
) -> dict[str, object]:
    """Report module/distribution availability and version compatibility."""
    module_available = util.find_spec(import_name) is not None
    try:
        version: str | None = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        version = None
    compatible = module_available and version is not None and accepts(version)
    return {
        "ready": compatible,
        "module_available": module_available,
        "distribution": distribution,
        "version": version,
        "requirement": requirement,
        "version_compatible": compatible,
    }


if TYPE_CHECKING:
    from rethinkskill.runtime.tasks import RenderedTask


@dataclass(frozen=True, slots=True)
class ModelOutcome:
    """Provider-neutral result from one complete benchmark task execution."""

    status: str
    response: str
    raw: str
    process: Mapping[str, object]
    attempted_calls: int
    completed_calls: int
    failure_class: str | None = None
    failure: str = ""

    def validate(self) -> None:
        """Reject malformed third-party executor outcomes at the trust boundary."""
        if type(self.status) is not str or self.status not in {"COMPLETED", "FAILED"}:
            raise ValueError(f"invalid model outcome status: {self.status!r}")
        if type(self.response) is not str or type(self.raw) is not str:
            raise ValueError("model outcome response and raw fields must be strings")
        if not isinstance(self.process, Mapping):
            raise ValueError("model outcome process field must be a mapping")
        try:
            canonical_json_bytes(thaw_json_mapping(self.process))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "model outcome process field must be canonical-JSON serializable"
            ) from exc
        exact_call_counts(self.attempted_calls, self.completed_calls, label="model outcome")
        if self.failure_class is not None and (
            type(self.failure_class) is not str or not self.failure_class.strip()
        ):
            raise ValueError("model outcome failure_class must be a non-empty string or None")
        if type(self.failure) is not str:
            raise ValueError("model outcome failure field must be a string")
        if self.status == "COMPLETED" and self.failure_class:
            raise ValueError("completed model outcome cannot carry a failure_class")
        if self.status == "COMPLETED" and (
            self.attempted_calls < 1 or self.completed_calls != self.attempted_calls
        ):
            raise ValueError("completed model outcome requires one or more fully accounted calls")
        if self.status == "FAILED" and (not self.failure_class):
            raise ValueError("failed model outcome requires a failure_class")

    @property
    def ok(self) -> bool:
        return (
            self.status == "COMPLETED"
            and self.attempted_calls >= 1
            and (self.completed_calls == self.attempted_calls)
            and (not self.failure_class)
        )

    def public(self) -> dict[str, object]:
        return {
            "status": self.status,
            "response": self.response,
            "process": thaw_json_mapping(self.process),
            "attempted_calls": self.attempted_calls,
            "completed_calls": self.completed_calls,
            "failure_class": self.failure_class,
            "failure": self.failure,
        }


def freeze_model_outcome(value: ModelOutcome) -> ModelOutcome:
    """Detach one exact executor outcome before accounting or persistence."""
    if type(value) is not ModelOutcome:
        raise TypeError("executor must return an exact ModelOutcome")
    try:
        process = freeze_json_mapping(value.process)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("model outcome process field must be finite canonical JSON") from exc
    frozen = ModelOutcome(
        status=value.status,
        response=value.response,
        raw=value.raw,
        process=process,
        attempted_calls=value.attempted_calls,
        completed_calls=value.completed_calls,
        failure_class=value.failure_class,
        failure=value.failure,
    )
    frozen.validate()
    return frozen


@dataclass(frozen=True, slots=True)
class ModelOutcomeBoundary:
    """Validate and detach every provider call, including interactive steps."""

    executor: Any
    manifest: Mapping[str, object]

    def public_manifest(self) -> Mapping[str, object]:
        return thaw_json_mapping(self.manifest)

    def execute(
        self, rendered: RenderedTask, *, workspace: Path, timeout_seconds: int
    ) -> ModelOutcome:
        return freeze_model_outcome(
            self.executor.execute(rendered, workspace=workspace, timeout_seconds=timeout_seconds)
        )


class BenchmarkHarness(Protocol):
    """Dataset and prompt behavior owned by one benchmark family."""

    def load_tasks(
        self,
        source: Path,
        *,
        split: str,
        limit: int | None,
        requested_ids: Sequence[str],
        seed: int,
        asset_root: Path | None,
    ) -> tuple[NativeTask, ...]: ...

    def render(self, task: NativeTask, skill: str) -> RenderedTask: ...

    def evaluate(self, task: NativeTask, response: str) -> Verification: ...


class ModelExecutor(Protocol):
    """A model process that knows nothing about benchmark semantics."""

    def public_manifest(self) -> Mapping[str, object]: ...

    def execute(
        self, rendered: RenderedTask, *, workspace: Path, timeout_seconds: int
    ) -> ModelOutcome: ...
