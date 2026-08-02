"""RethinkSkill evolution protocol."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rethinkskill.errors import ConfigurationError
from rethinkskill.utils.serde import RegularFileSnapshot, read_json_snapshot


@dataclass(frozen=True, slots=True)
class GateState:
    current_hard: float
    current_soft: float
    best_hard: float
    best_soft: float
    best_round: int
    noop_streak: int = 0
    regression_streak: int = 0

    def validate(self) -> None:
        for label, value in (
            ("current_hard", self.current_hard),
            ("current_soft", self.current_soft),
            ("best_hard", self.best_hard),
            ("best_soft", self.best_soft),
        ):
            finite_float(value, label)
        for label, value in (
            ("best_round", self.best_round),
            ("noop_streak", self.noop_streak),
            ("regression_streak", self.regression_streak),
        ):
            if type(value) is not int or value < 0:
                raise ConfigurationError(f"{label} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class GateDecision:
    action: str
    state: GateState


@dataclass(frozen=True, slots=True)
class GatePolicy:
    hard_dead_band: float
    soft_rescue_delta: float | None

    def validate(self) -> None:
        positive_float(self.hard_dead_band, "hard_dead_band")
        if self.soft_rescue_delta is not None:
            positive_float(self.soft_rescue_delta, "soft_rescue_delta")


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    name: str
    train_size: int | None
    validation_size: int | None
    test_size: int | None
    hard_dead_band: float
    soft_rescue_delta: float | None
    panels: tuple[str, ...]
    unavailable_panels: tuple[str, ...]

    def validate(self) -> None:
        if type(self.name) is not str or not self.name.strip():
            raise ConfigurationError("benchmark protocol name must be non-empty text")
        optional_nonnegative_int(self.train_size, "train_size")
        optional_nonnegative_int(self.validation_size, "validation_size")
        optional_nonnegative_int(self.test_size, "test_size")
        positive_float(self.hard_dead_band, "hard_dead_band")
        if self.soft_rescue_delta is not None:
            positive_float(self.soft_rescue_delta, "soft_rescue_delta")
        for label, values in (
            ("panels", self.panels),
            ("unavailable_panels", self.unavailable_panels),
        ):
            if type(values) is not tuple or not all(
                type(value) is str and value.strip() for value in values
            ):
                raise ConfigurationError(f"benchmark protocol {label} must be a tuple of text")
            if len(values) != len(set(values)):
                raise ConfigurationError(f"benchmark protocol {label} contains duplicates")

    @property
    def gate_policy(self) -> GatePolicy:
        return GatePolicy(
            hard_dead_band=self.hard_dead_band, soft_rescue_delta=self.soft_rescue_delta
        )


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    source: Path
    protocol_id: str
    rounds: int
    arms: Mapping[str, tuple[str, ...]]
    conditions: Mapping[str, str | None]
    robustness_repeats: int
    benchmarks: Mapping[str, BenchmarkProtocol]
    evidence_note: str
    source_snapshot: RegularFileSnapshot | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, Path):
            raise ConfigurationError("protocol source must be a Path")
        if (
            self.source_snapshot is not None
            and type(self.source_snapshot) is not RegularFileSnapshot
        ):
            raise ConfigurationError(
                "protocol source_snapshot must be an exact regular-file snapshot"
            )
        if type(self.protocol_id) is not str or not self.protocol_id.strip():
            raise ConfigurationError("protocol_id must be non-empty text")
        if type(self.rounds) is not int or self.rounds < 1:
            raise ConfigurationError("protocol rounds must be positive")
        if type(self.robustness_repeats) is not int or self.robustness_repeats < 1:
            raise ConfigurationError("protocol robustness_repeats must be positive")
        if type(self.evidence_note) is not str:
            raise ConfigurationError("protocol evidence_note must be text")
        if not isinstance(self.arms, Mapping):
            raise ConfigurationError("protocol arms must be a mapping")
        arms = dict(self.arms)
        if not arms or any(
            (
                type(name) is not str
                or not name.strip()
                or type(categories) is not tuple
                or (not categories)
                or any(
                    (type(category) is not str or not category.strip() for category in categories)
                )
                or (len(categories) != len(set(categories)))
                for name, categories in arms.items()
            )
        ):
            raise ConfigurationError("protocol arms must map text names to non-empty text tuples")
        if not isinstance(self.conditions, Mapping):
            raise ConfigurationError("protocol conditions must be a mapping")
        conditions = dict(self.conditions)
        if not conditions or any(
            (
                type(name) is not str
                or not name.strip()
                or (arm is not None and type(arm) is not str)
                for name, arm in conditions.items()
            )
        ):
            raise ConfigurationError("protocol conditions must map text names to text or null")
        if not isinstance(self.benchmarks, Mapping):
            raise ConfigurationError("protocol benchmarks must be a mapping")
        benchmarks = dict(self.benchmarks)
        if not benchmarks:
            raise ConfigurationError("protocol benchmarks must be non-empty")
        for name, benchmark in benchmarks.items():
            if (
                type(name) is not str
                or not name.strip()
                or type(benchmark) is not BenchmarkProtocol
                or (benchmark.name != name)
            ):
                raise ConfigurationError(f"invalid benchmark protocol registration: {name!r}")
            benchmark.validate()
        object.__setattr__(self, "arms", MappingProxyType(arms))
        object.__setattr__(self, "conditions", MappingProxyType(conditions))
        object.__setattr__(self, "benchmarks", MappingProxyType(benchmarks))

    @classmethod
    def load(cls, source: Path) -> ProtocolSpec:
        return load_protocol(source)

    def benchmark(self, name: str) -> BenchmarkProtocol:
        try:
            return self.benchmarks[name]
        except KeyError as exc:
            raise ConfigurationError(f"benchmark absent from protocol: {name}") from exc

    def plan(self, benchmark: str) -> dict[str, object]:
        return plan_protocol(self, benchmark)


def positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def optional_nonnegative_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ConfigurationError(f"{label} must be null or a nonnegative integer")
    return value


def positive_float(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{label} must be positive")
    return float(value)


def finite_float(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ConfigurationError(f"{label} must be a finite number")
    return float(value)


def plan_protocol(protocol: ProtocolSpec, benchmark: str) -> dict[str, object]:
    if type(protocol) is not ProtocolSpec:
        raise ConfigurationError("protocol planning requires an exact ProtocolSpec")
    if type(benchmark) is not str or not benchmark.strip():
        raise ConfigurationError("protocol benchmark must be non-empty text")
    if protocol.source_snapshot is None:
        raise ConfigurationError("protocol planning requires a loader-bound source snapshot")
    spec = protocol.benchmark(benchmark)
    robustness_jobs = len(spec.panels) * len(protocol.conditions) * protocol.robustness_repeats
    return {
        "schema_version": 1,
        "status": "RETHINKSKILL_PROTOCOL_PLAN_ZERO_CALL",
        "protocol_id": protocol.protocol_id,
        "protocol": {"path": str(protocol.source), "sha256": protocol.source_snapshot.sha256},
        "benchmark": benchmark,
        "sizes": {
            "train": spec.train_size,
            "validation": spec.validation_size,
            "test": spec.test_size,
        },
        "arms": {name: list(categories) for name, categories in protocol.arms.items()},
        "conditions": dict(protocol.conditions),
        "rounds": protocol.rounds,
        "gate": {
            "hard_dead_band": spec.hard_dead_band,
            "soft_rescue_delta": spec.soft_rescue_delta,
        },
        "stages": [
            {
                "name": "prepare",
                "requires": ["parent_skill", "frozen_inputs", "optimizer_prompts"],
                "model_calls": 0,
            },
            {
                "name": "stage1_evolution",
                "depends_on": ["prepare"],
                "arm_rounds": len(protocol.arms) * protocol.rounds,
            },
            {
                "name": "final_test",
                "depends_on": ["stage1_evolution_complete"],
                "conditions": list(protocol.conditions),
            },
            {
                "name": "robustness_transfer",
                "depends_on": ["final_test_complete", "frozen_panels"],
                "panels": list(spec.panels),
                "unavailable_panels": list(spec.unavailable_panels),
                "repeats": protocol.robustness_repeats,
                "condition_panel_repeat_jobs": robustness_jobs,
            },
        ],
        "unresolved_parameters": [
            name
            for name, item in (
                ("train_size", spec.train_size),
                ("validation_size", spec.validation_size),
                ("test_size", spec.test_size),
            )
            if item is None
        ],
        "evidence_note": protocol.evidence_note,
        "model_calls": 0,
        "optimizer_calls": 0,
        "target_calls": 0,
    }


def gate_decision(
    protocol: BenchmarkProtocol | GatePolicy,
    state: GateState,
    *,
    hard: float,
    soft: float,
    round_no: int,
) -> GateDecision:
    if type(protocol) not in (BenchmarkProtocol, GatePolicy):
        raise ConfigurationError("gate decision requires an exact BenchmarkProtocol or GatePolicy")
    if type(state) is not GateState:
        raise ConfigurationError("gate decision requires an exact GateState")
    protocol.validate()
    state.validate()
    hard = finite_float(hard, "hard")
    soft = finite_float(soft, "soft")
    if type(round_no) is not int or round_no < 1:
        raise ConfigurationError("round_no must be at least one")
    delta = hard - state.current_hard
    soft_delta = soft - state.current_soft
    if delta >= protocol.hard_dead_band:
        is_best = hard > state.best_hard
        return GateDecision(
            "accept_new_best" if is_best else "accept",
            GateState(
                current_hard=hard,
                current_soft=soft,
                best_hard=hard if is_best else state.best_hard,
                best_soft=soft if is_best else state.best_soft,
                best_round=round_no if is_best else state.best_round,
            ),
        )
    if delta <= -protocol.hard_dead_band:
        return GateDecision(
            "reject",
            GateState(
                current_hard=state.current_hard,
                current_soft=state.current_soft,
                best_hard=state.best_hard,
                best_soft=state.best_soft,
                best_round=state.best_round,
                noop_streak=0,
                regression_streak=state.regression_streak + 1,
            ),
        )
    if protocol.soft_rescue_delta is not None and soft_delta >= protocol.soft_rescue_delta:
        return GateDecision(
            "accept",
            GateState(
                current_hard=hard,
                current_soft=soft,
                best_hard=state.best_hard,
                best_soft=state.best_soft,
                best_round=state.best_round,
            ),
        )
    return GateDecision(
        "flat",
        GateState(
            current_hard=state.current_hard,
            current_soft=state.current_soft,
            best_hard=state.best_hard,
            best_soft=state.best_soft,
            best_round=state.best_round,
            noop_streak=state.noop_streak + 1,
            regression_streak=0,
        ),
    )


_EXPECTED_ARMS = {"normal", "fail_only", "success_only"}

_EXPECTED_CONDITIONS = {
    "parent": None,
    "normal_final": "normal",
    "fail_only_final": "fail_only",
    "success_only_final": "success_only",
}


def load_protocol(source: Path) -> ProtocolSpec:
    try:
        source_snapshot, value = read_json_snapshot(source)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"protocol source could not be snapshotted: {source}") from exc
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
    ):
        raise ConfigurationError(f"unsupported protocol schema: {source}")
    arms_raw = value.get("arms")
    conditions_raw = value.get("conditions")
    benchmarks_raw = value.get("benchmarks")
    if not isinstance(arms_raw, dict) or not isinstance(conditions_raw, dict):
        raise ConfigurationError("protocol arms and conditions must be objects")
    if not isinstance(benchmarks_raw, dict):
        raise ConfigurationError("protocol benchmarks must be an object")
    if any(
        (
            type(name) is not str
            or type(categories) is not list
            or (not categories)
            or any((type(category) is not str for category in categories))
            for name, categories in arms_raw.items()
        )
    ):
        raise ConfigurationError("protocol arms must map text names to non-empty text arrays")
    arms = {name: tuple(categories) for name, categories in arms_raw.items()}
    if set(arms) != _EXPECTED_ARMS:
        raise ConfigurationError(f"protocol arms drift: {set(arms)}")
    if any(
        (
            type(name) is not str or (arm is not None and type(arm) is not str)
            for name, arm in conditions_raw.items()
        )
    ):
        raise ConfigurationError("protocol conditions must map text names to text or null")
    conditions = dict(conditions_raw)
    if conditions != _EXPECTED_CONDITIONS:
        raise ConfigurationError(f"protocol conditions drift: {conditions}")
    benchmarks = {}
    for name, raw in benchmarks_raw.items():
        if type(name) is not str or not name.strip() or (not isinstance(raw, dict)):
            raise ConfigurationError(f"benchmark protocol must be an object: {name}")
        benchmarks[name] = BenchmarkProtocol(
            name=name,
            train_size=optional_nonnegative_int(raw.get("train_size"), f"{name}.train_size"),
            validation_size=optional_nonnegative_int(
                raw.get("validation_size"), f"{name}.validation_size"
            ),
            test_size=optional_nonnegative_int(raw.get("test_size"), f"{name}.test_size"),
            hard_dead_band=positive_float(raw.get("hard_dead_band"), f"{name}.hard_dead_band"),
            soft_rescue_delta=positive_float(
                raw.get("soft_rescue_delta"), f"{name}.soft_rescue_delta"
            )
            if raw.get("soft_rescue_delta") is not None
            else None,
            panels=_text_tuple(raw.get("panels", []), f"{name}.panels"),
            unavailable_panels=_text_tuple(
                raw.get("unavailable_panels", []), f"{name}.unavailable_panels"
            ),
        )
    protocol_id = value.get("protocol_id")
    evidence_note = value.get("evidence_note")
    if type(protocol_id) is not str or not protocol_id.strip():
        raise ConfigurationError("protocol_id must be non-empty text")
    if type(evidence_note) is not str:
        raise ConfigurationError("protocol evidence_note must be text")
    return ProtocolSpec(
        source=source.resolve(),
        source_snapshot=source_snapshot,
        protocol_id=protocol_id,
        rounds=positive_int(value.get("rounds"), "rounds"),
        arms=arms,
        conditions=conditions,
        robustness_repeats=positive_int(value.get("robustness_repeats"), "robustness_repeats"),
        benchmarks=benchmarks,
        evidence_note=evidence_note,
    )


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ConfigurationError(f"{label} must be an array of text")
    return tuple(value)
