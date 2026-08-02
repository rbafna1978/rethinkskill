from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rethinkskill.errors import ConfigurationError
from rethinkskill.evolution.protocol import (
    GatePolicy,
    GateState,
    ProtocolSpec,
    gate_decision,
)
from rethinkskill.utils.fs import Repository
from rethinkskill.utils.serde import sha256_bytes


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repository = Repository.discover(Path(__file__))
        cls.protocol = ProtocolSpec.load(repository.root / "configs/protocols/skillopt_lite.json")

    def test_protocol_owns_three_feedback_arms_and_four_conditions(self) -> None:
        self.assertEqual(
            self.protocol.arms,
            {
                "normal": ("failed", "passed"),
                "fail_only": ("failed",),
                "success_only": ("passed",),
            },
        )
        self.assertEqual(
            self.protocol.conditions,
            {
                "parent": None,
                "normal_final": "normal",
                "fail_only_final": "fail_only",
                "success_only_final": "success_only",
            },
        )

    def test_protocol_contract_maps_are_deeply_immutable(self) -> None:
        with self.assertRaises(TypeError):
            self.protocol.arms["normal"] = ("failed",)
        with self.assertRaises(TypeError):
            self.protocol.conditions["parent"] = "normal"
        with self.assertRaises(TypeError):
            self.protocol.benchmarks["docvqa"] = self.protocol.benchmark("searchqa")

    def test_protocol_planning_rejects_contract_subclasses(self) -> None:
        class OverridingProtocol(ProtocolSpec):
            pass

        overriding = OverridingProtocol(
            source=self.protocol.source,
            protocol_id=self.protocol.protocol_id,
            rounds=self.protocol.rounds,
            arms=self.protocol.arms,
            conditions=self.protocol.conditions,
            robustness_repeats=self.protocol.robustness_repeats,
            benchmarks=self.protocol.benchmarks,
            evidence_note=self.protocol.evidence_note,
        )
        with self.assertRaisesRegex(
            ConfigurationError,
            "exact ProtocolSpec",
        ):
            overriding.plan("docvqa")

    def test_gate_accept_reject_soft_rescue_and_flat(self) -> None:
        benchmark = self.protocol.benchmark("docvqa")
        state = GateState(
            current_hard=0.5,
            current_soft=0.6,
            best_hard=0.5,
            best_soft=0.6,
            best_round=0,
        )
        accepted = gate_decision(benchmark, state, hard=0.52, soft=0.6, round_no=1)
        self.assertEqual(accepted.action, "accept_new_best")
        rejected = gate_decision(benchmark, state, hard=0.48, soft=0.7, round_no=1)
        self.assertEqual(rejected.action, "reject")
        rescued = gate_decision(benchmark, state, hard=0.505, soft=0.63, round_no=1)
        self.assertEqual(rescued.action, "accept")
        flat = gate_decision(benchmark, state, hard=0.505, soft=0.61, round_no=1)
        self.assertEqual(flat.action, "flat")

    def test_plan_preserves_unresolved_searchqa_sizes(self) -> None:
        plan = self.protocol.plan("searchqa")
        self.assertEqual(plan["unresolved_parameters"], ["validation_size", "test_size"])
        self.assertEqual(plan["model_calls"], 0)

    def test_protocol_plan_hashes_the_bytes_that_were_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "protocol.json"
            original = self.protocol.source.read_bytes()
            source.write_bytes(original)
            protocol = ProtocolSpec.load(source)

            source.write_text("{}\n", encoding="utf-8")
            plan = protocol.plan("searchqa")

        self.assertEqual(
            plan["protocol"]["sha256"],
            sha256_bytes(original),
        )

    def test_protocol_loader_rejects_scalar_coercion(self) -> None:
        source = json.loads(self.protocol.source.read_text(encoding="utf-8"))
        invalid_values = (
            ("protocol_id", 123),
            ("rounds", True),
            ("arms", {**source["arms"], "normal": ["failed", 1]}),
            ("evidence_note", 123),
        )
        for field, value in invalid_values:
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as temporary,
            ):
                payload = dict(source)
                payload[field] = value
                path = Path(temporary) / "protocol.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ConfigurationError):
                    ProtocolSpec.load(path)

    def test_robustness_job_count_is_explicit(self) -> None:
        plan = self.protocol.plan("officeqa")
        stage = plan["stages"][-1]
        self.assertEqual(stage["condition_panel_repeat_jobs"], 4 * 6 * 3)

    def test_gate_policy_rejects_non_finite_thresholds(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                GatePolicy(
                    hard_dead_band=value,
                    soft_rescue_delta=None,
                ).validate()

    def test_gate_rejects_contract_and_scalar_subclasses(self) -> None:
        class OverridingPolicy(GatePolicy):
            pass

        class OverridingState(GateState):
            pass

        class OverridingFloat(float):
            pass

        policy = GatePolicy(0.01, None)
        state = GateState(0.5, 0.6, 0.5, 0.6, 0)
        cases = (
            (OverridingPolicy(0.01, None), state, 0.52, 0.6, 1),
            (
                policy,
                OverridingState(0.5, 0.6, 0.5, 0.6, 0),
                0.52,
                0.6,
                1,
            ),
            (policy, state, OverridingFloat(0.52), 0.6, 1),
        )
        for protocol, gate_state, hard, soft, round_no in cases:
            with (
                self.subTest(
                    protocol=type(protocol).__name__,
                    state=type(gate_state).__name__,
                    hard=type(hard).__name__,
                ),
                self.assertRaises(ConfigurationError),
            ):
                gate_decision(
                    protocol,
                    gate_state,
                    hard=hard,
                    soft=soft,
                    round_no=round_no,
                )

    def test_gate_rejects_non_finite_scores_and_invalid_state(self) -> None:
        policy = GatePolicy(0.01, None)
        state = GateState(0.5, 0.6, 0.5, 0.6, 0)
        for hard, soft in (
            (float("nan"), 0.6),
            (0.52, float("inf")),
        ):
            with self.subTest(hard=hard, soft=soft), self.assertRaises(ConfigurationError):
                gate_decision(
                    policy,
                    state,
                    hard=hard,
                    soft=soft,
                    round_no=1,
                )
        with self.assertRaises(ConfigurationError):
            gate_decision(
                policy,
                replace(state, noop_streak=-1),
                hard=0.52,
                soft=0.6,
                round_no=1,
            )
