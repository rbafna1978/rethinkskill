"""Focused unit tests for split evolution policy modules."""

from __future__ import annotations

import unittest

from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.evolution.loop import (
    EvaluationOutcome,
    EvolutionExecutionState,
    EvolutionOptions,
    OptimizationContext,
    ProposalOutcome,
    _evaluation_boundary,
    _optimizer_boundary,
    filter_feedback,
    with_feedback,
)


class EvolutionEngineUnitTests(unittest.TestCase):
    def test_execution_state_aggregates_calls_and_failure_location(self) -> None:
        initial = EvaluationOutcome(
            valid=True,
            hard=0.5,
            soft=0.75,
            feedback=(),
            reference="fixture://initial",
            attempted_calls=1,
            completed_calls=1,
        )
        state = EvolutionExecutionState.initialize(
            initial_skill="skill",
            initial_score=initial,
            accounting_known=True,
        )
        training = EvaluationOutcome(
            valid=True,
            hard=0.5,
            soft=0.75,
            feedback=(),
            reference="fixture://training",
            attempted_calls=2,
            completed_calls=1,
        )
        state.target_calls.add(training, accounting_known=False)
        proposal = ProposalOutcome(
            valid=True,
            candidate_skill="changed",
            operation="replace",
            rationale="fixture",
            attempted_calls=1,
            completed_calls=1,
        )
        state.optimizer_calls.add(proposal, accounting_known=True)
        state.fail(
            failure_class="fixture_failure",
            failure="fixture",
            stage="candidate_validation",
            round_no=2,
        )

        self.assertEqual(state.target_calls.attempted, 3)
        self.assertEqual(state.target_calls.completed, 2)
        self.assertFalse(state.target_calls.accounting_known)
        self.assertEqual(state.optimizer_calls.attempted, 1)
        self.assertTrue(state.optimizer_calls.accounting_known)
        self.assertTrue(state.failed)
        self.assertEqual(state.failure_stage, "candidate_validation")
        self.assertEqual(state.failure_round, 2)
        with self.assertRaisesRegex(
            ResultValidationError,
            "already recorded",
        ):
            state.fail(
                failure_class="second_failure",
                failure="fixture",
                stage="training",
                round_no=3,
            )

    def test_execution_state_rejects_untrusted_accounting_flags(self) -> None:
        initial = EvaluationOutcome(
            valid=True,
            hard=0.5,
            soft=0.75,
            feedback=(),
            reference="fixture://initial",
            attempted_calls=1,
            completed_calls=1,
        )
        with self.assertRaisesRegex(
            ResultValidationError,
            "accounting-known flag must be boolean",
        ):
            EvolutionExecutionState.initialize(
                initial_skill="skill",
                initial_score=initial,
                accounting_known=1,  # type: ignore[arg-type]
            )

    def test_component_boundaries_reject_outcome_subclasses(self) -> None:
        class OverridingEvaluation(EvaluationOutcome):
            def validate(self):
                return None

        class Evaluator:
            @staticmethod
            def evaluate(skill, *, split, round_no):
                return OverridingEvaluation(
                    valid=True,
                    hard=1.0,
                    soft=1.0,
                    feedback=(),
                    reference="forged://evaluation",
                    attempted_calls=1,
                    completed_calls=1,
                )

        evaluation, evaluation_known = _evaluation_boundary(
            Evaluator(),
            "skill",
            split="train",
            round_no=1,
            stage="training",
        )
        self.assertFalse(evaluation_known)
        self.assertEqual(
            evaluation.failure_class,
            "native_evaluation_invalid",
        )

        class OverridingProposal(ProposalOutcome):
            def validate(self, *, current_skill, max_skill_chars):
                return None

        class Optimizer:
            @staticmethod
            def propose(context):
                return OverridingProposal(
                    valid=True,
                    candidate_skill="changed",
                    operation="replace",
                    rationale="forged",
                    attempted_calls=1,
                    completed_calls=1,
                )

        context = OptimizationContext(
            round_no=1,
            current_skill="skill",
            current_sha256="0" * 64,
            feedback_categories=("failure",),
            training=EvaluationOutcome(
                valid=True,
                hard=1.0,
                soft=1.0,
                feedback=(),
                reference="fixture://training",
                attempted_calls=1,
                completed_calls=1,
            ),
            history=(),
        )
        proposal, proposal_known = _optimizer_boundary(
            Optimizer(),
            context,
            max_skill_chars=100,
        )
        self.assertFalse(proposal_known)
        self.assertEqual(proposal.failure_class, "optimizer_invalid")

    def test_evaluation_feedback_is_deeply_detached_and_read_only(self) -> None:
        source = {
            "category": "failure",
            "metrics": {"score": 1.0},
        }

        class Evaluator:
            @staticmethod
            def evaluate(skill, *, split, round_no):
                return EvaluationOutcome(
                    valid=True,
                    hard=1.0,
                    soft=1.0,
                    feedback=(source,),
                    reference="fixture://evaluation",
                    attempted_calls=1,
                    completed_calls=1,
                )

        outcome, accounting_known = _evaluation_boundary(
            Evaluator(),
            "skill",
            split="train",
            round_no=1,
            stage="training",
        )
        source["category"] = "forged"
        source["metrics"]["score"] = 0.0
        self.assertTrue(accounting_known)
        self.assertEqual(
            outcome.public()["feedback"],
            [{"category": "failure", "metrics": {"score": 1.0}}],
        )
        with self.assertRaises(TypeError):
            outcome.feedback[0]["category"] = "forged"  # type: ignore[index]
        metrics = outcome.feedback[0]["metrics"]
        with self.assertRaises(TypeError):
            metrics["score"] = 0.0  # type: ignore[index]

    def test_evaluation_rejects_invalid_call_accounting(self) -> None:
        outcome = EvaluationOutcome(
            valid=True,
            hard=1.0,
            soft=1.0,
            feedback=(),
            reference="fixture://one",
            attempted_calls=0,
            completed_calls=1,
        )
        with self.assertRaisesRegex(
            ResultValidationError,
            "0 <= completed <= attempted",
        ):
            outcome.validate()

    def test_evaluation_rejects_malformed_feedback(self) -> None:
        outcome = EvaluationOutcome(
            valid=True,
            hard=1.0,
            soft=1.0,
            feedback=("not-a-mapping",),  # type: ignore[arg-type]
            reference="fixture://one",
            attempted_calls=1,
            completed_calls=1,
        )
        with self.assertRaisesRegex(
            ResultValidationError,
            "tuple of mappings",
        ):
            outcome.validate()

    def test_proposal_operation_and_content_must_agree(self) -> None:
        proposal = ProposalOutcome(
            valid=True,
            candidate_skill="changed",
            operation="noop",
            rationale="invalid fixture",
            attempted_calls=1,
            completed_calls=1,
        )
        with self.assertRaisesRegex(
            ResultValidationError,
            "noop proposal must preserve",
        ):
            proposal.validate(
                current_skill="current",
                max_skill_chars=100,
            )

    def test_options_reject_duplicate_feedback_categories(self) -> None:
        options = EvolutionOptions(
            rounds=1,
            arm="normal",
            feedback_categories=("failure", "failure"),
        )
        with self.assertRaisesRegex(
            ConfigurationError,
            "contain duplicates",
        ):
            options.validate()

    def test_options_reject_wrong_runtime_types(self) -> None:
        invalid = (
            EvolutionOptions(
                rounds="1",  # type: ignore[arg-type]
                arm="normal",
                feedback_categories=("failure",),
            ),
            EvolutionOptions(
                rounds=1,
                arm=1,  # type: ignore[arg-type]
                feedback_categories=("failure",),
            ),
            EvolutionOptions(
                rounds=1,
                arm="normal",
                feedback_categories=["failure"],  # type: ignore[arg-type]
            ),
            EvolutionOptions(
                rounds=1,
                arm="normal",
                feedback_categories=(1,),  # type: ignore[arg-type]
            ),
            EvolutionOptions(
                rounds=1,
                arm="normal",
                feedback_categories=("failure",),
                max_skill_chars="100",  # type: ignore[arg-type]
            ),
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ConfigurationError):
                options.validate()

    def test_outcomes_reject_non_json_feedback_and_invalid_limits(self) -> None:
        outcome = EvaluationOutcome(
            valid=True,
            hard=1.0,
            soft=1.0,
            feedback=({"score": float("nan")},),
            reference="fixture://one",
            attempted_calls=1,
            completed_calls=1,
        )
        with self.assertRaisesRegex(
            ResultValidationError,
            "finite canonical JSON",
        ):
            outcome.validate()

        proposal = ProposalOutcome(
            valid=True,
            candidate_skill="changed",
            operation="replace",
            rationale="reason",
            attempted_calls=1,
            completed_calls=1,
        )
        with self.assertRaisesRegex(
            ResultValidationError,
            "positive integer",
        ):
            proposal.validate(
                current_skill="current",
                max_skill_chars=True,
            )

    def test_feedback_filter_preserves_outcome_provenance(self) -> None:
        outcome = EvaluationOutcome(
            valid=True,
            hard=0.5,
            soft=0.75,
            feedback=(
                {"category": "success", "id": "one"},
                {"category": "failure", "id": "two"},
            ),
            reference="fixture://training",
            attempted_calls=2,
            completed_calls=2,
        )
        filtered = filter_feedback(
            outcome.feedback,
            ("failure",),
        )
        updated = with_feedback(outcome, filtered)
        self.assertEqual(
            updated.feedback,
            ({"category": "failure", "id": "two"},),
        )
        self.assertEqual(updated.reference, outcome.reference)
        self.assertEqual(
            updated.attempted_calls,
            outcome.attempted_calls,
        )


if __name__ == "__main__":
    unittest.main()
