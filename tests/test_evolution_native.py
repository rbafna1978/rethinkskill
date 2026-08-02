"""Unit tests for native-ledger to evolution-outcome aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rethinkskill.errors import ResultValidationError
from rethinkskill.evolution.native import evaluation_outcome


class NativeEvolutionOutcomeTests(unittest.TestCase):
    def _write_rows(
        self,
        directory: str,
        rows: tuple[dict[str, object], ...],
    ) -> Path:
        path = Path(directory) / "results.jsonl"
        path.write_text(
            "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_valid_outcome_aggregates_scores_and_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = self._write_rows(
                directory,
                (
                    {
                        "case_id": "pass",
                        "hard": 1,
                        "soft": 1.0,
                        "answer": "A",
                    },
                    {
                        "case_id": "fail",
                        "hard": 0,
                        "soft": 0.5,
                        "answer": "B",
                    },
                ),
            )
            outcome = evaluation_outcome(
                receipt={
                    "status": "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED",
                    "attempted_calls": 2,
                    "completed_calls": 2,
                    "call_accounting_known": True,
                    "selected_tasks": 2,
                    "executed_tasks": 2,
                    "ledger_validation": {"status": "STRUCTURALLY_VALID"},
                },
                results_path=results,
                selected_tasks=2,
            )
        self.assertTrue(outcome.valid)
        self.assertEqual(outcome.hard, 0.5)
        self.assertEqual(outcome.soft, 0.75)
        self.assertEqual(
            tuple(item["category"] for item in outcome.feedback),
            ("success", "failure"),
        )
        self.assertIn("#sha256=", outcome.reference)

    def test_infrastructure_row_preserves_failure_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = self._write_rows(
                directory,
                (
                    {
                        "case_id": "broken",
                        "hard": 0,
                        "soft": 0.0,
                        "failure_class": "provider_timeout",
                        "fail_reason": "timed out",
                    },
                ),
            )
            outcome = evaluation_outcome(
                receipt={
                    "status": "RETHINKSKILL_NATIVE_EXECUTION_INVALID",
                    "attempted_calls": 1,
                    "completed_calls": 0,
                    "call_accounting_known": True,
                    "selected_tasks": 1,
                    "executed_tasks": 1,
                    "ledger_validation": {"status": "STRUCTURALLY_VALID"},
                },
                results_path=results,
                selected_tasks=1,
            )
        self.assertFalse(outcome.valid)
        self.assertEqual(outcome.failure_class, "provider_timeout")
        self.assertEqual(outcome.failure, "timed out")
        self.assertEqual(outcome.feedback[0]["category"], "infrastructure")

    def test_receipt_call_counts_reject_python_coercions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = self._write_rows(
                directory,
                ({"case_id": "one", "hard": 1, "soft": 1.0},),
            )
            for invalid in (False, 1.0, "1"):
                with (
                    self.subTest(invalid=invalid),
                    self.assertRaisesRegex(
                        ResultValidationError,
                        "0 <= completed <= attempted",
                    ),
                ):
                    evaluation_outcome(
                        receipt={
                            "status": ("RETHINKSKILL_NATIVE_EXECUTION_VALIDATED"),
                            "attempted_calls": invalid,
                            "completed_calls": 1,
                            "call_accounting_known": True,
                            "selected_tasks": 1,
                            "executed_tasks": 1,
                            "ledger_validation": {"status": "STRUCTURALLY_VALID"},
                        },
                        results_path=results,
                        selected_tasks=1,
                    )

    def test_unknown_receipt_call_accounting_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = self._write_rows(
                directory,
                ({"case_id": "one", "hard": 0, "soft": 0.0},),
            )
            with self.assertRaisesRegex(
                ResultValidationError,
                "call accounting is unknown",
            ):
                evaluation_outcome(
                    receipt={
                        "status": "RETHINKSKILL_NATIVE_EXECUTION_INVALID",
                        "attempted_calls": 1,
                        "completed_calls": 1,
                        "call_accounting_known": False,
                        "selected_tasks": 1,
                        "executed_tasks": 1,
                        "ledger_validation": {"status": "INFRASTRUCTURE_INVALID_ROWS_PRESENT"},
                    },
                    results_path=results,
                    selected_tasks=1,
                )

    def test_ledger_scores_reject_bool_and_string_coercions(self) -> None:
        receipt = {
            "status": "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED",
            "attempted_calls": 1,
            "completed_calls": 1,
            "call_accounting_known": True,
            "selected_tasks": 1,
            "executed_tasks": 1,
            "ledger_validation": {"status": "STRUCTURALLY_VALID"},
        }
        for field, invalid in (("hard", True), ("soft", "1.0")):
            with tempfile.TemporaryDirectory() as directory:
                row = {"case_id": "one", "hard": 1, "soft": 1.0}
                row[field] = invalid
                results = self._write_rows(directory, (row,))
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(
                        ResultValidationError,
                        "finite score in",
                    ),
                ):
                    evaluation_outcome(
                        receipt=receipt,
                        results_path=results,
                        selected_tasks=1,
                    )

    def test_valid_status_with_summary_mismatch_becomes_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = self._write_rows(
                directory,
                ({"case_id": "one", "hard": 1, "soft": 1.0},),
            )
            outcome = evaluation_outcome(
                receipt={
                    "status": "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED",
                    "attempted_calls": 1,
                    "completed_calls": 1,
                    "call_accounting_known": True,
                    "selected_tasks": 2,
                    "executed_tasks": 1,
                    "ledger_validation": {"status": "STRUCTURALLY_VALID"},
                },
                results_path=results,
                selected_tasks=1,
            )
        self.assertFalse(outcome.valid)
        self.assertEqual(outcome.failure_class, "native_evaluation_invalid")

    def test_duplicate_case_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = self._write_rows(
                directory,
                (
                    {"case_id": "same", "hard": 1, "soft": 1.0},
                    {"case_id": "same", "hard": 0, "soft": 0.0},
                ),
            )
            with self.assertRaisesRegex(
                ResultValidationError,
                "duplicate case_id",
            ):
                evaluation_outcome(
                    receipt={
                        "status": "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED",
                        "attempted_calls": 2,
                        "completed_calls": 2,
                        "call_accounting_known": True,
                        "selected_tasks": 2,
                        "executed_tasks": 2,
                        "ledger_validation": {"status": "STRUCTURALLY_VALID"},
                    },
                    results_path=results,
                    selected_tasks=2,
                )


if __name__ == "__main__":
    unittest.main()
