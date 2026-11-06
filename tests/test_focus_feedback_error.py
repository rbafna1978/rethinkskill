"""FOCUS fork: optimizer feedback carries the generated code's error."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rethinkskill.evolution.native import (
    GENERATED_CODE_ERROR_LIMIT,
    evaluation_outcome,
    generated_code_error,
)

RECEIPT = {
    "status": "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED",
    "attempted_calls": 4,
    "completed_calls": 4,
    "call_accounting_known": True,
    "selected_tasks": 4,
    "executed_tasks": 4,
    "ledger_validation": {"status": "STRUCTURALLY_VALID"},
}


def runner(*cases: dict, extraction_error: str = "") -> dict:
    return {"artifact_runner": {"extraction_error": extraction_error, "cases": list(cases)}}


CRASH = {
    "failure": "generated_code_nonzero_exit",
    "exception": "ModuleNotFoundError: No module named 'pandas'",
}


class GeneratedCodeErrorTests(unittest.TestCase):
    def test_crash_is_reported(self) -> None:
        row = {"execution": runner(CRASH, {"failure": "", "exception": ""})}
        self.assertEqual(
            generated_code_error(row),
            "generated_code_nonzero_exit: ModuleNotFoundError: No module named 'pandas'",
        )

    def test_duplicate_cases_appear_once_and_distinct_ones_join(self) -> None:
        other = {"failure": "generated_code_timeout", "exception": ""}
        row = {"execution": runner(CRASH, CRASH, other)}
        self.assertEqual(
            generated_code_error(row),
            "generated_code_nonzero_exit: ModuleNotFoundError: No module named 'pandas'"
            " | generated_code_timeout",
        )

    def test_extraction_error(self) -> None:
        row = {"execution": runner(extraction_error="spreadsheet_python_contract_error")}
        self.assertEqual(
            generated_code_error(row),
            "no runnable code extracted: spreadsheet_python_contract_error",
        )

    def test_missing_or_malformed_execution_is_empty(self) -> None:
        for row in (
            {},
            {"execution": None},
            {"execution": {"artifact_runner": "x"}},
            {"execution": {"artifact_runner": {"cases": "x"}}},
            {"execution": runner({"failure": 3, "exception": None}, "not a dict")},
        ):
            self.assertEqual(generated_code_error(row), "", row)

    def test_length_is_capped(self) -> None:
        long = {"failure": "generated_code_nonzero_exit", "exception": "E: " + "x" * 1000}
        self.assertEqual(
            len(generated_code_error({"execution": runner(long)})), GENERATED_CODE_ERROR_LIMIT
        )

    def test_feedback_items_carry_error_only_for_failures(self) -> None:
        rows = (
            {"case_id": "ok", "hard": 1, "soft": 1.0, "execution": runner(CRASH)},
            {
                "case_id": "crash",
                "hard": 0,
                "soft": 0.0,
                "execution": runner(CRASH),
                "reason": "spreadsheet_case_failure:workbook_missing",
            },
            {
                "case_id": "wrong",
                "hard": 0,
                "soft": 0.0,
                "execution": runner({"failure": "", "exception": ""}),
            },
            {"case_id": "plain", "hard": 0, "soft": 0.0},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            outcome = evaluation_outcome(receipt=RECEIPT, results_path=path, selected_tasks=4)
        errors = {item["case_id"]: item["error"] for item in outcome.feedback}
        self.assertEqual(
            errors,
            {
                "ok": "",
                "crash": "generated_code_nonzero_exit: ModuleNotFoundError: No module named 'pandas'",
                "wrong": "",
                "plain": "",
            },
        )
        public = json.dumps(outcome.public())
        self.assertIn("No module named 'pandas'", public)


if __name__ == "__main__":
    unittest.main()
