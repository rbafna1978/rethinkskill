from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rethinkskill.errors import ResultValidationError
from rethinkskill.evaluation.results import (
    LedgerReport,
    load_ids,
    recovery_plan,
    validate_ledger,
    validate_ledger_snapshot,
)


class ResultLedgerTests(unittest.TestCase):
    def _write_rows(self, path: Path, rows: list[object]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_structurally_valid_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            self._write_rows(path, [{"id": "a", "hard": 1}, {"id": "b", "hard": 0}])
            report = validate_ledger(path, expected_ids=("a", "b"))
            self.assertEqual(report.status, "STRUCTURALLY_VALID")
            self.assertEqual(report.unique_ids, 2)

    def test_immutable_ledger_snapshot_is_independent_of_later_path_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            original = b'{"id":"original","hard":1}\n'
            path.write_bytes(original)
            path.write_bytes(b'{"id":"replacement","hard":0}\n')

            report = validate_ledger_snapshot(
                original,
                path=path,
                expected_ids=("original",),
            )

            self.assertEqual(report.status, "STRUCTURALLY_VALID")
            self.assertEqual(report.unknown_ids, ())
            self.assertEqual(report.missing_ids, ())

    def test_ledger_snapshot_rejects_bytes_subclasses_and_bytearray(
        self,
    ) -> None:
        class ForgedBytes(bytes):
            pass

        for payload in (bytearray(b"{}\n"), ForgedBytes(b"{}\n")):
            with (
                self.subTest(payload_type=type(payload).__name__),
                self.assertRaisesRegex(
                    ResultValidationError,
                    "snapshot must be exact bytes",
                ),
            ):
                validate_ledger_snapshot(  # type: ignore[arg-type]
                    payload,
                    path=Path("results.jsonl"),
                )

    def test_result_ledger_rejects_symlink_without_reading_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "outside.jsonl"
            self._write_rows(target, [{"id": "secret"}])
            path = root / "results.jsonl"
            path.symlink_to(target)
            report = validate_ledger(path, expected_ids=("expected",))
        self.assertEqual(
            report.parse_errors,
            ("results file must not be a symlink",),
        )
        self.assertEqual(report.missing_ids, ("expected",))
        self.assertEqual(report.unique_ids, 0)

    def test_duplicates_missing_and_unknown_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            self._write_rows(path, [{"id": "a"}, {"id": "a"}, {"id": "c"}])
            report = validate_ledger(path, expected_ids=("a", "b"))
            self.assertEqual(report.duplicate_ids, ("a",))
            self.assertEqual(report.missing_ids, ("b",))
            self.assertEqual(report.unknown_ids, ("c",))

    def test_invalid_row_and_expected_ids_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            self._write_rows(
                path,
                [
                    {"id": 1},
                    {"id": " valid "},
                    {"id": "contains\ncontrol"},
                    {"id": "valid"},
                ],
            )
            report = validate_ledger(path)
            self.assertEqual(report.unique_ids, 1)
            self.assertEqual(report.valid_json_rows, 1)
            self.assertEqual(len(report.parse_errors), 3)
            with self.assertRaisesRegex(
                ResultValidationError,
                "duplicate task ID",
            ):
                validate_ledger(
                    path,
                    expected_ids=("valid", "valid"),
                )

    def test_task_id_file_rejects_duplicates_instead_of_deduplicating(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.txt"
            path.write_text("one\none\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ResultValidationError,
                "duplicate task ID",
            ):
                load_ids(path)

    def test_explicit_empty_expected_set_marks_every_row_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            self._write_rows(path, [{"id": "a"}, {"id": "b"}])
            report = validate_ledger(path, expected_ids=())
            self.assertEqual(report.missing_ids, ())
            self.assertEqual(report.unknown_ids, ("a", "b"))
            self.assertEqual(report.status, "INVALID_OR_INCOMPLETE")

    def test_recovery_plan_only_lists_infrastructure_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            self._write_rows(
                path,
                [
                    {"id": "behavior", "hard": 0, "fail_reason": "wrong answer"},
                    {
                        "id": "infra",
                        "hard": 0,
                        "failure_class": "target_timeout",
                    },
                    {
                        "id": "environment",
                        "hard": 0,
                        "failure_class": "environment_error",
                    },
                    {
                        "id": "optimizer",
                        "hard": 0,
                        "failure_class": "optimizer_invalid",
                    },
                    {
                        "id": "artifact",
                        "hard": 0,
                        "failure_class": "artifact_runner_error",
                    },
                ],
            )
            plan = recovery_plan(validate_ledger(path))
            self.assertEqual(
                plan["eligible_ids"],
                ["artifact", "environment", "infra", "optimizer"],
            )
            self.assertEqual(plan["mutations_performed"], 0)

    def test_heuristic_text_is_reported_but_not_recovery_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            self._write_rows(
                path,
                [
                    {
                        "id": "heuristic",
                        "fail_reason": ("The expected answer discusses rate limit policy."),
                    },
                    {
                        "id": "structured",
                        "failure_class": "target_timeout",
                    },
                ],
            )
            report = validate_ledger(path)
            plan = recovery_plan(report)
            self.assertEqual(
                report.infrastructure_invalid_ids,
                ("heuristic", "structured"),
            )
            self.assertEqual(
                report.structured_infrastructure_invalid_ids,
                ("structured",),
            )
            self.assertEqual(
                report.heuristic_infrastructure_invalid_ids,
                ("heuristic",),
            )
            self.assertEqual(plan["eligible_ids"], ["structured"])
            self.assertEqual(
                plan["ineligible_heuristic_ids"],
                ["heuristic"],
            )

    def test_recovery_plan_requires_validated_report(self) -> None:
        with self.assertRaisesRegex(
            ResultValidationError,
            "requires a LedgerReport",
        ):
            recovery_plan(object())  # type: ignore[arg-type]

        class OverridingReport(LedgerReport):
            @property
            def status(self):
                return "FORGED"

        report = OverridingReport(
            path="fixture.jsonl",
            total_lines=0,
            valid_json_rows=0,
            unique_ids=0,
            duplicate_ids=(),
            infrastructure_invalid_ids=(),
            parse_errors=(),
            missing_ids=(),
            unknown_ids=(),
        )
        with self.assertRaisesRegex(
            ResultValidationError,
            "exact core type",
        ):
            recovery_plan(report)

    def test_recovery_plan_excludes_ambiguous_duplicate_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            self._write_rows(
                path,
                [
                    {"id": "ambiguous", "failure_class": "target_timeout"},
                    {"id": "ambiguous", "fail_reason": "wrong answer"},
                    {"id": "eligible", "failure_class": "transport_error"},
                ],
            )
            report = validate_ledger(path)
            plan = recovery_plan(report)
            self.assertEqual(
                report.infrastructure_invalid_ids,
                ("ambiguous", "eligible"),
            )
            self.assertEqual(plan["eligible_ids"], ["eligible"])
            self.assertEqual(
                plan["ineligible_duplicate_ids"],
                ["ambiguous"],
            )
