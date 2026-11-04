"""FOCUS fork: grader exceptions on a model's workbook are task failures, not crashes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rethinkskill.benchmarks import scoring
from rethinkskill.benchmarks.scoring import Verdict, verify_spreadsheet

try:
    import openpyxl
except ImportError:  # pragma: no cover
    openpyxl = None


@unittest.skipIf(openpyxl is None, "openpyxl is required")
class FocusGraderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name).resolve()
        self.pred, self.gold = root / "pred.xlsx", root / "gold.xlsx"
        for path, value in ((self.pred, 3), (self.gold, 3)):
            wb = openpyxl.Workbook()
            wb.active.title = "Sheet1"
            wb.active["B1"] = value
            wb.save(path)

    def test_matching_answer_still_passes(self):
        result = verify_spreadsheet(self.pred, self.gold, "Sheet1!B1")
        self.assertEqual(result.verdict, Verdict.PASS)

    def test_unparseable_range_is_a_failure_not_an_exception(self):
        result = verify_spreadsheet(self.pred, self.gold, "Sheet1!NOT_A_RANGE")
        self.assertEqual(result.verdict, Verdict.FAIL)
        self.assertEqual(result.reason, "compare_error:ValueError")

    def test_error_while_reading_cells_is_a_failure(self):
        with patch.object(scoring, "_values_match", side_effect=TypeError("weird cell")):
            result = verify_spreadsheet(self.pred, self.gold, "Sheet1!B1")
        self.assertEqual(result.verdict, Verdict.FAIL)
        self.assertEqual(result.reason, "compare_error:TypeError")


if __name__ == "__main__":
    unittest.main()
