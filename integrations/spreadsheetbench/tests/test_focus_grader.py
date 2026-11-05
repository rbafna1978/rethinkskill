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


def _old_verify(prediction, reference, answer_position):
    """Upstream comparison (one .cell() per cell), kept here as the reference."""
    from openpyxl.utils.cell import range_boundaries

    predicted = openpyxl.load_workbook(prediction, data_only=True, read_only=True)
    expected = openpyxl.load_workbook(reference, data_only=True, read_only=True)
    checked = 0
    try:
        for raw in str(answer_position or "").split(","):
            part = raw.strip()
            if not part:
                continue
            if "!" in part:
                sheet, cell_range = part.split("!", 1)
                sheet = sheet.strip().strip("'\"")
            else:
                sheet, cell_range = (expected.sheetnames[0], part)
            cell_range = cell_range.strip().strip("'\"")
            if sheet not in predicted.sheetnames or sheet not in expected.sheetnames:
                return ("FAIL", f"sheet_missing:{sheet}", checked)
            min_col, min_row, max_col, max_row = range_boundaries(cell_range)
            for row in range(min_row, max_row + 1):
                for column in range(min_col, max_col + 1):
                    gold = expected[sheet].cell(row, column).value
                    seen = predicted[sheet].cell(row, column).value
                    checked += 1
                    if not scoring._values_match(seen, gold):
                        return ("FAIL", f"value_mismatch:{sheet}!{row}:{column}", checked)
    finally:
        predicted.close()
        expected.close()
    if checked == 0:
        return ("ABSTAIN", "answer_position_empty_or_unparseable", 0)
    return ("PASS", "official_answer_cell_values_match", checked)


@unittest.skipIf(openpyxl is None, "openpyxl is required")
class FocusGraderSpeedTests(unittest.TestCase):
    def test_same_results_as_upstream_on_random_workbooks(self):
        import random

        rnd = random.Random(7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for trial in range(120):
                paths = []
                for name in ("pred", "gold"):
                    wb = openpyxl.Workbook()
                    ws = wb.active
                    ws.title = "S"
                    other = wb.create_sheet("T")
                    for _ in range(rnd.randint(0, 60)):
                        target = ws if rnd.random() < 0.8 else other
                        target.cell(rnd.randint(1, 25), rnd.randint(1, 8)).value = rnd.choice(
                            [rnd.randint(0, 3), 1.5, "a", "b", None]
                        )
                    path = root / f"{name}{trial}.xlsx"
                    wb.save(path)
                    paths.append(path)
                r0, c0 = rnd.randint(1, 20), rnd.randint(1, 6)
                rng = f"{openpyxl.utils.get_column_letter(c0)}{r0}:" + (
                    f"{openpyxl.utils.get_column_letter(c0 + rnd.randint(0, 4))}"
                    f"{r0 + rnd.randint(0, 12)}"
                )
                position = rnd.choice([f"S!{rng}", f"'S'!{rng}, T!A1:B3", rng, f"T!{rng}"])
                new = verify_spreadsheet(paths[0], paths[1], position)
                self.assertEqual(
                    (new.verdict.name, new.reason, new.checked),
                    _old_verify(paths[0], paths[1], position),
                    position,
                )

    def test_large_range_is_fast(self):
        import time

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / "big.xlsx"
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "S"
            for row in range(1, 1201):
                for column in range(1, 16):
                    ws.cell(row, column).value = row * column
            wb.save(path)
            started = time.monotonic()
            result = verify_spreadsheet(path, path, "S!A1:O1200")
            elapsed = time.monotonic() - started
        self.assertEqual(result.verdict, Verdict.PASS)
        self.assertEqual(result.checked, 18000)
        self.assertLess(elapsed, 10.0)


if __name__ == "__main__":
    unittest.main()
