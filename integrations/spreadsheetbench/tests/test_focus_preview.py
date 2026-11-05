"""FOCUS fork: workbook previews also work for sheets that do not store their size."""

from __future__ import annotations

import io
import re
import unittest
import zipfile

from rethinkskill_spreadsheetbench.runtime import preview_workbook_bytes

try:
    import openpyxl
except ImportError:  # pragma: no cover
    openpyxl = None


def _workbook(cells: dict[str, object], rows: int = 0) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    for ref, value in cells.items():
        ws[ref] = value
    for row in range(1, rows + 1):
        for column in range(1, 20):
            ws.cell(row, column).value = row * column
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def _strip_dimension(payload: bytes) -> bytes:
    """Remove <dimension .../> from every sheet, like some real benchmark files."""
    source = zipfile.ZipFile(io.BytesIO(payload))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename.startswith("xl/worksheets/sheet"):
                data = re.sub(rb"<dimension[^>]*/>", b"", data)
            target.writestr(item, data)
    return out.getvalue()


@unittest.skipIf(openpyxl is None, "openpyxl is required")
class FocusPreviewTests(unittest.TestCase):
    def test_unsized_sheet_matches_sized_preview(self):
        for payload in (_workbook({"A1": "x", "C2": 5}), _workbook({}, rows=40)):
            stripped = _strip_dimension(payload)
            sheet = openpyxl.load_workbook(io.BytesIO(stripped), read_only=True).active
            self.assertIsNone(sheet.max_row)  # proves the fixture is really unsized
            self.assertEqual(preview_workbook_bytes(stripped), preview_workbook_bytes(payload))

    def test_large_unsized_sheet_is_truncated_with_real_size(self):
        preview = preview_workbook_bytes(_strip_dimension(_workbook({}, rows=40)))
        self.assertIn("[preview truncated; dimensions=40x19]", preview)

    def test_empty_unsized_sheet_does_not_crash(self):
        preview = preview_workbook_bytes(_strip_dimension(_workbook({})))
        self.assertIn("## Sheet: Data", preview)


if __name__ == "__main__":
    unittest.main()
