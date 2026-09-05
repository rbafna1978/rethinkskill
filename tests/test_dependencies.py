from __future__ import annotations

import unittest
from importlib import metadata
from unittest import mock

from rethinkskill.runtime.types import numeric_version, probe_distribution


class DependencyProbeTests(unittest.TestCase):
    def test_numeric_release_prefix(self) -> None:
        self.assertEqual(numeric_version("3.1.5"), (3, 1, 5))
        self.assertEqual(numeric_version("0.4.2+local"), (0, 4, 2))
        self.assertIsNone(numeric_version("release"))

    @mock.patch("rethinkskill.runtime.types.metadata.version", return_value="3.1.5")
    @mock.patch("rethinkskill.runtime.types.util.find_spec", return_value=object())
    def test_probe_requires_module_and_compatible_version(self, _find_spec, _version) -> None:
        value = probe_distribution(
            distribution="example",
            import_name="example",
            requirement=">=3.1,<4",
            accepts=lambda version: numeric_version(version) >= (3, 1),
        )
        self.assertTrue(value["ready"])
        self.assertEqual(value["version"], "3.1.5")
        self.assertEqual(value["requirement"], ">=3.1,<4")

    @mock.patch(
        "rethinkskill.runtime.types.metadata.version",
        side_effect=metadata.PackageNotFoundError,
    )
    @mock.patch("rethinkskill.runtime.types.util.find_spec", return_value=None)
    def test_absent_distribution_is_not_ready(self, _find_spec, _version) -> None:
        value = probe_distribution(
            distribution="missing",
            import_name="missing",
            requirement="==1",
            accepts=lambda version: version == "1",
        )
        self.assertFalse(value["ready"])
        self.assertFalse(value["module_available"])
        self.assertIsNone(value["version"])


if __name__ == "__main__":
    unittest.main()
