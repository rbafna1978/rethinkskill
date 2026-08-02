from __future__ import annotations

import re
import unittest
from pathlib import Path

from rethinkskill import __version__
from rethinkskill.utils.fs import Repository


class VersionTests(unittest.TestCase):
    def test_package_and_project_versions_match(self) -> None:
        repository = Repository.discover(Path(__file__))
        project = (repository.root / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(
            r'(?m)^version = "([^"]+)"$',
            project,
        )
        self.assertIsNotNone(match)
        self.assertEqual(__version__, match.group(1))
