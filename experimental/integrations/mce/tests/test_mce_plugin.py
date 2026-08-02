from __future__ import annotations

import unittest

from rethinkskill_mce.plugin import benchmark_specs

from rethinkskill.benchmarks.core import EvaluationMode
from rethinkskill.integrations.mce import (
    benchmark_specs as declarations,
)


class MCEPluginTests(unittest.TestCase):
    def test_plugin_materializations_preserve_declaration_metadata(self) -> None:
        declared = declarations()
        materialized = benchmark_specs()
        self.assertEqual(len(declared), len(materialized))
        for declaration, scorer in zip(declared, materialized, strict=True):
            self.assertIs(declaration.mode, EvaluationMode.DECLARED_ONLY)
            self.assertIs(scorer.mode, EvaluationMode.CASES)
            self.assertEqual(declaration.name, scorer.name)
            self.assertEqual(declaration.family, scorer.family)
            self.assertEqual(declaration.domain, scorer.domain)
            self.assertEqual(declaration.metrics, scorer.metrics)
            self.assertEqual(declaration.sources, scorer.sources)
            self.assertEqual(declaration.notes, scorer.notes)
            self.assertEqual(declaration.artifact_fields, scorer.artifact_fields)


if __name__ == "__main__":
    unittest.main()
