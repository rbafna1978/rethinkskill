from __future__ import annotations

import unittest

from rethinkskill_spreadsheetbench.plugin import spec

from rethinkskill.benchmarks.capabilities import CapabilityCatalog
from rethinkskill.benchmarks.core import benchmark_catalog
from rethinkskill.benchmarks.harness_catalog import EvaluationAuthority, HarnessCatalog
from rethinkskill.integrations.catalog import integration_catalog


class SpreadsheetBenchPluginTests(unittest.TestCase):
    def test_plugin_supplies_artifact_harness(self) -> None:
        harness = spec()
        harness.validate()
        self.assertEqual(harness.name, "spreadsheetbench")
        self.assertIs(
            harness.evaluation_authority,
            EvaluationAuthority.HARNESS,
        )

    def test_plugin_joins_the_core_capability_catalog(self) -> None:
        catalog = CapabilityCatalog(
            benchmark_catalog(),
            HarnessCatalog((spec(),)),
            integration_catalog(),
        )
        capability = catalog.resolve("spreadsheetbench")
        self.assertTrue(capability.scorer_available)
        self.assertTrue(capability.adapter_available)
        self.assertTrue(capability.evaluation_ready)
        self.assertEqual(
            capability.public()["native_execution"]["evaluation_authority"],
            "harness",
        )
        self.assertEqual(
            capability.integration.adapter_distribution,
            "rethinkskill-spreadsheetbench",
        )


if __name__ == "__main__":
    unittest.main()
