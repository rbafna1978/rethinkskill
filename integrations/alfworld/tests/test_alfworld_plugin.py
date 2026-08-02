from __future__ import annotations

import unittest

from rethinkskill_alfworld.plugin import spec

from rethinkskill.benchmarks.capabilities import CapabilityCatalog
from rethinkskill.benchmarks.core import benchmark_catalog
from rethinkskill.benchmarks.harness_catalog import (
    EvaluationAuthority,
    HarnessCatalog,
)
from rethinkskill.integrations.catalog import integration_catalog


class AlfWorldPluginTests(unittest.TestCase):
    def test_plugin_supplies_harness_owned_evaluation(self) -> None:
        harness = spec()
        harness.validate()
        self.assertEqual(harness.name, "alfworld")
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
        capability = catalog.resolve("alfworld")
        self.assertFalse(capability.scorer_available)
        self.assertTrue(capability.adapter_available)
        self.assertTrue(capability.evaluation_ready)
        self.assertEqual(
            capability.integration.adapter_distribution,
            "rethinkskill-alfworld",
        )


if __name__ == "__main__":
    unittest.main()
