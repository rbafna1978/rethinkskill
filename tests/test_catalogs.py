from __future__ import annotations

import unittest
from types import MappingProxyType
from unittest.mock import patch

from rethinkskill.benchmarks.capabilities import CapabilityCatalog, capability_catalog
from rethinkskill.benchmarks.core import BenchmarkCatalog, benchmark_catalog
from rethinkskill.benchmarks.harness_catalog import HarnessCatalog, native_harness_catalog
from rethinkskill.errors import ConfigurationError
from rethinkskill.evolution.optimizer import OptimizerCatalog, optimizer_catalog
from rethinkskill.integrations.catalog import IntegrationCatalog
from rethinkskill.providers.catalog import ProviderCatalog, provider_catalog


class CatalogSealingTests(unittest.TestCase):
    def test_assembly_catalogs_are_sealed_and_publish_state(self) -> None:
        catalogs = (
            benchmark_catalog(),
            native_harness_catalog(),
            provider_catalog(),
            optimizer_catalog(),
        )
        expected_versions = (4, 3, 5, 3)
        for catalog, schema_version in zip(
            catalogs,
            expected_versions,
            strict=True,
        ):
            with self.subTest(catalog=type(catalog).__name__):
                self.assertTrue(catalog.sealed)
                self.assertTrue(catalog.manifest()["sealed"])
                self.assertEqual(
                    catalog.manifest()["schema_version"],
                    schema_version,
                )
                self.assertIs(catalog.seal(), catalog)
                self.assertIsInstance(catalog._specs, MappingProxyType)

    def test_direct_catalogs_remain_mutable_until_explicit_seal(self) -> None:
        builtin = capability_catalog().resolve("searchqa")
        provider = provider_catalog().resolve("codex")
        optimizer = optimizer_catalog().resolve("model-skill")
        catalogs_and_specs = (
            (BenchmarkCatalog(), builtin.scoring),
            (HarnessCatalog(), builtin.native),
            (ProviderCatalog(), provider),
            (OptimizerCatalog(), optimizer),
        )
        for catalog, spec in catalogs_and_specs:
            with self.subTest(catalog=type(catalog).__name__):
                self.assertFalse(catalog.sealed)
                catalog.register(spec)
                self.assertEqual(len(catalog.names()), 1)
                catalog.seal()
                self.assertTrue(catalog.sealed)
                with self.assertRaisesRegex(
                    ConfigurationError,
                    "catalog is sealed",
                ):
                    catalog.register(spec)

    def test_sealed_catalog_rejects_plugin_discovery_before_metadata_access(
        self,
    ) -> None:
        for catalog in (
            BenchmarkCatalog().seal(),
            HarnessCatalog().seal(),
            ProviderCatalog().seal(),
            OptimizerCatalog().seal(),
        ):
            with (
                self.subTest(catalog=type(catalog).__name__),
                patch("rethinkskill.utils.plugins.metadata.entry_points") as entry_points,
                self.assertRaisesRegex(
                    ConfigurationError,
                    "catalog is sealed",
                ),
            ):
                catalog.load_entry_points()
            entry_points.assert_not_called()

    def test_capability_selection_seals_supplied_catalogs(self) -> None:
        builtin = capability_catalog().resolve("searchqa")
        scorers = BenchmarkCatalog((builtin.scoring,))
        harnesses = HarnessCatalog((builtin.native,))
        integrations = IntegrationCatalog((builtin.integration,))
        selected = CapabilityCatalog(
            scorers,
            harnesses,
            integrations,
        )
        self.assertTrue(scorers.sealed)
        self.assertTrue(harnesses.sealed)
        self.assertTrue(selected.manifest()["sealed"])
        self.assertEqual(selected.manifest()["schema_version"], 5)
        with self.assertRaisesRegex(ConfigurationError, "catalog is sealed"):
            scorers.register(builtin.scoring)


if __name__ == "__main__":
    unittest.main()
