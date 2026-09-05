from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rethinkskill.benchmarks.core import (
    DETERMINISTIC_BENCHMARKS,
    BenchmarkCatalog,
    BenchmarkSource,
    BenchmarkSpec,
    EvaluationMode,
    benchmark_catalog,
    builtin_specs,
    case_adapter,
)
from rethinkskill.benchmarks.harness_catalog import HarnessCatalog, NativeHarnessSpec
from rethinkskill.benchmarks.qa import SearchQAHarness
from rethinkskill.benchmarks.scoring import Verdict, Verification
from rethinkskill.errors import (
    ConfigurationError,
    ResultValidationError,
)


class BenchmarkCaseAdapterTests(unittest.TestCase):
    @staticmethod
    def _skillrl_entry_points(
        specs: tuple[BenchmarkSpec, ...],
        *,
        distribution: str = "rethinkskill-skillrl",
    ):
        class EntryPoint:
            name = "skillrl-offline"
            value = "rethinkskill_skillrl.plugin:benchmark_specs"
            dist = type(
                "Distribution",
                (),
                {
                    "metadata": {"Name": distribution},
                    "version": "0.1.0",
                    "files": None,
                },
            )()

            @staticmethod
            def load():
                return specs

        class EntryPoints:
            @staticmethod
            def select(*, group):
                if group != "rethinkskill.benchmarks":
                    raise AssertionError(group)
                return (EntryPoint(),)

        return EntryPoints()

    @staticmethod
    def _mce_entry_points(
        specs: tuple[BenchmarkSpec, ...],
        *,
        distribution: str = "rethinkskill-mce",
    ):
        class EntryPoint:
            name = "mce"
            value = "rethinkskill_mce.plugin:benchmark_specs"
            dist = type(
                "Distribution",
                (),
                {
                    "metadata": {"Name": distribution},
                    "version": "0.1.0",
                    "files": None,
                },
            )()

            @staticmethod
            def load():
                return specs

        class EntryPoints:
            @staticmethod
            def select(*, group):
                if group != "rethinkskill.benchmarks":
                    raise AssertionError(group)
                return (EntryPoint(),)

        return EntryPoints()

    def test_built_in_case_adapters_are_explicit(self) -> None:
        self.assertEqual(
            DETERMINISTIC_BENCHMARKS,
            (
                "searchqa",
                "officeqa",
                "docvqa",
                "livemath",
                "spreadsheetbench",
            ),
        )
        self.assertTrue(all(callable(case_adapter(name)) for name in DETERMINISTIC_BENCHMARKS))

    def test_catalog_separates_local_and_external_harnesses(self) -> None:
        catalog = benchmark_catalog()
        self.assertEqual(
            catalog.names(locally_evaluable=False),
            (
                "alfworld",
                "nq",
                "triviaqa",
                "popqa",
                "hotpotqa",
                "2wiki",
                "musique",
                "bamboogle",
                "finer",
                "uspto50k",
                "symptom2disease",
                "lawbench-charge",
                "aegis2",
                "webshop",
                "mcp-atlas",
            ),
        )
        manifest = catalog.manifest()
        self.assertEqual(
            manifest["counts"],
            {
                "total": 20,
                "locally_evaluable": 5,
                "external_harness": 3,
                "declared_only": 12,
            },
        )
        with self.assertRaisesRegex(
            ConfigurationError,
            "requires an external harness",
        ):
            case_adapter("webshop")

    def test_duplicate_plugin_registration_fails_closed(self) -> None:
        spec = BenchmarkSpec(
            name="custom",
            family="test",
            domain="test",
            mode=EvaluationMode.CASES,
            metrics=("pass_rate",),
            adapter=lambda row, base: case_adapter("searchqa")(row, base),
        )
        catalog = BenchmarkCatalog((spec,))
        with self.assertRaisesRegex(
            ConfigurationError,
            "duplicate benchmark",
        ):
            catalog.register(spec)

    def test_declared_only_spec_cannot_embed_executable_scoring(self) -> None:
        spec = BenchmarkSpec(
            name="declared",
            family="test",
            domain="test",
            mode=EvaluationMode.DECLARED_ONLY,
            metrics=("pass_rate",),
            adapter=lambda row, base: Verification(Verdict.PASS, "ok"),
        )
        with self.assertRaisesRegex(ConfigurationError, "declared-only"):
            spec.validate()

    def test_official_plugin_materializes_skillrl_declarations_atomically(
        self,
    ) -> None:
        from rethinkskill.integrations.skillrl import (
            benchmark_specs as skillrl_declarations,
        )

        candidates = tuple(
            replace(
                declaration,
                mode=EvaluationMode.CASES,
                adapter=lambda row, base: Verification(Verdict.PASS, "ok"),
            )
            for declaration in skillrl_declarations()
        )
        catalog = BenchmarkCatalog(builtin_specs(), source="builtin")
        with patch(
            "rethinkskill.utils.plugins.metadata.entry_points",
            return_value=self._skillrl_entry_points(candidates),
        ):
            loaded = catalog.load_entry_points()
        self.assertEqual(loaded, tuple(spec.name for spec in candidates))
        self.assertEqual(len(catalog.names()), 20)
        self.assertEqual(len(catalog.names(locally_evaluable=True)), 12)
        for candidate in candidates:
            self.assertIs(catalog.resolve(candidate.name), candidate)
            self.assertEqual(
                catalog.registration(candidate.name).distribution_name,
                "rethinkskill-skillrl",
            )

    def test_skillrl_materialization_rejects_wrong_distribution_and_drift(
        self,
    ) -> None:
        from rethinkskill.integrations.skillrl import (
            benchmark_specs as skillrl_declarations,
        )

        declarations = skillrl_declarations()
        candidates = tuple(
            replace(
                declaration,
                mode=EvaluationMode.CASES,
                adapter=lambda row, base: Verification(Verdict.PASS, "ok"),
            )
            for declaration in declarations
        )
        wrong_distribution = BenchmarkCatalog(builtin_specs(), source="builtin")
        with (
            patch(
                "rethinkskill.utils.plugins.metadata.entry_points",
                return_value=self._skillrl_entry_points(
                    candidates,
                    distribution="unrelated-package",
                ),
            ),
            self.assertRaisesRegex(ConfigurationError, "materialization rejected"),
        ):
            wrong_distribution.load_entry_points()
        self.assertIs(
            wrong_distribution.resolve("nq").mode,
            EvaluationMode.DECLARED_ONLY,
        )

        drifted = (
            replace(candidates[0], metrics=("pass_rate",)),
            *candidates[1:],
        )
        metadata_drift = BenchmarkCatalog(builtin_specs(), source="builtin")
        with (
            patch(
                "rethinkskill.utils.plugins.metadata.entry_points",
                return_value=self._skillrl_entry_points(drifted),
            ),
            self.assertRaisesRegex(ConfigurationError, "materialization rejected"),
        ):
            metadata_drift.load_entry_points()
        self.assertTrue(
            all(
                metadata_drift.resolve(name).mode is EvaluationMode.DECLARED_ONLY
                for name in tuple(spec.name for spec in declarations)
            )
        )

        partial = BenchmarkCatalog(builtin_specs(), source="builtin")
        with (
            patch(
                "rethinkskill.utils.plugins.metadata.entry_points",
                return_value=self._skillrl_entry_points(candidates[:1]),
            ),
            self.assertRaisesRegex(ConfigurationError, "suite materialization"),
        ):
            partial.load_entry_points()
        self.assertIs(partial.resolve("nq").mode, EvaluationMode.DECLARED_ONLY)

    def test_mce_materialization_is_owner_bound_and_atomic(self) -> None:
        from rethinkskill.integrations.mce import (
            benchmark_specs as mce_declarations,
        )

        candidates = tuple(
            replace(
                declaration,
                mode=EvaluationMode.CASES,
                adapter=lambda row, base: Verification(Verdict.PASS, "ok"),
            )
            for declaration in mce_declarations()
        )
        catalog = BenchmarkCatalog(builtin_specs(), source="builtin")
        with patch(
            "rethinkskill.utils.plugins.metadata.entry_points",
            return_value=self._mce_entry_points(candidates),
        ):
            loaded = catalog.load_entry_points()
        self.assertEqual(loaded, tuple(spec.name for spec in candidates))
        self.assertEqual(len(catalog.names(locally_evaluable=True)), 10)
        self.assertTrue(
            all(
                catalog.registration(spec.name).distribution_name == "rethinkskill-mce"
                for spec in candidates
            )
        )

        partial = BenchmarkCatalog(builtin_specs(), source="builtin")
        with (
            patch(
                "rethinkskill.utils.plugins.metadata.entry_points",
                return_value=self._mce_entry_points(candidates[:1]),
            ),
            self.assertRaisesRegex(ConfigurationError, "suite materialization"),
        ):
            partial.load_entry_points()
        self.assertTrue(
            all(
                partial.resolve(spec.name).mode is EvaluationMode.DECLARED_ONLY
                for spec in candidates
            )
        )

    def test_plugin_reducer_cannot_emit_non_finite_metrics(self) -> None:
        spec = BenchmarkSpec(
            name="custom",
            family="test",
            domain="test",
            mode=EvaluationMode.CASES,
            metrics=("score",),
            adapter=lambda row, base: Verification(Verdict.PASS, "ok"),
            reducer=lambda rows: {"score": float("nan")},
        )
        with self.assertRaisesRegex(
            ConfigurationError,
            "non-finite numeric metric",
        ):
            spec.summarize((Verification(Verdict.PASS, "ok"),))

    def test_entry_point_plugins_load_only_after_explicit_call(self) -> None:
        spec = BenchmarkSpec(
            name="plugin-example",
            family="test",
            domain="test",
            mode=EvaluationMode.CASES,
            metrics=("pass_rate",),
            adapter=lambda row, base: case_adapter("searchqa")(row, base),
        )

        class EntryPoint:
            name = "plugin-example"
            value = "plugin_package:SPEC"
            dist = type(
                "Distribution",
                (),
                {
                    "metadata": {"Name": "rethinkskill-benchmark-test"},
                    "version": "0.0.1",
                    "files": (Path("test_benchmark_adapters.py"),),
                    "locate_file": staticmethod(lambda item: Path(__file__).parent / item),
                },
            )()

            @staticmethod
            def load():
                return spec

        class EntryPoints:
            @staticmethod
            def select(*, group):
                self.assertEqual(group, "rethinkskill.benchmarks")
                return (EntryPoint(),)

        catalog = BenchmarkCatalog()
        self.assertEqual(catalog.names(), ())
        with patch(
            "rethinkskill.utils.plugins.metadata.entry_points",
            return_value=EntryPoints(),
        ):
            loaded = catalog.load_entry_points()
        self.assertEqual(loaded, ("plugin-example",))
        self.assertEqual(catalog.names(), ("plugin-example",))
        registration = catalog.manifest()["benchmarks"][0]["registration"]
        self.assertEqual(registration["source"], "entry_point")
        self.assertEqual(
            registration["entry_point_group"],
            "rethinkskill.benchmarks",
        )
        self.assertEqual(registration["entry_point_name"], "plugin-example")
        self.assertEqual(
            registration["distribution_name"],
            "rethinkskill-benchmark-test",
        )
        self.assertEqual(registration["distribution_version"], "0.0.1")
        self.assertNotIn("distribution_identity", registration)

    def test_builtin_registration_provenance_is_catalog_owned(self) -> None:
        manifest = benchmark_catalog().manifest()
        self.assertEqual(manifest["schema_version"], 4)
        self.assertTrue(manifest["sealed"])
        self.assertTrue(
            all(
                benchmark["registration"]["source"] == "builtin"
                and benchmark["registration"]["component_kind"] == "benchmark"
                for benchmark in manifest["benchmarks"]
            )
        )

        direct = BenchmarkSpec(
            name="direct",
            family="test",
            domain="test",
            mode=EvaluationMode.CASES,
            metrics=("pass_rate",),
            adapter=lambda row, base: Verification(Verdict.PASS, "ok"),
        )
        observed = BenchmarkCatalog((direct,)).manifest()
        self.assertEqual(
            observed["benchmarks"][0]["registration"]["source"],
            "direct",
        )

    def test_direct_registration_rejects_benchmark_spec_subclasses(self) -> None:
        class OverridingSpec(BenchmarkSpec):
            def validate(self):
                return None

        spec = OverridingSpec(
            name="subclass",
            family="test",
            domain="test",
            mode=EvaluationMode.CASES,
            metrics=("pass_rate",),
            adapter=lambda row, base: Verification(Verdict.PASS, "ok"),
        )
        with self.assertRaisesRegex(
            ConfigurationError,
            "requires an exact BenchmarkSpec",
        ):
            BenchmarkCatalog((spec,))

    def test_spec_rejects_benchmark_source_subclasses(self) -> None:
        class OverridingSource(BenchmarkSource):
            def validate(self):
                return None

        spec = BenchmarkSpec(
            name="source-subclass",
            family="test",
            domain="test",
            mode=EvaluationMode.CASES,
            metrics=("pass_rate",),
            adapter=lambda row, base: Verification(Verdict.PASS, "ok"),
            sources=(
                OverridingSource(
                    relation="",
                    title="",
                    url="",
                ),
            ),
        )
        with self.assertRaisesRegex(
            ConfigurationError,
            "exact BenchmarkSource values",
        ):
            spec.validate()

    def test_entry_point_batch_is_deterministic_and_atomic(self) -> None:
        valid = BenchmarkSpec(
            name="a-valid",
            family="test",
            domain="test",
            mode=EvaluationMode.CASES,
            metrics=("pass_rate",),
            adapter=lambda row, base: Verification(Verdict.PASS, "ok"),
        )
        invalid = BenchmarkSpec(
            name="z-invalid",
            family="test",
            domain="test",
            mode="cases",  # type: ignore[arg-type]
            metrics=("pass_rate",),
        )

        class EntryPoint:
            def __init__(self, name, value):
                self.name = name
                self.value = name
                self._value = value

            def load(self):
                return self._value

        class EntryPoints:
            @staticmethod
            def select(*, group):
                self.assertEqual(group, "rethinkskill.benchmarks")
                return (
                    EntryPoint("z-invalid", invalid),
                    EntryPoint("a-valid", valid),
                )

        catalog = BenchmarkCatalog()
        with (
            patch(
                "rethinkskill.utils.plugins.metadata.entry_points",
                return_value=EntryPoints(),
            ),
            self.assertRaisesRegex(ConfigurationError, "mode is invalid"),
        ):
            catalog.load_entry_points()
        self.assertEqual(catalog.names(), ())

    def test_entry_point_exception_is_redacted(self) -> None:
        secret = "plugin-secret-must-not-appear"

        class EntryPoint:
            name = "broken"
            value = "broken:SPEC"

            @staticmethod
            def load():
                raise RuntimeError(secret)

        class EntryPoints:
            @staticmethod
            def select(*, group):
                return (EntryPoint(),)

        with (
            patch(
                "rethinkskill.utils.plugins.metadata.entry_points",
                return_value=EntryPoints(),
            ),
            self.assertRaisesRegex(
                ConfigurationError,
                "loading failed: RuntimeError",
            ) as captured,
        ):
            BenchmarkCatalog().load_entry_points()
        self.assertNotIn(secret, str(captured.exception))

    def test_entry_point_rejects_benchmark_spec_subclasses(self) -> None:
        class OverridingSpec(BenchmarkSpec):
            def validate(self):
                return None

        spec = OverridingSpec(
            name="subclass",
            family="test",
            domain="test",
            mode=EvaluationMode.CASES,
            metrics=("pass_rate",),
            adapter=lambda row, base: Verification(Verdict.PASS, "ok"),
        )

        class EntryPoint:
            name = "subclass"
            value = "plugin:SPEC"

            @staticmethod
            def load():
                return spec

        class EntryPoints:
            @staticmethod
            def select(*, group):
                return (EntryPoint(),)

        catalog = BenchmarkCatalog()
        with (
            patch(
                "rethinkskill.utils.plugins.metadata.entry_points",
                return_value=EntryPoints(),
            ),
            self.assertRaisesRegex(
                ConfigurationError,
                "loading failed: TypeError",
            ),
        ):
            catalog.load_entry_points()
        self.assertEqual(catalog.names(), ())

    def test_external_harness_cannot_register_a_case_adapter(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError,
            "cannot masquerade",
        ):
            BenchmarkCatalog(
                (
                    BenchmarkSpec(
                        name="invalid",
                        family="test",
                        domain="test",
                        mode=EvaluationMode.EXTERNAL_HARNESS,
                        metrics=("score",),
                        adapter=lambda row, base: case_adapter("searchqa")(
                            row,
                            base,
                        ),
                    ),
                )
            )

    def test_livemath_rejects_malformed_choice_contract(self) -> None:
        with self.assertRaisesRegex(ResultValidationError, "choices must be"):
            case_adapter("livemath")(
                {
                    "case_id": "bad",
                    "frozen_response": "A",
                    "choices": "not-a-list",
                    "correct_choice": {"label": "A", "text": "x"},
                },
                Path.cwd(),
            )

    def test_original_case_adapters_replay_strict_json_contracts(self) -> None:
        cases = (
            (
                "officeqa",
                {
                    "frozen_response": "<answer>10</answer>",
                    "gold_answer": "10",
                },
            ),
            (
                "docvqa",
                {
                    "frozen_response": "<answer>Paris</answer>",
                    "gold_aliases": ["Paris"],
                },
            ),
            (
                "livemath",
                {
                    "frozen_response": "<answer>A</answer>",
                    "choices": [
                        {"label": "A", "text": "one"},
                        {"label": "B", "text": "two"},
                    ],
                    "correct_choice": {"label": "A", "text": "one"},
                },
            ),
        )
        for benchmark, row in cases:
            with self.subTest(benchmark=benchmark):
                permissive = case_adapter(benchmark)(
                    {"case_id": benchmark, **row},
                    Path.cwd(),
                )
                strict = case_adapter(benchmark)(
                    {"case_id": benchmark, **row, "strict_json": True},
                    Path.cwd(),
                )
                self.assertIs(permissive.verdict, Verdict.PASS)
                self.assertIs(strict.verdict, Verdict.FAIL)

    def test_spreadsheet_artifacts_cannot_escape_case_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            with self.assertRaisesRegex(ResultValidationError, "escapes"):
                case_adapter("spreadsheetbench")(
                    {
                        "case_id": "escape",
                        "prediction_artifact": {"path": "../prediction.xlsx"},
                        "reference_artifact": {"path": "reference.xlsx"},
                    },
                    base,
                )

    def test_optional_mce_classification_adapters_are_model_free(self) -> None:
        from rethinkskill_mce.scoring import (
            evaluate_aegis2,
            evaluate_label,
            evaluate_lawbench_charge,
        )

        finer = evaluate_label(
            {
                "case_id": "finance",
                "frozen_response": "<answer>LongTermDebt</answer>",
                "gold_label": "LongTermDebt",
            },
            Path.cwd(),
        )
        law = evaluate_lawbench_charge(
            {
                "case_id": "law",
                "frozen_response": "[罪名]盗窃罪;抢劫罪<eoa>",
                "gold_labels": ["盗窃罪", "诈骗罪"],
            },
            Path.cwd(),
        )
        aegis = evaluate_aegis2(
            {
                "case_id": "safety",
                "frozen_response": "unsafe",
                "gold_label": "unsafe",
            },
            Path.cwd(),
        )
        self.assertEqual(finer.verdict.value, "PASS")
        self.assertEqual(law.metrics["tp"], 1.0)
        self.assertEqual(law.metrics["fp"], 1.0)
        self.assertEqual(law.metrics["fn"], 1.0)
        self.assertEqual(aegis.verdict.value, "PASS")

    def test_native_harness_duplicates_fail_closed(self) -> None:
        spec = NativeHarnessSpec(
            name="example",
            harness=SearchQAHarness(),
            execution_scope="test",
            source="test",
        )
        with self.assertRaisesRegex(
            ConfigurationError,
            "duplicate native harness",
        ):
            HarnessCatalog((spec, spec))

    def test_native_harness_catalog_covers_file_backed_scorers(self) -> None:
        from rethinkskill.benchmarks.harness_catalog import native_harness_catalog

        manifest = native_harness_catalog().manifest()
        self.assertEqual(manifest["count"], 4)
        names = {harness["name"] for harness in manifest["harnesses"]}
        self.assertTrue({"searchqa", "officeqa", "docvqa", "livemath"}.issubset(names))
        self.assertNotIn("finer", names)
        self.assertNotIn("spreadsheetbench", names)
        self.assertNotIn("alfworld", names)
