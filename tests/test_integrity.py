from __future__ import annotations

import unittest
from collections.abc import Iterator, Mapping

from rethinkskill.errors import ConfigurationError
from rethinkskill.utils.integrity import (
    freeze_manifest,
    validate_frozen_plan_fields,
    validate_plan_manifest,
    validate_zero_call_preflight,
)


class PlanIntegrityTests(unittest.TestCase):
    def test_validated_manifest_reads_the_caller_mapping_once(self) -> None:
        class ReadOnceMapping(Mapping[str, object]):
            def __init__(self, source: Mapping[str, object]) -> None:
                self.source = dict(source)
                self.items_calls = 0

            def __getitem__(self, key: str) -> object:
                return self.source[key]

            def __iter__(self) -> Iterator[str]:
                return iter(self.source)

            def __len__(self) -> int:
                return len(self.source)

            def items(self):  # type: ignore[no-untyped-def]
                self.items_calls += 1
                if self.items_calls > 1:
                    raise RuntimeError("caller mapping was read twice")
                return self.source.items()

        manifest = freeze_manifest({"status": "PASS", "model_calls": 0})
        source = ReadOnceMapping(manifest)
        snapshot = validate_plan_manifest(source, label="test preflight")
        self.assertEqual(snapshot["status"], "PASS")
        self.assertEqual(source.items_calls, 1)

    def test_validated_manifest_is_a_detached_snapshot(self) -> None:
        source = {
            "status": "PASS",
            "model_calls": 0,
            "nested": {"items": ["one"]},
        }
        manifest = freeze_manifest(source)
        snapshot = validate_plan_manifest(manifest, label="test preflight")
        snapshot["nested"]["items"].append("two")  # type: ignore[index,union-attr]
        self.assertEqual(manifest["nested"]["items"], ("one",))  # type: ignore[index]

    def test_frozen_field_validation_reports_the_exact_field(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError,
            "^native frozen plan drifted: selection$",
        ):
            validate_frozen_plan_fields(
                {"benchmark": "demo", "selection": {"count": 1}},
                {"benchmark": "demo", "selection": {"count": 2}},
                label="native",
            )

    def test_zero_call_preflight_requires_exact_integer_zeros(self) -> None:
        for invalid in (False, 0.0, "0", None):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(
                    ConfigurationError,
                    "^native preflight status or call accounting is invalid$",
                ),
            ):
                validate_zero_call_preflight(
                    {
                        "status": "RETHINKSKILL_NATIVE_PREFLIGHT_PASS",
                        "model_calls": invalid,
                    },
                    label="native preflight",
                    expected_status="RETHINKSKILL_NATIVE_PREFLIGHT_PASS",
                    call_fields=("model_calls",),
                )
        validate_zero_call_preflight(
            {
                "status": "RETHINKSKILL_NATIVE_PREFLIGHT_PASS",
                "model_calls": 0,
            },
            label="native preflight",
            expected_status="RETHINKSKILL_NATIVE_PREFLIGHT_PASS",
            call_fields=("model_calls",),
        )

    def test_zero_call_preflight_rejects_status_and_field_contract_drift(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ConfigurationError,
            "status or call accounting is invalid",
        ):
            validate_zero_call_preflight(
                {"status": "WRONG", "model_calls": 0},
                label="frozen preflight",
                expected_status="RETHINKSKILL_STATIC_PREFLIGHT_PASS",
                call_fields=("model_calls",),
            )
        for call_fields in ((), ("model_calls", "model_calls"), ["x"]):
            with (
                self.subTest(call_fields=call_fields),
                self.assertRaisesRegex(
                    ConfigurationError,
                    "^preflight call fields must be unique exact names$",
                ),
            ):
                validate_zero_call_preflight(
                    {"status": "PASS", "model_calls": 0},
                    label="test preflight",
                    expected_status="PASS",
                    call_fields=call_fields,  # type: ignore[arg-type]
                )


if __name__ == "__main__":
    unittest.main()
