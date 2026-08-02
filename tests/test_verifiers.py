from __future__ import annotations

import unittest

from rethinkskill.benchmarks.scoring import (
    Verdict,
    extract_strict_json,
    verify_alfworld,
    verify_docvqa,
    verify_livemath,
    verify_officeqa,
    verify_searchqa,
)


class VerifierTests(unittest.TestCase):
    def test_searchqa_matches_frozen_squad_semantics(self) -> None:
        value = verify_searchqa("reasoning\n<answer>The Paris.</answer>", ["Paris"])
        self.assertEqual(value.verdict, Verdict.PASS)
        self.assertEqual(value.metrics["exact_match"], 1.0)

    def test_searchqa_retains_soft_metrics_on_hard_failure(self) -> None:
        value = verify_searchqa("<answer>Paris France</answer>", ["Paris"])
        self.assertEqual(value.verdict, Verdict.FAIL)
        self.assertEqual(value.metrics["substring"], 1.0)
        self.assertGreater(value.metrics["f1"], 0)

    def test_officeqa_numeric_normalization(self) -> None:
        value = verify_officeqa("<answer>1,234.50 dollars</answer>", "1234.50")
        self.assertEqual(value.verdict, Verdict.PASS)
        decimal_variant = verify_officeqa("<answer>113,864.0</answer>", "113864")
        self.assertEqual(decimal_variant.verdict, Verdict.FAIL)

    def test_docvqa_anls_and_hard_threshold(self) -> None:
        value = verify_docvqa("<answer>Washington, D.C.</answer>", ["Washington, D.C."])
        self.assertEqual(value.verdict, Verdict.PASS)
        soft = verify_docvqa(
            "<answer>competitor's joined the price war</answer>",
            ["As competitor's joined the price war"],
        )
        self.assertEqual(soft.verdict, Verdict.FAIL)
        self.assertGreater(soft.metrics["anls"], 0.5)
        tuple_alias = verify_docvqa(
            "<answer>$42.00</answer>",
            ("$42.00", "42 dollars"),
        )
        self.assertEqual(tuple_alias.verdict, Verdict.PASS)
        self.assertEqual(tuple_alias.metrics["anls"], 1.0)

    def test_livemath_label_and_text(self) -> None:
        choices = [{"label": "A", "text": "alpha"}, {"label": "B", "text": "beta"}]
        self.assertEqual(
            verify_livemath("A", choices, choices[0]).verdict,
            Verdict.PASS,
        )
        self.assertEqual(
            verify_livemath("beta", choices, choices[0]).verdict,
            Verdict.FAIL,
        )
        self.assertEqual(
            verify_livemath("A because it follows", choices, choices[0]).verdict,
            Verdict.PASS,
        )

    def test_strict_json_and_alfworld_abstention(self) -> None:
        self.assertEqual(extract_strict_json('{"answer":"A"}'), ("A", None))
        self.assertEqual(extract_strict_json('{"answer":"A","extra":1}')[1], "schema_mismatch")
        self.assertEqual(
            extract_strict_json('{"answer":NaN}')[1],
            "invalid_json",
        )
        self.assertEqual(verify_alfworld().verdict, Verdict.ABSTAIN)
