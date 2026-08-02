from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rethinkskill_skillrl.harness import SkillRLOfflineHarness
from rethinkskill_skillrl.plugin import benchmark_specs, harness_specs

from rethinkskill.benchmarks.scoring import Verdict


class SkillRLOfflineAdapterTests(unittest.TestCase):
    def test_all_seven_scorers_are_deterministic(self) -> None:
        specs = benchmark_specs()
        self.assertEqual(len(specs), 7)
        self.assertEqual(
            tuple(spec.name for spec in specs),
            (
                "nq",
                "triviaqa",
                "popqa",
                "hotpotqa",
                "2wiki",
                "musique",
                "bamboogle",
            ),
        )
        for spec in specs:
            spec.validate()
            verdict = spec.adapter(
                {
                    "case_id": "fixture",
                    "frozen_response": "<answer>Paris</answer>",
                    "gold_aliases": ["Paris", "City of Paris"],
                },
                Path("."),
            )
            self.assertIs(verdict.verdict, Verdict.PASS)

    def test_harness_load_render_and_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "dataset.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "id": "qa-1",
                            "question": "Capital of France?",
                            "context": "Paris is the capital of France.",
                            "gold_aliases": ["Paris"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            harness = SkillRLOfflineHarness()
            tasks = harness.load_tasks(
                source,
                split="test",
                limit=None,
                requested_ids=(),
                seed=0,
                asset_root=None,
            )
            self.assertEqual(tuple(task.task_id for task in tasks), ("qa-1",))
            self.assertIn("Capital of France?", harness.render(tasks[0], "qa").task_markdown)
            self.assertIs(
                harness.evaluate(tasks[0], "<answer>Paris</answer>").verdict,
                Verdict.PASS,
            )

    def test_all_harness_specs_share_the_offline_scope(self) -> None:
        specs = harness_specs()
        self.assertEqual(len(specs), 7)
        for spec in specs:
            spec.validate()
            self.assertIn("not equivalent", spec.execution_scope)
            self.assertIsInstance(spec.harness, SkillRLOfflineHarness)


if __name__ == "__main__":
    unittest.main()
