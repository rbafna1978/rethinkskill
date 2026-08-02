from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rethinkskill_mce.plugin import BENCHMARKS, benchmark_specs, harness_specs

from rethinkskill.benchmarks.scoring import Verdict


class MCEAdapterTests(unittest.TestCase):
    def test_all_five_scorers_are_deterministic(self) -> None:
        specs = {spec.name: spec for spec in benchmark_specs()}
        self.assertEqual(tuple(specs), BENCHMARKS)
        rows = {
            "finer": {
                "frozen_response": "<answer>LongTermDebt</answer>",
                "gold_label": "LongTermDebt",
            },
            "uspto50k": {
                "frozen_response": "C.O",
                "gold_reactants": "C.O",
            },
            "symptom2disease": {
                "frozen_response": "influenza",
                "gold_label": "Influenza",
            },
            "lawbench-charge": {
                "frozen_response": "[罪名]盗窃罪;诈骗罪<eoa>",
                "gold_labels": ["盗窃罪", "诈骗罪"],
            },
            "aegis2": {
                "frozen_response": "unsafe",
                "gold_label": "unsafe",
            },
        }
        for name, spec in specs.items():
            with self.subTest(name=name):
                spec.validate()
                verdict = spec.adapter(rows[name], Path("."))
                self.assertIs(verdict.verdict, Verdict.PASS)

    def test_all_harnesses_load_render_and_evaluate(self) -> None:
        specs = {spec.name: spec for spec in harness_specs()}
        self.assertEqual(tuple(specs), BENCHMARKS)
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "dataset.json"
            source.write_text(
                json.dumps(
                    [
                        {
                            "id": "finer-1",
                            "text": "Classify this financial sentence.",
                            "gold_label": "LongTermDebt",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            harness = specs["finer"].harness
            tasks = harness.load_tasks(
                source,
                split="test",
                limit=None,
                requested_ids=(),
                seed=0,
                asset_root=None,
            )
            self.assertEqual(tuple(task.task_id for task in tasks), ("finer-1",))
            self.assertIn("financial sentence", harness.render(tasks[0], "skill").task_markdown)
            self.assertIs(
                harness.evaluate(tasks[0], "<answer>LongTermDebt</answer>").verdict,
                Verdict.PASS,
            )


if __name__ == "__main__":
    unittest.main()
