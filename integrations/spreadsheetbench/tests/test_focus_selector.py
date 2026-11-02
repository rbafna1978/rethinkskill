"""FOCUS fork: end-to-end tests for the cluster / contrastive selector hook.

Runs the real `native-evolve` CLI on tiny SpreadsheetBench-style workbooks with
fake target and optimizer models, then replays the run with `validate-run`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rethinkskill.cli import main
from rethinkskill.runtime.types import ModelOutcome

try:
    import openpyxl
except ImportError:  # pragma: no cover
    openpyxl = None

TARGET_CODE = (
    "from openpyxl import load_workbook\n"
    "wb = load_workbook(INPUT_PATH)\n"
    "s = wb['Sheet1']\n"
    "s['B1'] = s['A1'].value + s['A2'].value\n"
    "wb.save(OUTPUT_PATH)\n"
)

# task id -> (sheet title, A1 value, golden B1). Outcomes with TARGET_CODE:
TASKS = {
    "t-ok1": ("Sheet1", 1, 3),  # pass
    "t-val1": ("Sheet1", 1, 5),  # wrong value, no exception
    "t-key1": ("Data", 1, 3),  # KeyError: no Sheet1
    "t-key2": ("Data", 1, 3),  # KeyError: no Sheet1
    "t-type1": ("Sheet1", "x", 3),  # TypeError: str + int
    "t-ok2": ("Sheet1", 1, 3),  # validation
    "t-key3": ("Data", 1, 3),  # validation
}
TRAIN = ["t-ok1", "t-val1", "t-key1", "t-key2", "t-type1"]
VAL = ["t-ok2", "t-key3"]

# Test selector: group failures by exception text, pick the largest unused group.
SELECTOR = textwrap.dedent(
    """
    import json, sys
    data = json.load(sys.stdin)
    if "--fail" in sys.argv:
        print("boom", file=sys.stderr); sys.exit(3)
    if "--junk" in sys.argv:
        print("not json"); sys.exit(0)
    if "--unknown" in sys.argv:
        print(json.dumps({"keep_case_ids": ["nope"]})); sys.exit(0)
    if "--empty" in sys.argv:
        print(json.dumps({"keep_case_ids": []})); sys.exit(0)
    rows = {}
    for line in open(data["results_path"], encoding="utf-8"):
        row = json.loads(line); rows[row["case_id"]] = row
    groups = {}
    for item in data["feedback"]:
        if item["category"] != "failure":
            continue
        cases = rows[item["case_id"]]["execution"]["artifact_runner"]["cases"]
        key = (cases[0].get("exception") or item["reason"]).split(":")[0]
        groups.setdefault(key, []).append(item["case_id"])
    used = {h["cluster_id"] for h in data["history"]}
    order = sorted(groups, key=lambda k: (-len(groups[k]), k))
    pick = next((k for k in order if k not in used), order[0])
    keep = list(groups[pick])
    if data["mode"] == "contrastive":
        keep += [i["case_id"] for i in data["feedback"] if i["category"] == "success"][:1]
    print("picked", pick, file=sys.stderr)
    print(json.dumps({"contract_version": 1, "keep_case_ids": keep, "cluster_id": pick,
                      "notes": "test", "clusters": groups}))
    """
)


class _TargetExecutor:
    def public_manifest(self):
        return {"kind": "fake-codegen", "model_calls": 0}

    def execute(self, rendered, *, workspace, timeout_seconds):
        del rendered, workspace, timeout_seconds
        return ModelOutcome(
            status="COMPLETED",
            response=f"```python\n{TARGET_CODE}```",
            raw="synthetic",
            process={"returncode": 0},
            attempted_calls=1,
            completed_calls=1,
        )


class _OptimizerExecutor:
    def __init__(self):
        self.prompts: list[str] = []

    def public_manifest(self):
        return {"kind": "fake-optimizer", "model_calls": 0}

    def execute(self, rendered, *, workspace, timeout_seconds):
        del workspace, timeout_seconds
        self.prompts.append(rendered.task_markdown)
        skill = f"# Skill\n\nUse openpyxl carefully. Version {len(self.prompts)}.\n"
        body = {"operation": "replace", "candidate_skill": skill, "rationale": "test"}
        return ModelOutcome(
            status="COMPLETED",
            response=json.dumps(body),
            raw="synthetic",
            process={"returncode": 0},
            attempted_calls=1,
            completed_calls=1,
        )


@unittest.skipIf(openpyxl is None, "openpyxl is required")
class FocusSelectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "pyproject.toml").write_text("", encoding="utf-8")
        corpus = self.root / "corpus"
        records = []
        for task_id, (title, a1, golden) in TASKS.items():
            folder = corpus / "spreadsheet" / task_id
            folder.mkdir(parents=True)
            for kind, b1 in (("init", None), ("golden", golden)):
                wb = openpyxl.Workbook()
                ws = wb.active
                ws.title = title
                ws["A1"], ws["A2"] = a1, 2
                if b1 is not None:
                    ws["B1"] = b1
                wb.save(folder / f"1_{task_id}_{kind}.xlsx")
            records.append(
                {
                    "id": task_id,
                    "instruction": "Put A1+A2 in B1.",
                    "instruction_type": "Cell-Level Manipulation",
                    "answer_position": f"{title}!B1",
                    "spreadsheet_path": f"spreadsheet/{task_id}",
                }
            )
        self.corpus = corpus
        self.dataset = self.root / "dataset.json"
        self.dataset.write_text(json.dumps(records), encoding="utf-8")
        (self.root / "train.txt").write_text("\n".join(TRAIN) + "\n", encoding="utf-8")
        (self.root / "val.txt").write_text("\n".join(VAL) + "\n", encoding="utf-8")
        (self.root / "skill.md").write_text("# Skill\n\nUse openpyxl.\n", encoding="utf-8")
        self.selector = self.root / "selector.py"
        self.selector.write_text(SELECTOR, encoding="utf-8")
        self.optimizer = _OptimizerExecutor()
        cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)
        env = patch.dict(os.environ, {"FOCUS_FAKE_KEY": "fake-key-value-123"})
        env.start()
        self.addCleanup(env.stop)

    def _prepare(self, config, *, authorized, catalog=None, role=None):
        del config, authorized, catalog
        return _TargetExecutor() if role == "target" else self.optimizer

    def cli(self, *arguments):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(arguments))
        text = out.getvalue().strip() or err.getvalue().strip()
        return code, json.loads(text)

    def evolve(self, arm, *, run="run", selector_args="", rounds=2, extra=()):
        argv = [
            "--repository",
            str(self.root),
            "--load-benchmark-plugins",
            "--load-harness-plugins",
            "native-evolve",
            "--benchmark",
            "spreadsheetbench",
            "--dataset",
            str(self.dataset),
            "--asset-root",
            str(self.corpus),
            "--skill",
            str(self.root / "skill.md"),
            "--out-root",
            str(self.root / "runs" / run),
            "--train-ids-file",
            str(self.root / "train.txt"),
            "--validation-ids-file",
            str(self.root / "val.txt"),
            "--rounds",
            str(rounds),
            "--arm",
            arm,
            *extra,
        ]
        for role in ("target", "optimizer"):
            argv += [
                f"--{role}-transport",
                "openai-compatible",
                f"--{role}-model",
                "fake",
                f"--{role}-api-base-url",
                "https://fake.invalid/v1",
                f"--{role}-api-key-env",
                "FOCUS_FAKE_KEY",
            ]
        if selector_args is not None and arm in {"cluster", "contrastive"}:
            argv += ["--selector-command", f"{sys.executable} {self.selector} {selector_args}"]
        argv.append("--authorize-model-calls")
        with patch("rethinkskill.cli.commands.prepare_executor", side_effect=self._prepare):
            return self.cli(*argv)

    def selection(self, run, round_no):
        path = self.root / "runs" / run / "selection" / f"round_{round_no:04d}" / "SELECTION.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_cluster_arm_shows_only_selected_failures_and_replays(self):
        code, receipt = self.evolve("cluster")
        self.assertEqual(code, 0, receipt)
        self.assertEqual(receipt["status"], "RETHINKSKILL_EVOLUTION_VALIDATED", receipt)
        first = self.selection("run", 1)
        self.assertEqual(first["kept"], ["t-key1", "t-key2"])
        self.assertEqual(first["output"]["cluster_id"], "KeyError")
        self.assertEqual(first["input"]["history"], [])
        self.assertEqual(first["input"]["mode"], "cluster")
        self.assertEqual(
            sorted(item["case_id"] for item in first["input"]["feedback"]),
            ["t-key1", "t-key2", "t-type1", "t-val1"],
        )
        prompt = self.optimizer.prompts[0]
        for kept in ("t-key1", "t-key2"):
            self.assertIn(kept, prompt)
        for dropped in ("t-type1", "t-val1", "t-ok1"):
            self.assertNotIn(dropped, prompt)
        second = self.selection("run", 2)
        self.assertEqual(second["input"]["history"][0]["cluster_id"], "KeyError")
        self.assertNotEqual(second["output"]["cluster_id"], "KeyError")
        self.assertNotIn("fake-key-value-123", json.dumps(second))
        code, report = self.cli("validate-run", str(self.root / "runs" / "run"))
        self.assertEqual(code, 0, report)

    def test_contrastive_arm_adds_a_success(self):
        code, receipt = self.evolve("contrastive", rounds=1)
        self.assertEqual(code, 0, receipt)
        self.assertEqual(self.selection("run", 1)["kept"], ["t-ok1", "t-key1", "t-key2"])
        self.assertIn("t-ok1", self.optimizer.prompts[0])
        self.assertNotIn("t-type1", self.optimizer.prompts[0])
        code, report = self.cli("validate-run", str(self.root / "runs" / "run"))
        self.assertEqual(code, 0, report)

    def test_fail_only_is_unchanged_and_has_no_selection(self):
        code, receipt = self.evolve("fail_only", rounds=1)
        self.assertEqual(code, 0, receipt)
        self.assertFalse((self.root / "runs" / "run" / "selection").exists())
        for case_id in ("t-key1", "t-key2", "t-type1", "t-val1"):
            self.assertIn(case_id, self.optimizer.prompts[0])
        code, report = self.cli("validate-run", str(self.root / "runs" / "run"))
        self.assertEqual(code, 0, report)

    def test_tampered_selection_fails_validation(self):
        code, receipt = self.evolve("cluster", rounds=1)
        self.assertEqual(code, 0, receipt)
        path = self.root / "runs/run/selection/round_0001/SELECTION.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["kept"] = ["t-type1"]
        path.write_text(json.dumps(record), encoding="utf-8")
        code, _ = self.cli("validate-run", str(self.root / "runs" / "run"))
        self.assertNotEqual(code, 0)

    def test_broken_selectors_stop_the_run(self):
        for flag in ("--fail", "--junk", "--unknown", "--empty"):
            with self.subTest(flag=flag):
                run = f"bad{flag}"
                code, receipt = self.evolve("cluster", run=run, selector_args=flag, rounds=1)
                status = receipt.get("status")
                self.assertNotEqual(status, "RETHINKSKILL_EVOLUTION_VALIDATED", receipt)
                self.assertEqual(receipt.get("failure_class"), "selector_error", receipt)
                self.assertIn("error", self.selection(run, 1))
                self.assertEqual(self.optimizer.prompts, [])

    def test_round_without_failures_skips_selector(self):
        (self.root / "train.txt").write_text("t-ok1\n", encoding="utf-8")
        code, receipt = self.evolve("cluster", rounds=1)
        self.assertEqual(code, 0, receipt)
        self.assertEqual(receipt["status"], "RETHINKSKILL_EVOLUTION_VALIDATED", receipt)
        record = self.selection("run", 1)
        self.assertIn("skipped", record)
        self.assertEqual(record["kept"], [])
        code, report = self.cli("validate-run", str(self.root / "runs" / "run"))
        self.assertEqual(code, 0, report)

    def test_cli_rejects_missing_or_misplaced_selector(self):
        code, error = self.evolve("cluster", selector_args=None, rounds=1)
        self.assertNotEqual(code, 0)
        self.assertIn("requires --selector-command", json.dumps(error))
        code, error = self.evolve("normal", rounds=1, extra=("--selector-command", "python x.py"))
        self.assertNotEqual(code, 0)
        self.assertIn("only valid with --arm cluster", json.dumps(error))


if __name__ == "__main__":
    unittest.main()
