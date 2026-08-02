from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from rethinkskill_alfworld.harness import AlfWorldHarness
from rethinkskill_alfworld.runtime import InstalledAlfWorldFactory

from rethinkskill.benchmarks.capabilities import CapabilityCatalog
from rethinkskill.benchmarks.core import benchmark_catalog
from rethinkskill.benchmarks.harness_catalog import (
    EvaluationAuthority,
    HarnessCatalog,
    NativeHarnessSpec,
)
from rethinkskill.integrations.catalog import integration_catalog
from rethinkskill.runtime.runner import (
    NativeRunOptions,
    execute_native_plan,
    plan_native_run,
)
from rethinkskill.runtime.tasks import RenderedTask
from rethinkskill.runtime.types import ModelOutcome
from rethinkskill.utils.fs import Repository

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
ALFWORLD_ASSET_ROOT = os.environ.get("RETHINKSKILL_ALFWORLD_ASSET_ROOT", "")


class _LookExecutor:
    calls = 0

    def public_manifest(self) -> Mapping[str, object]:
        return {
            "kind": "deterministic-installed-alfworld-smoke",
            "model_calls": 0,
        }

    def execute(
        self,
        rendered: RenderedTask,
        *,
        workspace: Path,
        timeout_seconds: int,
    ) -> ModelOutcome:
        del workspace, timeout_seconds
        self.calls += 1
        if "`look`" not in rendered.task_markdown:
            raise AssertionError("installed ALFWorld smoke requires `look`")
        response = "<think>Inspect the current room.</think><action>look</action>"
        return ModelOutcome(
            status="COMPLETED",
            response=response,
            raw=response,
            process={"returncode": 0, "timed_out": False},
            attempted_calls=1,
            completed_calls=1,
        )


@unittest.skipUnless(
    ALFWORLD_ASSET_ROOT,
    "set RETHINKSKILL_ALFWORLD_ASSET_ROOT for installed ALFWorld integration",
)
class InstalledAlfWorldIntegrationTests(unittest.TestCase):
    def test_real_environment_runs_through_materialized_verifier_workspace(
        self,
    ) -> None:
        asset_root = Path(ALFWORLD_ASSET_ROOT).expanduser().resolve()
        self.assertTrue((asset_root / "logic/alfred.pddl").is_file())
        self.assertTrue((asset_root / "logic/alfred.twl2").is_file())
        dataset = FIXTURE_ROOT / "dataset"
        skill = FIXTURE_ROOT / "skill.md"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text("", encoding="utf-8")
            repository = Repository.discover(root)
            harness = AlfWorldHarness(
                environment_factory=InstalledAlfWorldFactory(),
                max_steps=1,
            )
            harnesses = HarnessCatalog(
                (
                    NativeHarnessSpec(
                        name="alfworld",
                        harness=harness,
                        execution_scope="installed environment integration",
                        source="explicit local ALFWorld corpus",
                        evaluation_authority=EvaluationAuthority.HARNESS,
                    ),
                )
            )
            catalog = CapabilityCatalog(
                benchmark_catalog(),
                harnesses,
                integration_catalog(),
            )
            plan = plan_native_run(
                repository=repository,
                catalog=catalog,
                benchmark="alfworld",
                dataset=dataset,
                skill=skill,
                output_root=root / "runs/alfworld-installed",
                options=NativeRunOptions(
                    split="test",
                    limit=1,
                    seed=42,
                    asset_root=asset_root,
                ),
            )
            executor = _LookExecutor()
            receipt = execute_native_plan(
                plan,
                executor=executor,
                authorized=True,
            )
            self.assertEqual(
                receipt["status"],
                "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED",
                receipt,
            )
            self.assertEqual(receipt["attempted_calls"], 1)
            self.assertEqual(receipt["completed_calls"], 1)
            self.assertEqual(executor.calls, 1)
            row = json.loads(
                (root / "runs/alfworld-installed/results.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(row["hard"], 0)
            runner = row["execution"]["interactive_runner"]
            self.assertEqual(runner["transcript"][0]["action"], "look")
            task_root = root / "runs/alfworld-installed/tasks/alfworld-smoke-valid-unseen-1"
            self.assertFalse((task_root / "workspace/environment/game.tw-pddl").exists())
            self.assertTrue((task_root / "verifier/environment/game.tw-pddl").is_file())
