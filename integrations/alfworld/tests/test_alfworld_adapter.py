from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from rethinkskill_alfworld.harness import AlfWorldHarness
from rethinkskill_alfworld.runtime import (
    AlfWorldState,
    extract_alfworld_action,
)

from rethinkskill.benchmarks.capabilities import CapabilityCatalog
from rethinkskill.benchmarks.core import benchmark_catalog
from rethinkskill.benchmarks.harness_catalog import (
    EvaluationAuthority,
    HarnessCatalog,
    NativeHarnessSpec,
)
from rethinkskill.evidence.run import validate_run_evidence
from rethinkskill.integrations.catalog import integration_catalog
from rethinkskill.runtime.runner import (
    NativeRunOptions,
    execute_native_plan,
    plan_native_run,
)
from rethinkskill.runtime.tasks import RenderedTask
from rethinkskill.runtime.types import ModelOutcome
from rethinkskill.utils.fs import Repository


class _Episode:
    def __init__(self):
        self.turn = 0
        self.closed = False

    def reset(self) -> AlfWorldState:
        return AlfWorldState(
            observation="You are in a kitchen. Your task is to place the apple.",
            admissible_actions=("go north", "look"),
            done=False,
            won=False,
            info={},
        )

    def step(self, action: str) -> AlfWorldState:
        self.turn += 1
        if self.turn == 1:
            if action != "go north":
                raise AssertionError(action)
            return AlfWorldState(
                observation="The apple is here.",
                admissible_actions=("take apple", "look"),
                done=False,
                won=False,
                info={"won": False},
            )
        if action != "take apple":
            raise AssertionError(action)
        return AlfWorldState(
            observation="You complete the task.",
            admissible_actions=(),
            done=True,
            won=True,
            info={"won": True},
        )

    def close(self) -> None:
        self.closed = True


class _Factory:
    def __init__(self):
        self.episodes: list[_Episode] = []
        self.opened_gamefiles: list[Path] = []

    def dependency_manifest(self) -> Mapping[str, object]:
        return {
            "kind": "fake-alfworld",
            "alfworld_available": True,
        }

    def open(
        self,
        *,
        gamefile: Path,
        data_root: Path,
        verifier_workspace: Path,
        seed: int,
        split: str,
    ) -> _Episode:
        del data_root, verifier_workspace, seed, split
        if not gamefile.is_file():
            raise AssertionError("materialized gamefile missing")
        self.opened_gamefiles.append(gamefile)
        episode = _Episode()
        self.episodes.append(episode)
        return episode


class _FailingFactory:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def dependency_manifest(self) -> Mapping[str, object]:
        return {
            "kind": "failing-alfworld",
            "alfworld_available": True,
        }

    def open(self, **kwargs):
        del kwargs
        raise RuntimeError(self.secret)


class _FailingStepEpisode(_Episode):
    def __init__(self, secret: str) -> None:
        super().__init__()
        self.secret = secret

    def step(self, action: str) -> AlfWorldState:
        del action
        raise RuntimeError(self.secret)


class _FailingStepFactory(_Factory):
    def __init__(self, secret: str) -> None:
        super().__init__()
        self.secret = secret

    def open(self, **kwargs) -> _FailingStepEpisode:
        del kwargs
        episode = _FailingStepEpisode(self.secret)
        self.episodes.append(episode)
        return episode


class _StepExecutor:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    def public_manifest(self) -> Mapping[str, object]:
        return {"kind": "fake-interactive", "model_calls": 0}

    def execute(
        self,
        rendered: RenderedTask,
        *,
        workspace: Path,
        timeout_seconds: int,
    ) -> ModelOutcome:
        del rendered, workspace, timeout_seconds
        response = self.responses[self.calls]
        self.calls += 1
        return ModelOutcome(
            status="COMPLETED",
            response=response,
            raw=response,
            process={"returncode": 0, "timed_out": False},
            attempted_calls=1,
            completed_calls=1,
        )


class AlfWorldNativeTests(unittest.TestCase):
    def test_state_detaches_info_and_rejects_coercible_terminal_values(
        self,
    ) -> None:
        info = {"won": False}
        state = AlfWorldState(
            observation="room",
            admissible_actions=("look",),
            done=False,
            won=False,
            info=info,
        )
        info["won"] = True
        self.assertFalse(state.info["won"])
        with self.assertRaises(TypeError):
            AlfWorldState(
                observation="room",
                admissible_actions=("look",),
                done=0,  # type: ignore[arg-type]
                won=False,
                info={},
            )

    def _repository(self, temporary: str) -> Repository:
        root = Path(temporary)
        (root / "pyproject.toml").write_text("", encoding="utf-8")
        return Repository.discover(root)

    def _plan(
        self,
        repository: Repository,
        factory: _Factory,
    ):
        corpus = repository.root / "alfworld-corpus"
        (corpus / "logic").mkdir(parents=True)
        (corpus / "logic/alfred.pddl").write_text(
            "(define (domain alfred))",
            encoding="utf-8",
        )
        (corpus / "logic/alfred.twl2").write_text(
            "grammar",
            encoding="utf-8",
        )
        game_directory = corpus / "json_2.1.1/valid_unseen/pick_and_place/task-1"
        game_directory.mkdir(parents=True)
        gamefile = game_directory / "game.tw-pddl"
        gamefile.write_text("synthetic game", encoding="utf-8")
        (game_directory / "traj_data.json").write_text(
            "{}",
            encoding="utf-8",
        )
        dataset = repository.root / "dataset/test"
        dataset.mkdir(parents=True)
        dataset.joinpath("items.json").write_text(
            json.dumps(
                [
                    {
                        "id": "alf-1",
                        "gamefile": str(gamefile.relative_to(corpus)),
                    }
                ]
            ),
            encoding="utf-8",
        )
        skill = repository.root / "skill.md"
        skill.write_text(
            "Track inventory and choose valid actions.",
            encoding="utf-8",
        )
        harness = AlfWorldHarness(
            environment_factory=factory,
            max_steps=3,
        )
        harnesses = HarnessCatalog(
            (
                NativeHarnessSpec(
                    name="alfworld",
                    harness=harness,
                    execution_scope="test interactive episode",
                    source="synthetic test",
                    evaluation_authority=EvaluationAuthority.HARNESS,
                ),
            )
        )
        catalog = CapabilityCatalog(
            benchmark_catalog(),
            harnesses,
            integration_catalog(),
        )
        return plan_native_run(
            repository=repository,
            catalog=catalog,
            benchmark="alfworld",
            dataset=dataset.parent,
            skill=skill,
            output_root=repository.root / "runs/alfworld",
            options=NativeRunOptions(
                split="test",
                limit=1,
                seed=42,
                asset_root=corpus,
            ),
        )

    def test_native_episode_uses_multiple_provider_calls_and_won_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = self._repository(temporary)
            factory = _Factory()
            plan = self._plan(repository, factory)
            executor = _StepExecutor(
                [
                    "<think>Move to the object.</think><action>go north</action>",
                    "<think>Pick it up.</think><action>take apple</action>",
                ]
            )
            receipt = execute_native_plan(
                plan,
                executor=executor,
                authorized=True,
            )
            replay = validate_run_evidence(plan.output_root)
            task_root = repository.root / "runs/alfworld/tasks/alf-1"
            row = json.loads(
                (repository.root / "runs/alfworld/results.jsonl").read_text(encoding="utf-8")
            )
            game_visible = (task_root / "workspace/environment/game.tw-pddl").exists()
            game_hidden = (task_root / "verifier/environment/game.tw-pddl").is_file()
        self.assertFalse(game_visible)
        self.assertTrue(game_hidden)
        self.assertEqual(executor.calls, 2)
        self.assertEqual(receipt["attempted_calls"], 2)
        self.assertEqual(receipt["completed_calls"], 2)
        self.assertEqual(
            replay["artifacts"]["interactive_step_evidence_rows"],
            2,
        )
        self.assertEqual(row["hard"], 1)
        self.assertEqual(row["soft"], 1.0)
        self.assertEqual(row["metrics"]["success"], 1.0)
        self.assertTrue(factory.episodes[0].closed)
        self.assertEqual(
            receipt["status"],
            "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED",
        )

    def test_completed_invalid_action_contract_is_behavior_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = self._repository(temporary)
            factory = _Factory()
            plan = self._plan(repository, factory)
            receipt = execute_native_plan(
                plan,
                executor=_StepExecutor(["<action>look</action>"]),
                authorized=True,
            )
            replay = validate_run_evidence(plan.output_root)
            row = json.loads(
                (repository.root / "runs/alfworld/results.jsonl").read_text(encoding="utf-8")
            )
        self.assertEqual(row["hard"], 0)
        self.assertEqual(row["failure_class"] if "failure_class" in row else None, None)
        self.assertIn(
            "output_contract_error",
            row["execution"]["interactive_runner"]["behavior_failure"],
        )
        self.assertEqual(
            receipt["status"],
            "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED",
        )
        self.assertEqual(
            replay["artifacts"]["interactive_step_evidence_rows"],
            1,
        )

    def test_environment_initialization_failure_redacts_exception_message(
        self,
    ) -> None:
        secret = "alfworld-environment-secret"
        with tempfile.TemporaryDirectory() as temporary:
            repository = self._repository(temporary)
            plan = self._plan(
                repository,
                _FailingFactory(secret),  # type: ignore[arg-type]
            )
            receipt = execute_native_plan(
                plan,
                executor=_StepExecutor(["<think>Inspect.</think><action>look</action>"]),
                authorized=True,
            )
            replay = validate_run_evidence(plan.output_root)
            row = json.loads(
                (repository.root / "runs/alfworld/results.jsonl").read_text(encoding="utf-8")
            )
        self.assertEqual(
            row["failure_class"],
            "environment_initialization_error",
        )
        self.assertIn("RuntimeError", row["fail_reason"])
        self.assertNotIn(secret, json.dumps(row))
        self.assertEqual(receipt["attempted_calls"], 0)
        self.assertEqual(
            replay["artifacts"]["interactive_step_evidence_rows"],
            0,
        )

    def test_environment_step_failure_redacts_exception_message(self) -> None:
        secret = "alfworld-step-secret"
        with tempfile.TemporaryDirectory() as temporary:
            repository = self._repository(temporary)
            plan = self._plan(repository, _FailingStepFactory(secret))
            receipt = execute_native_plan(
                plan,
                executor=_StepExecutor(["<think>Inspect.</think><action>look</action>"]),
                authorized=True,
            )
            replay = validate_run_evidence(plan.output_root)
            row = json.loads(
                (repository.root / "runs/alfworld/results.jsonl").read_text(encoding="utf-8")
            )
        self.assertEqual(row["failure_class"], "environment_error")
        self.assertIn("RuntimeError", row["fail_reason"])
        self.assertNotIn(secret, json.dumps(row))
        self.assertEqual(receipt["attempted_calls"], 1)
        self.assertEqual(receipt["completed_calls"], 1)
        self.assertEqual(
            replay["artifacts"]["interactive_step_evidence_rows"],
            1,
        )

    def test_action_parser_preserves_contract(self) -> None:
        self.assertEqual(
            extract_alfworld_action("<think>Inspect.</think><action>LOOK</action>"),
            "look",
        )
        with self.assertRaisesRegex(ValueError, "missing <think>"):
            extract_alfworld_action("<action>look</action>")
