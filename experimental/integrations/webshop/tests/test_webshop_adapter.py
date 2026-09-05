from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock, patch

import rethinkskill_webshop.runtime as runtime
from rethinkskill_webshop.bridge import _error, _state, _strict_json_loads
from rethinkskill_webshop.harness import WebShopHarness, extract_webshop_action
from rethinkskill_webshop.plugin import spec
from rethinkskill_webshop.runtime import SubprocessWebShopFactory, WebShopState, _BridgeEpisode

from rethinkskill.benchmarks.harness_catalog import EvaluationAuthority
from rethinkskill.benchmarks.scoring import Verdict
from rethinkskill.errors import ConfigurationError
from rethinkskill.runtime.types import ModelOutcome


class ReadyFactory:
    def dependency_manifest(self) -> Mapping[str, object]:
        return {"kind": "test", "ready": True}

    def open(self, **_kwargs):  # pragma: no cover - execution is tested in core
        raise AssertionError("not used")


class FakeEpisode:
    def __init__(self, *, step_error: str | None = None, close_error: bool = False) -> None:
        self.step_error = step_error
        self.close_error = close_error
        self.closed = False

    def reset(self) -> WebShopState:
        return WebShopState(
            observation="Instruction: buy a red cup",
            available_actions={"has_search_bar": True, "clickables": ["search"]},
            reward=0.0,
            done=False,
            info={},
        )

    def step(self, action: str) -> WebShopState:
        if self.step_error is not None:
            raise RuntimeError(self.step_error)
        if action != "search[red cup]":
            raise AssertionError(action)
        return WebShopState(
            observation="Purchase complete",
            available_actions={"has_search_bar": False, "clickables": []},
            reward=1.0,
            done=True,
            info={},
        )

    def close(self) -> None:
        self.closed = True
        if self.close_error:
            raise RuntimeError("synthetic close failure")


class FakeFactory:
    def __init__(
        self,
        *,
        open_error: str | None = None,
        step_error: str | None = None,
        close_error: bool = False,
    ) -> None:
        self.open_error = open_error
        self.episode = FakeEpisode(step_error=step_error, close_error=close_error)

    def dependency_manifest(self) -> Mapping[str, object]:
        return {"kind": "fake_webshop", "ready": True}

    def open(self, **kwargs):
        self.kwargs = kwargs
        if self.open_error is not None:
            raise RuntimeError(self.open_error)
        return self.episode


class FakeExecutor:
    def public_manifest(self) -> Mapping[str, object]:
        return {"kind": "fake"}

    def execute(self, rendered, *, workspace, timeout_seconds):
        del rendered, workspace, timeout_seconds
        return ModelOutcome(
            status="COMPLETED",
            response="<think>Search first.</think><action>search[red cup]</action>",
            raw="fake",
            process={"kind": "fake"},
            attempted_calls=1,
            completed_calls=1,
        )


class WebShopAdapterTests(unittest.TestCase):
    @staticmethod
    def _task(harness: WebShopHarness, root: Path):
        dataset = root / "items.json"
        dataset.write_text(json.dumps([{"id": "shop-0", "session": 0}]), encoding="utf-8")
        return harness.load_tasks(
            dataset,
            split="test",
            limit=1,
            requested_ids=(),
            seed=42,
            asset_root=None,
        )[0]

    @staticmethod
    def _official_runtime_paths(root: Path) -> tuple[Path, Path, Path]:
        checkout = root / "webshop"
        (checkout / "web_agent_site/envs").mkdir(parents=True)
        (checkout / "web_agent_site/envs/web_agent_text_env.py").write_text(
            "# fixture", encoding="utf-8"
        )
        (checkout / "data").mkdir()
        (checkout / "search_engine").mkdir()
        python = root / "python3.8"
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o755)
        java_home = root / "jdk"
        (java_home / "bin").mkdir(parents=True)
        java = java_home / "bin/java"
        java.write_text("#!/bin/sh\n", encoding="utf-8")
        java.chmod(0o755)
        return checkout, python, java_home

    def test_bridge_normalizes_none_info(self) -> None:
        environment = MagicMock()
        environment.get_available_actions.return_value = {
            "has_search_bar": True,
            "clickables": ["search"],
        }
        value = _state(environment, "Instruction", 0.0, False, None)
        self.assertEqual(value["info"], {})

    def test_bridge_reader_rejects_unstructured_error_without_echoing_it(self) -> None:
        secret = "bridge-error-secret"
        process = MagicMock()
        process.stdout = io.StringIO(json.dumps({"status": "error", "error": secret}) + "\n")
        episode = _BridgeEpisode(
            process=process,
            session=0,
            timeout_seconds=1,
            log_path=Path("bridge.log"),
            log_stream=io.StringIO(),
        )
        with (
            patch("rethinkskill_webshop.runtime.selectors.DefaultSelector") as selector,
            self.assertRaises(ConfigurationError) as caught,
        ):
            selector.return_value.select.return_value = [object()]
            episode._read()
        self.assertEqual(str(caught.exception), "WebShop bridge returned invalid error evidence")
        self.assertNotIn(secret, str(caught.exception))

    def test_bridge_error_evidence_redacts_exception_message(self) -> None:
        secret = "bridge-secret-must-not-persist"
        value = _error("request_failed", RuntimeError(secret))
        self.assertEqual(
            value,
            {
                "status": "error",
                "error_code": "request_failed",
                "error_type": "RuntimeError",
            },
        )
        self.assertNotIn(secret, json.dumps(value))

    def test_bridge_rejects_duplicate_json_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            _strict_json_loads('{"op": "reset", "op": "close"}')

    def test_action_contract_accepts_search_and_click(self) -> None:
        self.assertEqual(
            extract_webshop_action("<think>query</think><action>search[red mug]</action>"),
            "search[red mug]",
        )
        self.assertEqual(
            extract_webshop_action("<think>choose</think><action>click[Buy Now]</action>"),
            "click[Buy Now]",
        )
        for invalid in (
            "search[x]",
            "<action>search[x]</action>",
            "<think>x</think><action>buy[x]</action>",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                extract_webshop_action(invalid)

    def test_plugin_declares_harness_owned_evaluation(self) -> None:
        plugin = spec()
        self.assertEqual(plugin.name, "webshop")
        self.assertEqual(plugin.evaluation_authority, EvaluationAuthority.HARNESS)

    def test_dependency_manifest_is_a_lightweight_readiness_check(self) -> None:
        manifest = SubprocessWebShopFactory().dependency_manifest()
        self.assertFalse(manifest["ready"])
        self.assertFalse(manifest["python_configured"])
        self.assertFalse(manifest["java_home_configured"])
        self.assertFalse(manifest["checkout_configured"])
        self.assertNotIn("environment_source_sha256", manifest)
        self.assertNotIn("python_distribution_inventory", manifest)

    def test_dependency_manifest_requires_explicit_java_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, _ = self._official_runtime_paths(root)
            manifest = SubprocessWebShopFactory(root=checkout, python=python).dependency_manifest()
        self.assertFalse(manifest["ready"])
        self.assertFalse(manifest["java_home_configured"])
        self.assertFalse(manifest["java_ready"])

    def test_load_tasks_uses_explicit_sessions_and_ids(self) -> None:
        harness = WebShopHarness(environment_factory=ReadyFactory())
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.jsonl"
            source.write_text(
                json.dumps({"id": "item-7", "session": 7, "max_steps": 4}) + "\n",
                encoding="utf-8",
            )
            tasks = harness.load_tasks(
                source,
                split="test",
                limit=None,
                requested_ids=(),
                seed=0,
                asset_root=None,
            )
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "item-7")
        self.assertEqual(tasks[0].payload["session"], 7)
        self.assertEqual(tasks[0].payload["max_steps"], 4)

    def test_load_tasks_fails_when_external_runtime_is_not_ready(self) -> None:
        harness = WebShopHarness(environment_factory=SubprocessWebShopFactory())
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "test.jsonl"
            source.write_text(json.dumps({"session": 0}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "not ready"):
                harness.load_tasks(
                    source,
                    split="test",
                    limit=None,
                    requested_ids=(),
                    seed=0,
                    asset_root=None,
                )

    def test_state_detaches_environment_mappings(self) -> None:
        actions = {"clickables": ["Buy Now"], "has_search_bar": False}
        info = {"session": 1}
        state = WebShopState(
            observation="product",
            available_actions=actions,
            reward=1.0,
            done=True,
            info=info,
        )
        actions["clickables"].append("Back")
        info["session"] = 2
        self.assertEqual(state.available_actions["clickables"], ("Buy Now",))
        self.assertEqual(state.info["session"], 1)
        for reward in (True, float("nan"), "1"):
            with self.subTest(reward=reward), self.assertRaises(TypeError):
                WebShopState(
                    observation="product",
                    available_actions={},
                    reward=reward,  # type: ignore[arg-type]
                    done=False,
                    info={},
                )

    def test_fake_episode_executes_and_closes(self) -> None:
        factory = FakeFactory()
        harness = WebShopHarness(environment_factory=factory)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(harness, root)
            outcome = harness.execute_task(
                task,
                harness.render(task, "Use exact click labels."),
                FakeExecutor(),
                workspace=root / "workspace",
                verifier_workspace=root / "verifier",
                timeout_seconds=30,
            )
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.attempted_calls, 1)
        self.assertTrue(factory.episode.closed)
        self.assertIs(harness.evaluate(task, outcome.response).verdict, Verdict.PASS)

    def test_environment_initialization_failure_redacts_message(self) -> None:
        secret = "webshop-environment-secret"
        harness = WebShopHarness(environment_factory=FakeFactory(open_error=secret))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(harness, root)
            outcome = harness.execute_task(
                task,
                harness.render(task, "Use exact labels."),
                FakeExecutor(),
                workspace=root / "workspace",
                verifier_workspace=root / "verifier",
                timeout_seconds=30,
            )
        self.assertEqual(outcome.failure_class, "environment_initialization_error")
        self.assertIn("RuntimeError", outcome.failure)
        self.assertNotIn(secret, outcome.failure)

    def test_environment_step_failure_redacts_message(self) -> None:
        secret = "webshop-step-secret"
        harness = WebShopHarness(environment_factory=FakeFactory(step_error=secret))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(harness, root)
            outcome = harness.execute_task(
                task,
                harness.render(task, "Use exact labels."),
                FakeExecutor(),
                workspace=root / "workspace",
                verifier_workspace=root / "verifier",
                timeout_seconds=30,
            )
        self.assertEqual(outcome.failure_class, "environment_error")
        self.assertIn("RuntimeError", outcome.failure)
        self.assertNotIn(secret, json.dumps(outcome.public()))

    def test_close_failure_does_not_override_terminal_reward(self) -> None:
        factory = FakeFactory(close_error=True)
        harness = WebShopHarness(environment_factory=factory)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self._task(harness, root)
            outcome = harness.execute_task(
                task,
                harness.render(task, "Use exact labels."),
                FakeExecutor(),
                workspace=root / "workspace",
                verifier_workspace=root / "verifier",
                timeout_seconds=30,
            )
        self.assertTrue(factory.episode.closed)
        self.assertTrue(outcome.ok)
        self.assertEqual(json.loads(outcome.response)["reward"], 1.0)

    def test_bridge_receives_only_allowlisted_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, java_home = self._official_runtime_paths(root)
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=python,
                java_home=java_home,
            )
            process = MagicMock()
            with (
                patch.dict(
                    os.environ,
                    {
                        "PRIVATE_PROVIDER_TOKEN": "must-not-enter-webshop-environment",
                        "PYTHONPATH": "/private/injected/path",
                    },
                ),
                patch("rethinkskill_webshop.runtime.subprocess.Popen", return_value=process) as popen,
                patch.object(
                    runtime._BridgeEpisode,
                    "_read",
                    return_value={"status": "ready", "protocol": 1},
                ),
            ):
                episode = factory.open(
                    session=0,
                    num_products=None,
                    verifier_workspace=root / "verifier",
                    timeout_seconds=30,
                )
            try:
                child_environment = popen.call_args.kwargs["env"]
                self.assertNotIn("PRIVATE_PROVIDER_TOKEN", child_environment)
                self.assertNotIn("PYTHONPATH", child_environment)
                self.assertEqual(child_environment["PYTHONNOUSERSITE"], "1")
                self.assertEqual(child_environment["JAVA_HOME"], str(java_home.resolve()))
            finally:
                episode.log_stream.close()

    def test_bridge_launch_failure_closes_log_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, java_home = self._official_runtime_paths(root)
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=python,
                java_home=java_home,
            )
            stream = MagicMock(closed=False)
            with (
                patch(
                    "rethinkskill_webshop.runtime.subprocess.Popen",
                    side_effect=FileNotFoundError("synthetic launcher failure"),
                ),
                patch.object(Path, "open", return_value=stream),
                self.assertRaises(FileNotFoundError),
            ):
                factory.open(
                    session=0,
                    num_products=None,
                    verifier_workspace=root / "verifier",
                    timeout_seconds=30,
                )
            stream.close.assert_called_once_with()

    def test_evaluate_uses_official_reward(self) -> None:
        harness = WebShopHarness(environment_factory=ReadyFactory())
        passed = harness.evaluate(None, '{"reward":1.0}')  # type: ignore[arg-type]
        failed = harness.evaluate(None, '{"reward":0.4}')  # type: ignore[arg-type]
        self.assertEqual(passed.verdict, Verdict.PASS)
        self.assertEqual(failed.verdict, Verdict.FAIL)
        self.assertEqual(failed.metrics["score"], 0.4)


if __name__ == "__main__":
    unittest.main()
