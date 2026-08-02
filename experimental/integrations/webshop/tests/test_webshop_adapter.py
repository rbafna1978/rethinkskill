from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path
from unittest.mock import MagicMock, patch

import rethinkskill_webshop.runtime as runtime
from rethinkskill_webshop.bridge import _error, _state, _strict_json_loads
from rethinkskill_webshop.harness import (
    WebShopHarness,
    extract_webshop_action,
)
from rethinkskill_webshop.plugin import spec
from rethinkskill_webshop.runtime import (
    SubprocessWebShopFactory,
    WebShopState,
    _BridgeEpisode,
)

from rethinkskill.benchmarks.scoring import Verdict
from rethinkskill.errors import ConfigurationError
from rethinkskill.runtime.types import ModelOutcome
from rethinkskill.utils.serde import canonical_json_bytes, sha256_bytes


def _python_environment_identity() -> dict[str, object]:
    distributions = [
        {
            "distribution": "fixture-package",
            "normalized_name": "fixture-package",
            "version": "1.0",
            "sha256": "1" * 64,
            "files": 2,
            "bytes": 7,
        }
    ]
    return {
        "schema_version": 1,
        "kind": "python_distribution_environment",
        "python": {
            "implementation": "cpython",
            "version": [3, 8, 20],
            "cache_tag": "cpython-38",
        },
        "sha256": sha256_bytes(canonical_json_bytes(distributions)),
        "distributions": distributions,
        "distribution_count": 1,
        "files": 2,
        "bytes": 7,
    }


class _Episode:
    def __init__(self, *, close_error: bool = False) -> None:
        self.closed = False
        self.close_error = close_error

    def reset(self) -> WebShopState:
        return WebShopState(
            observation="Instruction: buy a red cup",
            available_actions={
                "has_search_bar": True,
                "clickables": ["search"],
            },
            reward=0.0,
            done=False,
            info={},
        )

    def step(self, action: str) -> WebShopState:
        if action != "search[red cup]":
            raise AssertionError(action)
        return WebShopState(
            observation="Purchase complete",
            available_actions={
                "has_search_bar": False,
                "clickables": [],
            },
            reward=1.0,
            done=True,
            info={},
        )

    def close(self) -> None:
        self.closed = True
        if self.close_error:
            raise RuntimeError("synthetic close failure")


class _Factory:
    def __init__(self, *, close_error: bool = False) -> None:
        self.episode = _Episode(close_error=close_error)

    def dependency_manifest(self):
        return {
            "kind": "fake_webshop",
            "ready": True,
        }

    def open(self, **kwargs):
        self.kwargs = kwargs
        return self.episode


class _FailingFactory:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def dependency_manifest(self):
        return {"kind": "failing_webshop", "ready": True}

    def open(self, **kwargs):
        del kwargs
        raise RuntimeError(self.secret)


class _FailingStepEpisode(_Episode):
    def __init__(self, secret: str) -> None:
        super().__init__()
        self.secret = secret

    def step(self, action: str) -> WebShopState:
        del action
        raise RuntimeError(self.secret)


class _FailingStepFactory(_Factory):
    def __init__(self, secret: str) -> None:
        self.episode = _FailingStepEpisode(secret)


class _Executor:
    def public_manifest(self):
        return {"kind": "fake"}

    def execute(self, rendered, *, workspace, timeout_seconds):
        return ModelOutcome(
            status="COMPLETED",
            response=(
                "<think>Search for the requested product.</think><action>search[red cup]</action>"
            ),
            raw="fake",
            process={"kind": "fake"},
            attempted_calls=1,
            completed_calls=1,
        )


class WebShopAdapterTests(unittest.TestCase):
    def test_bridge_normalizes_official_none_info_to_object(self) -> None:
        environment = MagicMock()
        environment.get_available_actions.return_value = {
            "has_search_bar": True,
            "clickables": ["search"],
        }
        value = _state(
            environment,
            "Instruction: buy a red cup",
            0.0,
            False,
            None,
        )
        self.assertEqual(value["info"], {})

    def test_bridge_reader_rejects_raw_error_without_echoing_it(
        self,
    ) -> None:
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
        self.assertEqual(
            str(caught.exception),
            "WebShop bridge returned invalid error evidence",
        )
        self.assertNotIn(secret, str(caught.exception))

    def test_state_detaches_mappings_and_rejects_coercible_scalars(self) -> None:
        actions = {"clickables": ["buy"]}
        info = {"session": 1}
        state = WebShopState(
            observation="ready",
            available_actions=actions,
            reward=0.0,
            done=False,
            info=info,
        )
        actions["clickables"].append("secret")
        info["session"] = 2
        self.assertEqual(list(state.available_actions["clickables"]), ["buy"])
        self.assertEqual(state.info["session"], 1)
        for reward in (True, float("nan"), "1"):
            with self.subTest(reward=reward), self.assertRaises(TypeError):
                WebShopState(
                    observation="ready",
                    available_actions={},
                    reward=reward,  # type: ignore[arg-type]
                    done=False,
                    info={},
                )

    def test_bridge_error_evidence_never_contains_exception_message(self) -> None:
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

    def test_bridge_rejects_duplicate_json_object_keys(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "duplicate JSON object key",
        ):
            _strict_json_loads('{"op": "reset", "op": "close"}')

    def _official_runtime_paths(
        self,
        root: Path,
    ) -> tuple[Path, Path, Path]:
        checkout = root / "webshop"
        (checkout / "web_agent_site/envs").mkdir(parents=True)
        (checkout / "web_agent_site/envs/web_agent_text_env.py").write_text(
            "# fixture",
            encoding="utf-8",
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
        (java_home / "release").write_text(
            'JAVA_VERSION="11.0.0"\n',
            encoding="utf-8",
        )
        return checkout, python, java_home

    def test_action_contract(self) -> None:
        self.assertEqual(
            extract_webshop_action("<think>x</think><action>click[buy now]</action>"),
            "click[buy now]",
        )
        with self.assertRaises(ValueError):
            extract_webshop_action("click[buy now]")

    def test_plugin_uses_harness_owned_evaluation(self) -> None:
        value = spec()
        self.assertEqual(value.name, "webshop")
        self.assertEqual(value.evaluation_authority.value, "harness")

    def test_fake_official_episode_contract(self) -> None:
        factory = _Factory()
        harness = WebShopHarness(environment_factory=factory)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "items.json"
            dataset.write_text(
                json.dumps([{"id": "shop-0", "session": 0}]),
                encoding="utf-8",
            )
            tasks = harness.load_tasks(
                dataset,
                split="test",
                limit=1,
                requested_ids=(),
                seed=42,
                asset_root=None,
            )
            rendered = harness.render(tasks[0], "Use exact click labels.")
            outcome = harness.execute_task(
                tasks[0],
                rendered,
                _Executor(),
                workspace=root / "workspace",
                verifier_workspace=root / "verifier",
                timeout_seconds=30,
            )
            self.assertTrue(outcome.ok)
            self.assertEqual(outcome.attempted_calls, 1)
            verification = harness.evaluate(tasks[0], outcome.response)
            self.assertIs(verification.verdict, Verdict.PASS)
            self.assertEqual(verification.metrics["score"], 1.0)
            self.assertTrue(factory.episode.closed)

    def test_environment_initialization_failure_redacts_exception_message(
        self,
    ) -> None:
        secret = "webshop-environment-secret"
        harness = WebShopHarness(environment_factory=_FailingFactory(secret))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "items.json"
            dataset.write_text(
                json.dumps([{"id": "shop-0", "session": 0}]),
                encoding="utf-8",
            )
            task = harness.load_tasks(
                dataset,
                split="test",
                limit=1,
                requested_ids=(),
                seed=42,
                asset_root=None,
            )[0]
            outcome = harness.execute_task(
                task,
                harness.render(task, "Use exact labels."),
                _Executor(),
                workspace=root / "workspace",
                verifier_workspace=root / "verifier",
                timeout_seconds=30,
            )
        self.assertEqual(
            outcome.failure_class,
            "environment_initialization_error",
        )
        self.assertIn("RuntimeError", outcome.failure)
        self.assertNotIn(secret, outcome.failure)

    def test_environment_step_failure_redacts_exception_message(self) -> None:
        secret = "webshop-step-secret"
        harness = WebShopHarness(environment_factory=_FailingStepFactory(secret))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "items.json"
            dataset.write_text(
                json.dumps([{"id": "shop-0", "session": 0}]),
                encoding="utf-8",
            )
            task = harness.load_tasks(
                dataset,
                split="test",
                limit=1,
                requested_ids=(),
                seed=42,
                asset_root=None,
            )[0]
            outcome = harness.execute_task(
                task,
                harness.render(task, "Use exact labels."),
                _Executor(),
                workspace=root / "workspace",
                verifier_workspace=root / "verifier",
                timeout_seconds=30,
            )
        self.assertEqual(outcome.failure_class, "environment_error")
        self.assertIn("RuntimeError", outcome.failure)
        self.assertNotIn(secret, json.dumps(outcome.public()))

    def test_close_failure_does_not_override_terminal_reward(self) -> None:
        factory = _Factory(close_error=True)
        harness = WebShopHarness(environment_factory=factory)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "items.json"
            dataset.write_text(
                json.dumps([{"id": "shop-0", "session": 0}]),
                encoding="utf-8",
            )
            task = harness.load_tasks(
                dataset,
                split="test",
                limit=1,
                requested_ids=(),
                seed=42,
                asset_root=None,
            )[0]
            outcome = harness.execute_task(
                task,
                harness.render(task, "Use exact click labels."),
                _Executor(),
                workspace=root / "workspace",
                verifier_workspace=root / "verifier",
                timeout_seconds=30,
            )
        self.assertTrue(factory.episode.closed)
        self.assertTrue(outcome.ok)
        self.assertEqual(json.loads(outcome.response)["reward"], 1.0)

    def test_official_bridge_receives_only_allowlisted_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, java_home = self._official_runtime_paths(root)
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=python,
                java_home=java_home,
            )
            process = MagicMock()
            os.environ["PRIVATE_PROVIDER_TOKEN"] = "must-not-enter-webshop-environment"
            os.environ["PYTHONPATH"] = "/private/injected/path"
            try:
                with (
                    patch(
                        "rethinkskill_webshop.runtime.subprocess.Popen",
                        return_value=process,
                    ) as popen,
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
            finally:
                os.environ.pop("PRIVATE_PROVIDER_TOKEN", None)
                os.environ.pop("PYTHONPATH", None)
            try:
                child_environment = popen.call_args.kwargs["env"]
                self.assertNotIn(
                    "PRIVATE_PROVIDER_TOKEN",
                    child_environment,
                )
                self.assertNotIn("PYTHONPATH", child_environment)
                self.assertEqual(
                    child_environment["PYTHONNOUSERSITE"],
                    "1",
                )
                self.assertEqual(
                    child_environment["JAVA_HOME"],
                    str(java_home.resolve()),
                )
            finally:
                episode.log_stream.close()

    def test_runtime_identity_freezes_code_data_python_and_java(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, java_home = self._official_runtime_paths(root)
            (checkout / "data/items.json").write_text(
                '[{"id":1}]\n',
                encoding="utf-8",
            )
            (checkout / "search_engine/index.json").write_text(
                '{"version":1}\n',
                encoding="utf-8",
            )
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=python,
                java_home=java_home,
            )
            with patch.object(
                SubprocessWebShopFactory,
                "_python_environment_identity",
                return_value=_python_environment_identity(),
            ):
                first = factory.runtime_identity()
                (checkout / "data/items.json").write_text(
                    '[{"id":2}]\n',
                    encoding="utf-8",
                )
                second = factory.runtime_identity()
                (java_home / "release").write_text(
                    'JAVA_VERSION="11.0.1"\n',
                    encoding="utf-8",
                )
                third = factory.runtime_identity()
        self.assertEqual(first["schema_version"], 3)
        self.assertTrue(first["python_environment_packages_frozen"])
        self.assertTrue(first["java_runtime_files_frozen"])
        self.assertEqual(
            first["python_environment"]["distribution_count"],
            1,
        )
        self.assertNotEqual(
            first["data_tree"]["sha256"],
            second["data_tree"]["sha256"],
        )
        self.assertEqual(
            first["search_engine_tree"]["sha256"],
            second["search_engine_tree"]["sha256"],
        )
        self.assertEqual(first["python_executable"]["size"], 10)
        self.assertEqual(first["java_executable"]["size"], 10)
        self.assertEqual(first["web_agent_site_tree"]["files"], 1)
        self.assertEqual(first["java_runtime_tree"]["files"], 2)
        self.assertEqual(
            first["java_runtime_tree"]["sha256"],
            second["java_runtime_tree"]["sha256"],
        )
        self.assertNotEqual(
            second["java_runtime_tree"]["sha256"],
            third["java_runtime_tree"]["sha256"],
        )

    def test_dependency_manifest_requires_explicit_java_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, _ = self._official_runtime_paths(root)
            manifest = SubprocessWebShopFactory(
                root=checkout,
                python=python,
            ).dependency_manifest()
        self.assertFalse(manifest["ready"])
        self.assertFalse(manifest["java_home_configured"])
        self.assertFalse(manifest["java_ready"])

    def test_environment_probe_is_isolated_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, java_home = self._official_runtime_paths(root)
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=python,
                java_home=java_home,
            )
            response = json.dumps(
                {
                    "status": "ok",
                    "identity": _python_environment_identity(),
                },
                sort_keys=True,
            ).encode("utf-8")
            completed = MagicMock(returncode=0, stdout=response)
            os.environ["PRIVATE_PROVIDER_TOKEN"] = "probe-secret"
            os.environ["PYTHONPATH"] = "/private/injected/path"
            try:
                with patch(
                    "rethinkskill_webshop.environment_identity.subprocess.run",
                    return_value=completed,
                ) as run:
                    identity = factory._python_environment_identity()
            finally:
                os.environ.pop("PRIVATE_PROVIDER_TOKEN", None)
                os.environ.pop("PYTHONPATH", None)
        self.assertEqual(identity["distribution_count"], 1)
        argv = run.call_args.args[0]
        self.assertEqual(argv[1:3], ["-B", "-I"])
        child_environment = run.call_args.kwargs["env"]
        self.assertNotIn("PRIVATE_PROVIDER_TOKEN", child_environment)
        self.assertNotIn("PYTHONPATH", child_environment)
        self.assertEqual(child_environment["PYTHONNOUSERSITE"], "1")

    def test_environment_probe_preserves_configured_venv_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, _, java_home = self._official_runtime_paths(root)
            launcher = root / "venv/bin/python"
            launcher.parent.mkdir(parents=True)
            launcher.symlink_to(Path(sys.executable).resolve())
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=launcher,
                java_home=java_home,
            )
            response = json.dumps(
                {
                    "status": "ok",
                    "identity": _python_environment_identity(),
                }
            ).encode("utf-8")
            completed = MagicMock(returncode=0, stdout=response)
            with patch(
                "rethinkskill_webshop.environment_identity.subprocess.run",
                return_value=completed,
            ) as run:
                factory._python_environment_identity()
            executable = factory._python_executable_identity(launcher)
        self.assertEqual(run.call_args.args[0][0], str(launcher.absolute()))
        self.assertTrue(executable["configured_is_symlink"])
        self.assertEqual(executable["configured_path"], str(launcher.absolute()))
        self.assertEqual(
            executable["resolved_path"],
            str(Path(sys.executable).resolve()),
        )

    def test_environment_probe_rejects_forged_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, java_home = self._official_runtime_paths(root)
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=python,
                java_home=java_home,
            )
            identity = _python_environment_identity()
            identity["sha256"] = "0" * 64
            completed = MagicMock(
                returncode=0,
                stdout=json.dumps({"status": "ok", "identity": identity}).encode("utf-8"),
            )
            with (
                patch(
                    "rethinkskill_webshop.environment_identity.subprocess.run",
                    return_value=completed,
                ),
                self.assertRaisesRegex(
                    ConfigurationError,
                    "summary is invalid",
                ),
            ):
                factory._python_environment_identity()

    def test_real_venv_distribution_drift_changes_runtime_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, _, java_home = self._official_runtime_paths(root)
            environment = root / "webshop-environment"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            site_packages = (
                environment
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )
            package = site_packages / "fixture_package"
            distribution = site_packages / "fixture_package-1.0.dist-info"
            package.mkdir()
            distribution.mkdir()
            source = package / "__init__.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            (distribution / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: fixture-package\nVersion: 1.0\n",
                encoding="utf-8",
            )
            (distribution / "RECORD").write_text(
                "fixture_package/__init__.py,,\n"
                "fixture_package-1.0.dist-info/METADATA,,\n"
                "fixture_package-1.0.dist-info/RECORD,,\n",
                encoding="utf-8",
            )
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=environment / "bin/python",
                java_home=java_home,
            )
            first = factory._python_environment_identity()
            source.write_text("VALUE = 2\n", encoding="utf-8")
            second = factory._python_environment_identity()
            executable = factory._python_executable_identity(environment / "bin/python")
        self.assertTrue(executable["configured_is_symlink"])
        self.assertEqual(first["distribution_count"], 1)
        self.assertNotEqual(
            first["sha256"],
            second["sha256"],
        )

    def test_real_clean_venv_probe_accepts_packaged_pth_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / "environment"
            venv.EnvBuilder(with_pip=True, symlinks=True).create(environment)
            python = environment / "bin/python"
            probe = Path(runtime.__file__).with_name("environment_probe.py")
            completed = subprocess.run(
                [str(python), "-B", "-I", str(probe)],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
        self.assertEqual(completed.returncode, 0)
        value = json.loads(completed.stdout)
        self.assertEqual(value["status"], "ok")
        self.assertGreaterEqual(value["identity"]["distribution_count"], 1)

    def test_runtime_identity_rejects_non_python_3_8_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, java_home = self._official_runtime_paths(root)
            identity = _python_environment_identity()
            identity["python"]["version"] = [3, 9, 0]
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=python,
                java_home=java_home,
            )
            with (
                patch.object(
                    SubprocessWebShopFactory,
                    "_python_environment_identity",
                    return_value=identity,
                ),
                self.assertRaisesRegex(
                    ConfigurationError,
                    "requires CPython 3.8",
                ),
            ):
                factory.runtime_identity()

    def test_runtime_identity_rejects_symlinks_in_consumed_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, java_home = self._official_runtime_paths(root)
            source = root / "outside.json"
            source.write_text("{}\n", encoding="utf-8")
            (checkout / "data/linked.json").symlink_to(source)
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=python,
                java_home=java_home,
            )
            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                factory.runtime_identity()

    def test_runtime_identity_rejects_external_java_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, java_home = self._official_runtime_paths(root)
            outside = root / "outside-java-library"
            outside.write_text("binary\n", encoding="utf-8")
            (java_home / "external-library").symlink_to(outside)
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=python,
                java_home=java_home,
            )
            with (
                patch.object(
                    SubprocessWebShopFactory,
                    "_python_environment_identity",
                    return_value=_python_environment_identity(),
                ),
                self.assertRaisesRegex(
                    ConfigurationError,
                    "Java runtime contains an external or broken symlink",
                ),
            ):
                factory.runtime_identity()

    def test_bridge_launch_failure_closes_log_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkout, python, java_home = self._official_runtime_paths(root)
            factory = SubprocessWebShopFactory(
                root=checkout,
                python=python,
                java_home=java_home,
            )
            with (
                patch(
                    "rethinkskill_webshop.runtime.subprocess.Popen",
                    side_effect=FileNotFoundError("synthetic launcher failure"),
                ),
                patch.object(
                    Path,
                    "open",
                    return_value=MagicMock(closed=False),
                ) as opened,
            ):
                with self.assertRaises(FileNotFoundError):
                    factory.open(
                        session=0,
                        num_products=None,
                        verifier_workspace=root / "verifier",
                        timeout_seconds=30,
                    )
            opened.return_value.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
