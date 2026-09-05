"""Bounded subprocess bridge to an independently installed WebShop runtime."""

from __future__ import annotations

import json
import math
import os
import re
import selectors
import signal
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from rethinkskill.errors import ConfigurationError
from rethinkskill.utils.serde import (
    freeze_json_mapping,
    strict_json_loads,
)

_ROOT_ENV = "RETHINKSKILL_WEBSHOP_ROOT"
_PYTHON_ENV = "RETHINKSKILL_WEBSHOP_PYTHON"
_JAVA_HOME_ENV = "JAVA_HOME"
_CHILD_ENVIRONMENT_ALLOWLIST = (
    "HOME",
    "JAVA_HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "TZ",
)
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ERROR_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def _child_environment() -> dict[str, str]:
    environment = {
        key: os.environ[key] for key in _CHILD_ENVIRONMENT_ALLOWLIST if key in os.environ
    }
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    return environment


@dataclass(frozen=True, slots=True)
class WebShopState:
    observation: str
    available_actions: Mapping[str, object]
    reward: float
    done: bool
    info: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.observation) is not str:
            raise TypeError("WebShop observation must be exact text")
        if type(self.reward) not in (int, float) or not math.isfinite(self.reward):
            raise TypeError("WebShop reward must be a finite exact number")
        if type(self.done) is not bool:
            raise TypeError("WebShop done state must be an exact boolean")
        object.__setattr__(
            self,
            "available_actions",
            freeze_json_mapping(self.available_actions),
        )
        object.__setattr__(self, "info", freeze_json_mapping(self.info))


class WebShopEpisode(Protocol):
    def reset(self) -> WebShopState: ...

    def step(self, action: str) -> WebShopState: ...

    def close(self) -> None: ...


class WebShopEnvironmentFactory(Protocol):
    def dependency_manifest(self) -> Mapping[str, object]: ...

    def open(
        self,
        *,
        session: int,
        num_products: int | None,
        verifier_workspace: Path,
        timeout_seconds: int,
    ) -> WebShopEpisode: ...


@dataclass(slots=True)
class _BridgeEpisode:
    process: subprocess.Popen[str]
    session: int
    timeout_seconds: int
    log_path: Path
    log_stream: TextIO

    def _read(self) -> Mapping[str, object]:
        if self.process.stdout is None:
            raise ConfigurationError("WebShop bridge stdout is unavailable")
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            events = selector.select(self.timeout_seconds)
        finally:
            selector.close()
        if not events:
            self._terminate()
            raise ConfigurationError("WebShop bridge timed out")
        line = self.process.stdout.readline()
        if not line:
            raise ConfigurationError("WebShop bridge exited before a response")
        try:
            value = strict_json_loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigurationError("WebShop bridge returned invalid JSON") from exc
        if type(value) is not dict:
            raise ConfigurationError("WebShop bridge response must be an object")
        if value.get("status") == "error":
            code = value.get("error_code")
            error_type = value.get("error_type")
            if (
                set(value) != {"status", "error_code", "error_type"}
                or type(code) is not str
                or _ERROR_CODE.fullmatch(code) is None
                or type(error_type) is not str
                or _ERROR_TYPE.fullmatch(error_type) is None
            ):
                raise ConfigurationError("WebShop bridge returned invalid error evidence")
            raise ConfigurationError(f"WebShop bridge reported {code}")
        return value

    def _write(self, value: Mapping[str, object]) -> Mapping[str, object]:
        if self.process.stdin is None or self.process.poll() is not None:
            raise ConfigurationError("WebShop bridge is not running")
        self.process.stdin.write(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        self.process.stdin.flush()
        return self._read()

    @staticmethod
    def _state(value: Mapping[str, object]) -> WebShopState:
        if (
            set(value)
            != {
                "status",
                "observation",
                "available_actions",
                "reward",
                "done",
                "info",
            }
            or value.get("status") != "state"
        ):
            raise ConfigurationError("WebShop bridge state has an invalid schema")
        actions = value.get("available_actions")
        info = value.get("info")
        if type(actions) is not dict:
            raise ConfigurationError("WebShop bridge state has no available_actions object")
        if info is None:
            info = {}
        if type(info) is not dict:
            raise ConfigurationError("WebShop bridge info must be an object")
        observation = value.get("observation")
        reward = value.get("reward")
        done = value.get("done")
        if (
            type(observation) is not str
            or type(reward) not in (int, float)
            or not math.isfinite(reward)
            or type(done) is not bool
        ):
            raise ConfigurationError("WebShop bridge returned an invalid state")
        return WebShopState(
            observation=observation,
            available_actions=actions,
            reward=float(reward),
            done=done,
            info=info,
        )

    def reset(self) -> WebShopState:
        return self._state(self._write({"op": "reset", "session": self.session}))

    def step(self, action: str) -> WebShopState:
        return self._state(self._write({"op": "step", "action": action}))

    def _terminate(self) -> None:
        try:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
        finally:
            if not self.log_stream.closed:
                self.log_stream.close()

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._write({"op": "close"})
            except (BrokenPipeError, ConfigurationError):
                pass
        self._terminate()


@dataclass(frozen=True, slots=True)
class SubprocessWebShopFactory:
    """Launch the official Python 3.8 environment as a separate process."""

    root: Path | None = None
    python: Path | None = None
    java_home: Path | None = None

    @classmethod
    def from_environment(cls) -> SubprocessWebShopFactory:
        root = os.environ.get(_ROOT_ENV, "").strip()
        python = os.environ.get(_PYTHON_ENV, "").strip()
        java_home = os.environ.get(_JAVA_HOME_ENV, "").strip()
        return cls(
            root=Path(root).expanduser() if root else None,
            python=Path(python).expanduser() if python else None,
            java_home=(Path(java_home).expanduser() if java_home else None),
        )

    def dependency_manifest(self) -> Mapping[str, object]:
        root = self.root.resolve() if self.root is not None else None
        python = self.python.expanduser().absolute() if self.python is not None else None
        java_home = self.java_home.expanduser().absolute() if self.java_home is not None else None
        java = java_home / "bin/java" if java_home is not None else None
        environment_source = (
            root / "web_agent_site/envs/web_agent_text_env.py" if root is not None else None
        )
        data_root = root / "data" if root is not None else None
        search_root = root / "search_engine" if root is not None else None
        python_ready = bool(python is not None and python.is_file() and os.access(python, os.X_OK))
        java_ready = bool(
            java_home is not None
            and java_home.is_dir()
            and java is not None
            and java.is_file()
            and os.access(java, os.X_OK)
        )
        checkout_ready = bool(
            root is not None
            and root.is_dir()
            and environment_source is not None
            and environment_source.is_file()
            and data_root is not None
            and data_root.is_dir()
            and search_root is not None
            and search_root.is_dir()
        )
        return {
            "kind": "official_webshop_subprocess",
            "ready": python_ready and java_ready and checkout_ready,
            "python_configured": self.python is not None,
            "java_home_configured": self.java_home is not None,
            "checkout_configured": self.root is not None,
            "python_ready": python_ready,
            "java_ready": java_ready,
            "checkout_ready": checkout_ready,
            "python": str(python) if python is not None else None,
            "java_home": str(java_home) if java_home is not None else None,
            "java": str(java) if java is not None else None,
            "checkout_root": str(root) if root is not None else None,
            "configuration_environment": {
                "python": _PYTHON_ENV,
                "java_home": _JAVA_HOME_ENV,
                "checkout_root": _ROOT_ENV,
            },
            "child_environment_allowlist": list(_CHILD_ENVIRONMENT_ALLOWLIST),
        }

    def _runtime_environment(self) -> dict[str, str]:
        assert self.java_home is not None
        environment = _child_environment()
        environment[_JAVA_HOME_ENV] = str(self.java_home.expanduser().resolve(strict=True))
        return environment

    def open(
        self,
        *,
        session: int,
        num_products: int | None,
        verifier_workspace: Path,
        timeout_seconds: int,
    ) -> WebShopEpisode:
        manifest = self.dependency_manifest()
        if manifest["ready"] is not True:
            raise ConfigurationError(
                "WebShop runtime is not ready; configure "
                f"{_ROOT_ENV}, {_PYTHON_ENV}, and {_JAVA_HOME_ENV}"
            )
        assert self.root is not None
        assert self.python is not None
        bridge = Path(__file__).with_name("bridge.py").resolve()
        verifier_workspace.mkdir(parents=True, exist_ok=True)
        log_path = verifier_workspace / "webshop_bridge.log"
        log_stream = log_path.open("w", encoding="utf-8")
        argv = [
            str(self.python.expanduser().absolute()),
            str(bridge),
            "--root",
            str(self.root.resolve()),
        ]
        if num_products is not None:
            argv.extend(("--num-products", str(num_products)))
        try:
            process = subprocess.Popen(
                argv,
                cwd=self.root.resolve(),
                env=self._runtime_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log_stream,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError:
            log_stream.close()
            raise
        episode = _BridgeEpisode(
            process=process,
            session=session,
            timeout_seconds=max(1, timeout_seconds),
            log_path=log_path,
            log_stream=log_stream,
        )
        started = time.monotonic()
        try:
            ready = episode._read()
        except Exception:
            episode._terminate()
            raise
        if ready != {"status": "ready", "protocol": 1}:
            episode._terminate()
            raise ConfigurationError("WebShop bridge failed readiness")
        if time.monotonic() - started > timeout_seconds:
            episode._terminate()
            raise ConfigurationError("WebShop bridge readiness exceeded timeout")
        return episode


__all__ = [
    "SubprocessWebShopFactory",
    "WebShopEnvironmentFactory",
    "WebShopEpisode",
    "WebShopState",
]
