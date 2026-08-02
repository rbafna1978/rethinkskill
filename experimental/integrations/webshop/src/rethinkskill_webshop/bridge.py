"""JSONL bridge executed inside the official WebShop Python environment."""

from __future__ import annotations

import argparse
import builtins
import contextlib
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_BUILTIN_EXCEPTION_NAMES = {
    value: value.__name__
    for value in vars(builtins).values()
    if isinstance(value, type) and issubclass(value, BaseException)
}


def _safe_exception_type(exc: BaseException) -> str:
    for exception_type in type(exc).__mro__:
        name = _BUILTIN_EXCEPTION_NAMES.get(exception_type)
        if name is not None:
            return name
    return "Exception"


def _error(code: str, exc: BaseException | None = None) -> dict[str, object]:
    return {
        "status": "error",
        "error_code": code,
        "error_type": (_safe_exception_type(exc) if exc is not None else "ProtocolError"),
    }


def _reject_json_constant(value: str) -> None:
    raise json.JSONDecodeError(
        f"non-standard JSON constant is forbidden: {value}",
        value,
        0,
    )


def _strict_json_object(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise json.JSONDecodeError(
                "duplicate JSON object key is forbidden",
                "",
                0,
            )
        value[key] = nested
    return value


def _strict_json_loads(value: str) -> object:
    parsed = json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_strict_json_object,
    )
    try:
        json.dumps(parsed, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise json.JSONDecodeError(
            "parsed JSON contains a non-canonical value",
            value,
            0,
        ) from exc
    return parsed


def _emit(value: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    sys.stdout.flush()


def _state(env, observation, reward, done, info) -> dict[str, object]:
    actions = env.get_available_actions()
    # The pinned official WebShop environment returns ``None`` for ``info``
    # from both reset() and ordinary step() calls. Normalize that Gym
    # shape at the external-runtime boundary so the JSON protocol always
    # exposes one stable object schema.
    if info is None:
        info = {}
    if (
        type(observation) is not str
        or type(actions) is not dict
        or type(reward) not in (int, float)
        or not math.isfinite(reward)
        or type(done) is not bool
        or type(info) is not dict
    ):
        raise TypeError("WebShop environment returned an invalid state")
    return {
        "status": "state",
        "observation": observation,
        "available_actions": actions,
        "reward": float(reward),
        "done": done,
        "info": info,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num-products", type=int)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    sys.path.insert(0, str(root))
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import gym
            from web_agent_site.envs import WebAgentTextEnv  # noqa: F401

            kwargs = {"observation_mode": "text"}
            if args.num_products is not None:
                kwargs["num_products"] = args.num_products
            env = gym.make("WebAgentTextEnv-v0", **kwargs)
        _emit({"status": "ready", "protocol": 1})
        for line in sys.stdin:
            try:
                request = _strict_json_loads(line)
                if type(request) is not dict:
                    raise TypeError("request must be a JSON object")
                operation = request.get("op")
                if operation == "reset":
                    session = request.get("session")
                    if type(session) is not int or session < 0:
                        raise TypeError("reset session must be a nonnegative integer")
                    with contextlib.redirect_stdout(sys.stderr):
                        observation, info = env.reset(session=session)
                    _emit(_state(env, observation, 0.0, False, info))
                elif operation == "step":
                    action = request.get("action")
                    if type(action) is not str or not action:
                        raise TypeError("step action must be non-empty text")
                    with contextlib.redirect_stdout(sys.stderr):
                        observation, reward, done, info = env.step(action)
                    _emit(_state(env, observation, reward, done, info))
                elif operation == "close":
                    with contextlib.redirect_stdout(sys.stderr):
                        env.close()
                    _emit({"status": "closed"})
                    return 0
                else:
                    _emit(_error("unknown_operation"))
            except Exception as exc:  # noqa: BLE001
                _emit(_error("request_failed", exc))
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit(_error("startup_failed", exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
