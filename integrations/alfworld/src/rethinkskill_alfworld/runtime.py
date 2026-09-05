"""Optional ALFWorld environment boundary and strict action contract."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.runtime.types import probe_distribution
from rethinkskill.utils.integrity import boundary_failure
from rethinkskill.utils.serde import freeze_json_mapping

_ACTION = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_CHINESE = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True, slots=True)
class AlfWorldState:
    observation: str
    admissible_actions: tuple[str, ...]
    done: bool
    won: bool
    info: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.observation) is not str:
            raise TypeError("ALFWorld observation must be exact text")
        if type(self.admissible_actions) is not tuple or any(
            type(action) is not str for action in self.admissible_actions
        ):
            raise TypeError("ALFWorld actions must be an exact text tuple")
        if type(self.done) is not bool or type(self.won) is not bool:
            raise TypeError("ALFWorld terminal state must use exact booleans")
        object.__setattr__(self, "info", freeze_json_mapping(self.info))


class AlfWorldEpisode(Protocol):
    def reset(self) -> AlfWorldState: ...

    def step(self, action: str) -> AlfWorldState: ...

    def close(self) -> None: ...


class AlfWorldEnvironmentFactory(Protocol):
    def dependency_manifest(self) -> Mapping[str, object]: ...

    def open(
        self,
        *,
        gamefile: Path,
        data_root: Path,
        verifier_workspace: Path,
        seed: int,
        split: str,
    ) -> AlfWorldEpisode: ...


def extract_alfworld_action(response: str) -> str:
    """Parse the think/action behavioral output contract."""

    raw = str(response or "")
    action = _ACTION.search(raw)
    think = _THINK.search(raw)
    if action is None:
        raise ValueError("ALFWorld response is missing <action>...</action>")
    if think is None:
        raise ValueError("ALFWorld response is missing <think>...</think>")
    if _CHINESE.search(raw):
        raise ValueError("ALFWorld response contains disallowed Chinese text")
    value = action.group(1).strip().lower()
    if not value:
        raise ValueError("ALFWorld action is empty")
    return value


def _first(value: Any, default: Any = None) -> Any:
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return value[0] if value else default
    return value if value is not None else default


def _public_info(infos: Mapping[str, Any]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, raw in infos.items():
        item = _first(raw)
        if type(key) is str and (type(item) in (str, int, float, bool) or item is None):
            value[key] = item
    return value


class _InstalledEpisode:
    def __init__(self, environment: Any):
        self.environment = environment

    def _state(
        self,
        observation: Any,
        done: Any,
        infos: Mapping[str, Any],
    ) -> AlfWorldState:
        text = _first(observation, "")
        actions = _first(infos.get("admissible_commands"), ())
        done_value = _first(done, False)
        won_value = _first(infos.get("won"), False)
        if type(text) is not str:
            raise TypeError("ALFWorld observation must be exact text")
        if (
            not isinstance(actions, Sequence)
            or isinstance(actions, (str, bytes, bytearray))
            or any(type(action) is not str for action in actions)
        ):
            raise TypeError("ALFWorld admissible actions must be text")
        if type(done_value) is not bool or type(won_value) is not bool:
            raise TypeError("ALFWorld terminal values must be exact booleans")
        return AlfWorldState(
            observation=text,
            admissible_actions=tuple(action for action in actions if action != "help"),
            done=done_value,
            won=won_value,
            info=_public_info(infos),
        )

    def reset(self) -> AlfWorldState:
        observation, infos = self.environment.reset()
        return self._state(observation, False, infos)

    def step(self, action: str) -> AlfWorldState:
        observation, _, done, infos = self.environment.step([action])
        return self._state(observation, done, infos)

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True, slots=True)
class InstalledAlfWorldFactory:
    """Lazy adapter over the pip-installed official ALFWorld package."""

    def dependency_manifest(self) -> Mapping[str, object]:
        dependency = probe_distribution(
            distribution="alfworld",
            import_name="alfworld",
            requirement="==0.4.2",
            accepts=lambda value: value == "0.4.2",
        )
        return {
            "kind": "installed_alfworld",
            **dependency,
            "alfworld_available": dependency["module_available"],
        }

    def _configuration(
        self,
        *,
        gamefile: Path,
        verifier_workspace: Path,
    ) -> dict[str, object]:
        logic = verifier_workspace / "logic"
        data = gamefile.parent
        return {
            "dataset": {
                # The verifier workspace contains exactly the selected game
                # and its trajectory metadata. Point every split at that
                # bounded directory so ALFWorld validates and reports the
                # selected game before we pin ``game_files`` explicitly.
                "data_path": str(data),
                "eval_id_data_path": str(data),
                "eval_ood_data_path": str(data),
                "num_train_games": -1,
                "num_eval_games": -1,
            },
            "logic": {
                "domain": str(logic / "alfred.pddl"),
                "grammar": str(logic / "alfred.twl2"),
            },
            "env": {
                "type": "AlfredTWEnv",
                "domain_randomization": False,
                "task_types": [1, 2, 3, 4, 5, 6],
                "expert_timeout_steps": 150,
                "expert_type": "handcoded",
                "goal_desc_human_anns_prob": 0.0,
            },
            "general": {
                "random_seed": 42,
                "use_cuda": False,
                "hide_init_receptacles": False,
                "training_method": "dagger",
            },
            "rl": {
                "action_space": "admissible",
                "max_target_length": 20,
            },
            "dagger": {
                "action_space": "generation",
                "max_target_length": 20,
                "training": {"max_nb_steps_per_episode": 50},
            },
        }

    def open(
        self,
        *,
        gamefile: Path,
        data_root: Path,
        verifier_workspace: Path,
        seed: int,
        split: str,
    ) -> AlfWorldEpisode:
        del data_root
        manifest = self.dependency_manifest()
        if not manifest["ready"]:
            raise ConfigurationError(
                "ALFWorld native execution requires alfworld==0.4.2; "
                f"observed {manifest.get('version') or 'not installed'}"
            )
        try:
            from alfworld.agents.environment import get_environment
        except ImportError as exc:
            raise ConfigurationError(
                "the installed ALFWorld package cannot import its environment"
            ) from exc
        train_eval = {
            "train": "train",
            "val": "eval_in_distribution",
            "valid_seen": "eval_in_distribution",
            "test": "eval_out_of_distribution",
            "valid_unseen": "eval_out_of_distribution",
        }.get(split, "eval_out_of_distribution")
        config = self._configuration(
            gamefile=gamefile,
            verifier_workspace=verifier_workspace,
        )
        try:
            base = get_environment("AlfredTWEnv")(
                config,
                train_eval=train_eval,
            )
            base.game_files = [str(gamefile)]
            if hasattr(base, "num_games"):
                base.num_games = 1
            environment = base.init_env(batch_size=1)
            seed_method = getattr(environment, "seed", None)
            if callable(seed_method):
                seed_method(seed)
        except Exception as exc:  # noqa: BLE001
            raise ResultValidationError(
                boundary_failure(
                    "installed ALFWorld environment initialization",
                    exc,
                )
            ) from exc
        return _InstalledEpisode(environment)
