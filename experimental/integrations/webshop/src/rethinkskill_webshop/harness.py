"""Provider-neutral, harness-managed WebShop interaction loop."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rethinkskill.benchmarks.scoring import Verdict, Verification
from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.runtime.interactive import (
    InteractiveEpisodePolicy,
    execute_interactive_episode,
)
from rethinkskill.runtime.tasks import (
    NativeTask,
    RenderedTask,
    load_json_records,
    select_tasks,
)
from rethinkskill.runtime.types import ModelExecutor, ModelOutcome
from rethinkskill.utils.serde import strict_json_loads
from rethinkskill_webshop.runtime import (
    SubprocessWebShopFactory,
    WebShopEnvironmentFactory,
    WebShopState,
)

_ACTION = re.compile(
    r"^\s*<think>.*?</think>\s*<action>(.*?)</action>\s*$",
    flags=re.DOTALL,
)
_VALID_ACTION = re.compile(r"^(search|click)\[[^\r\n\[\]]+\]$")


def extract_webshop_action(value: str) -> str:
    match = _ACTION.fullmatch(value)
    if match is None:
        raise ValueError("expected <think>...</think><action>search[...] or click[...]</action>")
    action = match.group(1).strip()
    if len(action) > 500 or _VALID_ACTION.fullmatch(action) is None:
        raise ValueError(f"invalid WebShop action: {action!r}")
    return action


def _dataset_files(source: Path, split: str) -> tuple[Path, ...]:
    resolved = source.expanduser().absolute()
    if resolved.is_symlink():
        raise ResultValidationError(f"WebShop dataset source must not be a symlink: {resolved}")
    if resolved.is_file():
        return (resolved,)
    candidates = (
        resolved / split / "items.json",
        resolved / split / "items.jsonl",
        resolved / f"{split}.json",
        resolved / f"{split}.jsonl",
    )
    files = tuple(path for path in candidates if path.is_file())
    if not files:
        raise ResultValidationError(f"WebShop split {split!r} is absent under {resolved}")
    return files


@dataclass(frozen=True, slots=True)
class WebShopHarness:
    environment_factory: WebShopEnvironmentFactory = field(
        default_factory=SubprocessWebShopFactory.from_environment
    )
    default_max_steps: int = 15

    def dependency_manifest(self) -> Mapping[str, object]:
        return dict(self.environment_factory.dependency_manifest())

    def provenance_files(
        self,
        source: Path,
        *,
        split: str,
    ) -> tuple[Path, ...]:
        return _dataset_files(source, split)

    def load_tasks(
        self,
        source: Path,
        *,
        split: str,
        limit: int | None,
        requested_ids: Sequence[str],
        seed: int,
        asset_root: Path | None,
    ) -> tuple[NativeTask, ...]:
        del seed, asset_root
        dependency = self.dependency_manifest()
        if dependency.get("ready") is not True:
            raise ConfigurationError(
                "WebShop adapter is installed but its external runtime is not ready"
            )
        tasks: list[NativeTask] = []
        for path in _dataset_files(source, split):
            for row in load_json_records(path):
                session = row.get("session")
                if type(session) is not int or session < 0:
                    raise ResultValidationError(
                        "WebShop row requires a non-negative integer session"
                    )
                raw_id = row.get("id")
                if raw_id is None:
                    task_id = f"webshop-{session}"
                elif type(raw_id) is str and raw_id.strip():
                    task_id = raw_id.strip()
                else:
                    raise ResultValidationError("WebShop id must be non-empty text when supplied")
                max_steps = row.get("max_steps", self.default_max_steps)
                num_products = row.get("num_products")
                if type(max_steps) is not int or max_steps <= 0:
                    raise ResultValidationError(f"WebShop max_steps must be positive: {task_id}")
                if num_products is not None and (
                    type(num_products) is not int or num_products <= 0
                ):
                    raise ResultValidationError(f"WebShop num_products must be positive: {task_id}")
                tasks.append(
                    NativeTask(
                        task_id=task_id,
                        payload={
                            "session": session,
                            "max_steps": max_steps,
                            "num_products": num_products,
                            "split": split,
                        },
                    )
                )
        return select_tasks(
            tasks,
            requested_ids=requested_ids,
            limit=limit,
        )

    def render(self, task: NativeTask, skill: str) -> RenderedTask:
        task.validate()
        rendered = RenderedTask(
            task_markdown=(
                "# WebShop Task\n\n"
                f"Task ID: `{task.task_id}`\n\n"
                f"Official session index: `{task.payload['session']}`\n\n"
                "The official environment will reveal the instruction and "
                "available actions after reset. Follow the supplied skill and "
                "finish the purchase within the bounded step budget."
            ),
            skill_markdown=(
                "---\n"
                'name: "rethinkskill-target"\n'
                'description: "Dynamic skill for WebShop navigation."\n'
                "---\n\n"
                f"{skill.strip()}\n"
            ),
            invocation=(
                "Return exactly `<think>brief reasoning</think>"
                "<action>one available WebShop action</action>`."
            ),
        )
        rendered.validate()
        return rendered

    @staticmethod
    def _step_rendered(
        rendered: RenderedTask,
        *,
        state: WebShopState,
        step_index: int,
        history: Sequence[Mapping[str, object]],
    ) -> RenderedTask:
        clickables = state.available_actions.get("clickables", [])
        has_search = state.available_actions.get("has_search_bar", False)
        recent = list(history[-5:])
        value = RenderedTask(
            task_markdown=(
                "# WebShop Interaction\n\n"
                f"Step: {step_index}\n\n"
                f"Search bar available: {bool(has_search)}\n\n"
                "## Observation\n\n"
                f"{state.observation}\n\n"
                "## Clickable values\n\n"
                f"{json.dumps(clickables, ensure_ascii=False, allow_nan=False)}\n\n"
                "## Recent trajectory\n\n"
                f"{json.dumps(recent, ensure_ascii=False, allow_nan=False)}\n\n"
                "Return one action using the exact environment syntax "
                "`search[keywords]` or `click[value]`."
            ),
            skill_markdown=rendered.skill_markdown,
            invocation=rendered.invocation,
        )
        value.validate()
        return value

    def execute_task(
        self,
        task: NativeTask,
        rendered: RenderedTask,
        executor: ModelExecutor,
        *,
        workspace: Path,
        verifier_workspace: Path,
        timeout_seconds: int,
    ) -> ModelOutcome:
        return execute_interactive_episode(
            InteractiveEpisodePolicy(
                kind="webshop_episode",
                initialization_label=("WebShop environment initialization"),
                environment_label="WebShop environment",
                action_failure="output_contract_error:webshop_action",
                open_episode=lambda: self.environment_factory.open(
                    session=task.payload["session"],
                    num_products=task.payload.get("num_products"),
                    verifier_workspace=verifier_workspace,
                    timeout_seconds=timeout_seconds,
                ),
                state_done=_webshop_done,
                render_step=lambda base, state, index, history: self._step_rendered(
                    base,
                    state=_webshop_state(state),
                    step_index=index,
                    history=history,
                ),
                parse_action=extract_webshop_action,
                transcript_row=_webshop_transcript_row,
                behavior_failure_row=_webshop_behavior_failure_row,
                summarize=_webshop_summary,
            ),
            rendered,
            executor,
            workspace=workspace,
            timeout_seconds=timeout_seconds,
            max_steps=task.payload["max_steps"],
        )

    def evaluate(self, task: NativeTask, response: str) -> Verification:
        del task
        try:
            value = strict_json_loads(response)
        except json.JSONDecodeError:
            return Verification(Verdict.FAIL, "webshop_summary_invalid_json")
        reward = value.get("reward") if type(value) is dict else None
        if type(reward) not in (int, float) or not math.isfinite(reward):
            return Verification(Verdict.FAIL, "webshop_summary_missing_reward")
        score = float(reward)
        success = score >= 1.0
        return Verification(
            Verdict.PASS if success else Verdict.FAIL,
            "webshop_full_reward" if success else "webshop_partial_or_zero_reward",
            metrics={"score": score, "success": float(success)},
        )


def _webshop_state(value: object) -> WebShopState:
    if type(value) is not WebShopState:
        raise TypeError("WebShop episode returned an invalid state")
    return value


def _webshop_done(value: object) -> bool:
    return _webshop_state(value).done


def _webshop_transcript_row(
    step: int,
    previous: object,
    response: str,
    action: str,
    current: object,
) -> Mapping[str, object]:
    before = _webshop_state(previous)
    after = _webshop_state(current)
    return {
        "step": step,
        "observation": before.observation,
        "available_actions": dict(before.available_actions),
        "model_response": response,
        "action": action,
        "environment_feedback": after.observation,
        "reward": after.reward,
        "done": after.done,
    }


def _webshop_behavior_failure_row(
    step: int,
    state: object,
    response: str,
    failure: str,
) -> Mapping[str, object]:
    return {
        "step": step,
        "observation": _webshop_state(state).observation,
        "model_response": response,
        "behavior_failure": failure,
    }


def _webshop_summary(
    state: object | None,
    transcript: Sequence[Mapping[str, object]],
    behavior_failure: str,
) -> Mapping[str, object]:
    current = _webshop_state(state)
    reward = float(current.reward)
    return {
        "reward": reward,
        "success": reward >= 1.0,
        "done": current.done,
        "steps": len(transcript),
        "behavior_failure": behavior_failure,
    }


__all__ = ["WebShopHarness", "extract_webshop_action"]
