"""Optional ALFWorld interactive harness."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rethinkskill.benchmarks.scoring import Verdict, Verification, verify_alfworld
from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.runtime.interactive import (
    InteractiveEpisodePolicy,
    execute_interactive_episode,
)
from rethinkskill.runtime.tasks import (
    NativeAsset,
    NativeTask,
    RenderedTask,
    dataset_files,
    load_json_records,
    select_tasks,
)
from rethinkskill.runtime.types import ModelExecutor, ModelOutcome
from rethinkskill.utils.serde import strict_json_loads
from rethinkskill_alfworld.runtime import (
    AlfWorldEnvironmentFactory,
    AlfWorldState,
    InstalledAlfWorldFactory,
    extract_alfworld_action,
)


def _resolve_alfworld_gamefile(root: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        if "json_2.1.1" in raw.parts:
            index = raw.parts.index("json_2.1.1")
            candidates.append(root.joinpath(*raw.parts[index:]))
    else:
        candidates.append(root / raw)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            if not resolved.is_relative_to(root):
                raise ResultValidationError(f"ALFWorld gamefile escapes --asset-root: {resolved}")
            return resolved
    raise ResultValidationError(f"ALFWorld gamefile is absent under --asset-root: {value}")


@dataclass(frozen=True, slots=True)
class AlfWorldHarness:
    """One-episode-at-a-time native ALFWorld interaction loop."""

    environment_factory: AlfWorldEnvironmentFactory = field(
        default_factory=InstalledAlfWorldFactory
    )
    max_steps: int = 50

    def dependency_manifest(self) -> Mapping[str, object]:
        return dict(self.environment_factory.dependency_manifest())

    def provenance_files(
        self,
        source: Path,
        *,
        split: str,
    ) -> tuple[Path, ...]:
        return dataset_files(source, split)

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
        if asset_root is None:
            raise ConfigurationError(
                "ALFWorld native execution requires --asset-root pointing "
                "to the frozen ALFWorld game corpus"
            )
        dependency = dict(self.environment_factory.dependency_manifest())
        ready = dependency.get(
            "ready",
            dependency.get("alfworld_available", False),
        )
        if ready is not True:
            raise ConfigurationError(
                "ALFWorld native execution requires alfworld==0.4.2; "
                f"observed {dependency.get('version') or 'not installed'}"
            )
        root = asset_root.expanduser().resolve()
        domain = root / "logic/alfred.pddl"
        grammar = root / "logic/alfred.twl2"
        if not domain.is_file() or not grammar.is_file():
            raise ResultValidationError(
                "ALFWorld asset root is missing logic/alfred.pddl or logic/alfred.twl2"
            )
        tasks: list[NativeTask] = []
        for path in dataset_files(source, split):
            for record in load_json_records(path):
                raw_gamefile = record.get("gamefile")
                if type(raw_gamefile) is not str or not raw_gamefile.strip():
                    raise ResultValidationError("ALFWorld gamefile must be non-empty text")
                raw_gamefile = raw_gamefile.strip()
                gamefile = _resolve_alfworld_gamefile(
                    root,
                    raw_gamefile,
                )
                raw_id = record.get("id")
                if raw_id is None:
                    task_id = f"gamefile:{Path(raw_gamefile).as_posix()}"
                elif type(raw_id) is str and raw_id.strip():
                    task_id = raw_id.strip()
                else:
                    raise ResultValidationError("ALFWorld id must be non-empty text when supplied")
                assets = [
                    NativeAsset.freeze(
                        gamefile,
                        target="environment/game.tw-pddl",
                        media_type="application/x-pddl",
                        role="environment_gamefile",
                        visibility="verifier",
                    ),
                    NativeAsset.freeze(
                        domain,
                        target="logic/alfred.pddl",
                        media_type="application/x-pddl",
                        role="environment_domain",
                        visibility="verifier",
                    ),
                    NativeAsset.freeze(
                        grammar,
                        target="logic/alfred.twl2",
                        media_type="text/plain",
                        role="environment_grammar",
                        visibility="verifier",
                    ),
                ]
                trajectory = gamefile.parent / "traj_data.json"
                if trajectory.is_file():
                    assets.append(
                        NativeAsset.freeze(
                            trajectory,
                            target="environment/traj_data.json",
                            media_type="application/json",
                            role="environment_metadata",
                            visibility="verifier",
                        )
                    )
                tasks.append(
                    NativeTask(
                        task_id=task_id,
                        payload={
                            "data_root": str(root),
                            "split": split,
                            "seed": seed,
                            "max_steps": self.max_steps,
                            "gamefile_target": "environment/game.tw-pddl",
                            "execution_mode": "interactive_episode",
                        },
                        assets=tuple(assets),
                    )
                )
        return select_tasks(
            tasks,
            limit=limit,
            requested_ids=requested_ids,
        )

    def render(self, task: NativeTask, skill: str) -> RenderedTask:
        task.validate()
        rendered = RenderedTask(
            task_markdown=(
                "# ALFWorld Interactive Episode\n\n"
                f"Task ID: `{task.task_id}`\n\n"
                f"Maximum steps: {task.payload['max_steps']}\n\n"
                "The environment is initialized by the harness. At every "
                "step you will receive the current observation and admissible "
                "actions. Return exactly "
                "`<think>brief reasoning</think><action>one action</action>`. "
                "Use English only. Do not call tools or access files."
            ),
            skill_markdown=(
                "---\n"
                'name: "rethinkskill-target"\n'
                'description: "Dynamic skill for native ALFWorld '
                'interaction."\n'
                "---\n\n"
                f"{skill.strip()}\n"
            ),
            invocation=(
                "Read `.agents/skills/rethinkskill-target/SKILL.md` and "
                "`task.md`, then return one ALFWorld action using the exact "
                "<think>...</think><action>...</action> contract."
            ),
        )
        rendered.validate()
        return rendered

    def _step_rendered(
        self,
        rendered: RenderedTask,
        *,
        step_index: int,
        observation: str,
        actions: Sequence[str],
        history: Sequence[Mapping[str, object]],
    ) -> RenderedTask:
        recent = history[-2:]
        history_text = (
            "\n".join(
                f"- Step {row['step']}: action={row['action']!r}; "
                f"observation={str(row['observation'])[:800]!r}"
                for row in recent
            )
            or "(none)"
        )
        action_text = (
            "\n".join(f"- `{action}`" for action in actions)
            or "- `(environment supplied no admissible actions)`"
        )
        value = RenderedTask(
            task_markdown=(
                "# ALFWorld Action Step\n\n"
                f"Step: {step_index + 1}\n\n"
                "## Recent history\n\n"
                f"{history_text}\n\n"
                "## Current observation\n\n"
                f"{observation}\n\n"
                "## Admissible actions\n\n"
                f"{action_text}\n\n"
                "Return exactly `<think>brief reasoning</think><action>one "
                "action</action>`. Use English only."
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
        gamefile = verifier_workspace / str(task.payload["gamefile_target"])
        return execute_interactive_episode(
            InteractiveEpisodePolicy(
                kind="alfworld_episode",
                initialization_label=("ALFWorld environment initialization"),
                environment_label="ALFWorld environment",
                action_failure="output_contract_error:alfworld_action",
                open_episode=lambda: self.environment_factory.open(
                    gamefile=gamefile,
                    data_root=Path(task.payload["data_root"]),
                    verifier_workspace=verifier_workspace,
                    seed=task.payload["seed"],
                    split=task.payload["split"],
                ),
                state_done=_alfworld_done,
                render_step=lambda base, state, index, history: self._step_rendered(
                    base,
                    step_index=index,
                    observation=_alfworld_state(state).observation,
                    actions=_alfworld_state(state).admissible_actions,
                    history=history,
                ),
                parse_action=extract_alfworld_action,
                transcript_row=_alfworld_transcript_row,
                behavior_failure_row=_alfworld_behavior_failure_row,
                summarize=_alfworld_summary,
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
            summary = strict_json_loads(response)
        except json.JSONDecodeError:
            return Verification(
                Verdict.FAIL,
                "alfworld_episode_summary_invalid_json",
            )
        if not isinstance(summary, dict) or not isinstance(
            summary.get("won"),
            bool,
        ):
            return Verification(
                Verdict.FAIL,
                "alfworld_episode_summary_missing_won",
            )
        return verify_alfworld(won=summary["won"])


def _alfworld_state(value: object) -> AlfWorldState:
    if type(value) is not AlfWorldState:
        raise TypeError("ALFWorld episode returned an invalid state")
    return value


def _alfworld_done(value: object) -> bool:
    return _alfworld_state(value).done


def _alfworld_transcript_row(
    step: int,
    previous: object,
    response: str,
    action: str,
    current: object,
) -> Mapping[str, object]:
    before = _alfworld_state(previous)
    after = _alfworld_state(current)
    return {
        "step": step,
        "observation": before.observation,
        "admissible_actions": list(before.admissible_actions),
        "model_response": response,
        "action": action,
        "environment_feedback": after.observation,
        "done": after.done,
        "won": after.won,
    }


def _alfworld_behavior_failure_row(
    step: int,
    state: object,
    response: str,
    failure: str,
) -> Mapping[str, object]:
    return {
        "step": step,
        "observation": _alfworld_state(state).observation,
        "model_response": response,
        "behavior_failure": failure,
    }


def _alfworld_summary(
    state: object | None,
    transcript: Sequence[Mapping[str, object]],
    behavior_failure: str,
) -> Mapping[str, object]:
    current = _alfworld_state(state)
    return {
        "won": current.won,
        "done": current.done,
        "steps": len(transcript),
        "behavior_failure": behavior_failure,
    }
