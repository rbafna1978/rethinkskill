from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from rethinkskill.errors import ResultValidationError
from rethinkskill.evidence.interactive import (
    replay_interactive_evidence,
)
from rethinkskill.runtime.interactive import (
    InteractiveEpisodePolicy,
    execute_interactive_episode,
)
from rethinkskill.runtime.tasks import RenderedTask
from rethinkskill.runtime.types import (
    ModelOutcome,
)
from rethinkskill.utils.serde import atomic_write_json, sha256_file, thaw_json_mapping


@dataclass(frozen=True, slots=True)
class _State:
    observation: str
    done: bool
    score: int


class _Episode:
    def __init__(self, *, terminal_on_reset: bool = False) -> None:
        self.terminal_on_reset = terminal_on_reset
        self.closed = False
        self.steps = 0

    def reset(self) -> _State:
        return _State("start", self.terminal_on_reset, 0)

    def step(self, action: str) -> _State:
        self.steps += 1
        return _State(
            f"after {action}",
            action == "finish",
            self.steps,
        )

    def close(self) -> None:
        self.closed = True


class _Executor:
    def __init__(self, outcomes: Sequence[ModelOutcome]) -> None:
        self.outcomes = tuple(outcomes)
        self.calls = 0

    def public_manifest(self) -> Mapping[str, object]:
        return {"kind": "fake-interactive"}

    def execute(
        self,
        rendered: RenderedTask,
        *,
        workspace: Path,
        timeout_seconds: int,
    ) -> ModelOutcome:
        del rendered, workspace, timeout_seconds
        outcome = self.outcomes[self.calls]
        self.calls += 1
        return outcome


class _RaisingExecutor:
    def public_manifest(self) -> Mapping[str, object]:
        return {"kind": "raising-interactive"}

    def execute(
        self,
        rendered: RenderedTask,
        *,
        workspace: Path,
        timeout_seconds: int,
    ) -> ModelOutcome:
        del rendered, workspace, timeout_seconds
        raise RuntimeError("executor boundary failure")


def _completed(response: str) -> ModelOutcome:
    return ModelOutcome(
        status="COMPLETED",
        response=response,
        raw=response,
        process={"kind": "fake-step"},
        attempted_calls=1,
        completed_calls=1,
    )


def _state(value: object) -> _State:
    if type(value) is not _State:
        raise TypeError("invalid synthetic state")
    return value


def _policy(episode: _Episode) -> InteractiveEpisodePolicy:
    def parse_action(response: str) -> str:
        if response == "invalid":
            raise ValueError("invalid synthetic action")
        return response

    def render_step(
        rendered: RenderedTask,
        state: object,
        step: int,
        history: Sequence[Mapping[str, object]],
    ) -> RenderedTask:
        current = _state(state)
        value = RenderedTask(
            task_markdown=(
                f"step={step}; observation={current.observation}; history={len(history)}"
            ),
            skill_markdown=rendered.skill_markdown,
            invocation=rendered.invocation,
        )
        value.validate()
        return value

    def transcript_row(
        step: int,
        previous: object,
        response: str,
        action: str,
        current: object,
    ) -> Mapping[str, object]:
        return {
            "step": step,
            "observation": _state(previous).observation,
            "model_response": response,
            "action": action,
            "environment_feedback": _state(current).observation,
        }

    def behavior_failure_row(
        step: int,
        state: object,
        response: str,
        failure: str,
    ) -> Mapping[str, object]:
        return {
            "step": step,
            "observation": _state(state).observation,
            "model_response": response,
            "behavior_failure": failure,
        }

    def summarize(
        state: object | None,
        transcript: Sequence[Mapping[str, object]],
        behavior_failure: str,
    ) -> Mapping[str, object]:
        current = _state(state)
        return {
            "score": current.score,
            "done": current.done,
            "steps": len(transcript),
            "behavior_failure": behavior_failure,
        }

    return InteractiveEpisodePolicy(
        kind="synthetic_episode",
        initialization_label="synthetic initialization",
        environment_label="synthetic environment",
        action_failure="output_contract_error:synthetic_action",
        open_episode=lambda: episode,
        state_done=lambda value: _state(value).done,
        render_step=render_step,
        parse_action=parse_action,
        transcript_row=transcript_row,
        behavior_failure_row=behavior_failure_row,
        summarize=summarize,
    )


def _replace_parser(
    policy: InteractiveEpisodePolicy,
    parser: Callable[[str], str],
) -> InteractiveEpisodePolicy:
    return replace(policy, parse_action=parser)


def _rendered() -> RenderedTask:
    value = RenderedTask(
        task_markdown="synthetic task",
        skill_markdown="synthetic skill",
        invocation="synthetic invocation",
    )
    value.validate()
    return value


class InteractiveEpisodeTests(unittest.TestCase):
    def test_success_preserves_transcript_and_exact_call_counts(self) -> None:
        episode = _Episode()
        executor = _Executor((_completed("advance"), _completed("finish")))
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary)
            outcome = execute_interactive_episode(
                _policy(episode),
                _rendered(),
                executor,
                workspace=task_root / "workspace",
                timeout_seconds=30,
                max_steps=3,
            )
            replayed_steps = replay_interactive_evidence(
                task_root=task_root,
                outcome=outcome,
            )
        self.assertTrue(outcome.ok)
        self.assertTrue(episode.closed)
        self.assertEqual(outcome.attempted_calls, 2)
        self.assertEqual(outcome.completed_calls, 2)
        self.assertEqual(json.loads(outcome.response)["score"], 2)
        transcript = json.loads(outcome.raw)
        self.assertEqual([row["action"] for row in transcript], ["advance", "finish"])
        runner = outcome.process["interactive_runner"]
        self.assertEqual(runner["kind"], "synthetic_episode")
        self.assertEqual(replayed_steps, 2)

    def test_replay_rejects_self_consistently_rehashed_step_forgery(
        self,
    ) -> None:
        episode = _Episode()
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary)
            outcome = execute_interactive_episode(
                _policy(episode),
                _rendered(),
                _Executor((_completed("finish"),)),
                workspace=task_root / "workspace",
                timeout_seconds=30,
                max_steps=1,
            )
            execution_path = task_root / "workspace" / "steps" / "000" / "EXECUTION.json"
            execution = json.loads(execution_path.read_text())
            execution["response"] = "forged-action"
            atomic_write_json(execution_path, execution)
            process = thaw_json_mapping(outcome.process)
            process["interactive_runner"]["step_evidence"][0]["execution"]["sha256"] = sha256_file(
                execution_path
            )
            forged = replace(outcome, process=process)
            with self.assertRaisesRegex(
                ResultValidationError,
                "transcript does not bind step responses",
            ):
                replay_interactive_evidence(
                    task_root=task_root,
                    outcome=forged,
                )

    def test_action_contract_failure_is_completed_behavior_evidence(self) -> None:
        episode = _Episode()
        with tempfile.TemporaryDirectory() as temporary:
            outcome = execute_interactive_episode(
                _policy(episode),
                _rendered(),
                _Executor((_completed("invalid"),)),
                workspace=Path(temporary) / "workspace",
                timeout_seconds=30,
                max_steps=1,
            )
        self.assertTrue(outcome.ok)
        self.assertTrue(episode.closed)
        self.assertEqual(
            json.loads(outcome.response)["behavior_failure"],
            "output_contract_error:synthetic_action",
        )
        self.assertEqual(json.loads(outcome.raw)[0]["observation"], "start")

    def test_executor_failure_preserves_step_process_and_aggregate_counts(
        self,
    ) -> None:
        episode = _Episode()
        failed = ModelOutcome(
            status="FAILED",
            response="",
            raw="provider evidence",
            process={"provider": {"request_id": "synthetic"}},
            attempted_calls=1,
            completed_calls=0,
            failure_class="provider_error",
            failure="synthetic provider failure",
        )
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary)
            outcome = execute_interactive_episode(
                _policy(episode),
                _rendered(),
                _Executor((_completed("advance"), failed)),
                workspace=task_root / "workspace",
                timeout_seconds=30,
                max_steps=3,
            )
            replayed_steps = replay_interactive_evidence(
                task_root=task_root,
                outcome=outcome,
            )
        self.assertFalse(outcome.ok)
        self.assertTrue(episode.closed)
        self.assertEqual(outcome.attempted_calls, 2)
        self.assertEqual(outcome.completed_calls, 1)
        self.assertEqual(outcome.failure_class, "provider_error")
        self.assertIn("provider", outcome.process)
        self.assertEqual(len(json.loads(outcome.raw)), 1)
        self.assertEqual(replayed_steps, 2)

    def test_terminal_reset_fails_closed_without_a_model_call(self) -> None:
        episode = _Episode(terminal_on_reset=True)
        with tempfile.TemporaryDirectory() as temporary:
            outcome = execute_interactive_episode(
                _policy(episode),
                _rendered(),
                _Executor(()),
                workspace=Path(temporary) / "workspace",
                timeout_seconds=30,
                max_steps=1,
            )
        self.assertFalse(outcome.ok)
        self.assertTrue(episode.closed)
        self.assertEqual(outcome.attempted_calls, 0)
        self.assertEqual(outcome.failure_class, "environment_no_model_call")

    def test_unexpected_parser_exception_is_type_only_harness_evidence(
        self,
    ) -> None:
        secret = "parser-secret-must-not-persist"
        episode = _Episode()

        def parser(response: str) -> str:
            del response
            raise RuntimeError(secret)

        with tempfile.TemporaryDirectory() as temporary:
            outcome = execute_interactive_episode(
                _replace_parser(_policy(episode), parser),
                _rendered(),
                _Executor((_completed("advance"),)),
                workspace=Path(temporary) / "workspace",
                timeout_seconds=30,
                max_steps=1,
            )
        self.assertFalse(outcome.ok)
        self.assertTrue(episode.closed)
        self.assertEqual(outcome.failure_class, "interactive_harness_error")
        self.assertIn("RuntimeError", outcome.failure)
        self.assertNotIn(secret, json.dumps(outcome.public()))

    def test_unexpected_executor_exception_propagates_after_close(self) -> None:
        episode = _Episode()
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(
                RuntimeError,
                "executor boundary failure",
            ),
        ):
            execute_interactive_episode(
                _policy(episode),
                _rendered(),
                _RaisingExecutor(),
                workspace=Path(temporary) / "workspace",
                timeout_seconds=30,
                max_steps=1,
            )
        self.assertTrue(episode.closed)


if __name__ == "__main__":
    unittest.main()
