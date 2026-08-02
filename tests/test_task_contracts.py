from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rethinkskill.errors import ConfigurationError, ResultValidationError
from rethinkskill.runtime.runner import freeze_native_tasks
from rethinkskill.runtime.tasks import (
    NativeAsset,
    NativeTask,
    RenderedTask,
    materialize_rendered_workspace,
    select_tasks,
)
from rethinkskill.utils.serde import thaw_json_mapping


class NativeTaskContractTests(unittest.TestCase):
    @staticmethod
    def _asset(
        root: Path,
        name: str,
        target: str,
        *,
        visibility: str = "workspace",
    ) -> NativeAsset:
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(name, encoding="utf-8")
        return NativeAsset.freeze(
            source,
            target=target,
            media_type="text/plain",
            role="fixture",
            visibility=visibility,
        )

    def test_workspace_assets_cannot_replace_executor_control_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            targets = (
                "task.md",
                "TASK.MD",
                ".agents/skills/other/SKILL.md",
                ".claude/settings.json",
                ".codex/config.toml",
                ".gemini/skills/other/SKILL.md",
                ".git/config",
                ".rethinkskill/RUN_MANIFEST.json",
                "steps/000/workspace/task.md",
                r"nested\escape.txt",
            )
            for index, target in enumerate(targets):
                with self.subTest(target=target), self.assertRaises(ResultValidationError):
                    self._asset(root, f"source-{index}", target)

    def test_asset_freeze_returns_matching_bytes_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.txt"
            source.write_bytes(b"frozen asset\n")
            asset, snapshot = NativeAsset.freeze_with_snapshot(
                source,
                target="input.txt",
                media_type="text/plain",
                role="fixture",
            )
            self.assertEqual(snapshot.payload, b"frozen asset\n")
            self.assertEqual(asset.sha256, snapshot.sha256)

            link = root / "source-link.txt"
            link.symlink_to(source)
            with self.assertRaisesRegex(ResultValidationError, "non-symlink"):
                NativeAsset.freeze(
                    link,
                    target="linked.txt",
                    media_type="text/plain",
                    role="fixture",
                )

    def test_asset_targets_are_portable_and_prefix_collision_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_collision = NativeTask(
                task_id="case",
                payload={},
                assets=(
                    self._asset(root, "one", "Data/File.txt"),
                    self._asset(root, "two", "data/file.TXT"),
                ),
            )
            prefix_collision = NativeTask(
                task_id="prefix",
                payload={},
                assets=(
                    self._asset(root, "three", "data"),
                    self._asset(root, "four", "data/file.txt"),
                ),
            )
            for task in (case_collision, prefix_collision):
                with self.subTest(task=task.task_id), self.assertRaises(ResultValidationError):
                    task.validate()

    def test_same_target_is_allowed_across_visibility_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = NativeTask(
                task_id="separate",
                payload={},
                assets=(
                    self._asset(root, "one", "data/file.txt"),
                    self._asset(
                        root,
                        "two",
                        "data/file.txt",
                        visibility="verifier",
                    ),
                ),
            )
            self.assertIsNone(task.validate())

    def test_task_rejects_native_asset_subclasses(self) -> None:
        class OverridingAsset(NativeAsset):
            def validate(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "missing.txt"
            task = NativeTask(
                task_id="case",
                payload={},
                assets=(
                    OverridingAsset(
                        source=source,
                        target="asset.txt",
                        media_type="text/plain",
                        role="fixture",
                        sha256="forged",
                    ),
                ),
            )
            with self.assertRaisesRegex(
                ResultValidationError,
                "exact NativeAsset values",
            ):
                task.validate()

    def test_harness_boundary_rejects_native_task_subclasses(self) -> None:
        class OverridingTask(NativeTask):
            def validate(self):
                return None

        tasks = (OverridingTask(task_id="case", payload={}),)
        with self.assertRaisesRegex(
            ConfigurationError,
            "exact NativeTask",
        ):
            freeze_native_tasks(tasks)
        with self.assertRaisesRegex(
            ResultValidationError,
            "exact NativeTask",
        ):
            select_tasks(tasks, limit=None, requested_ids=())

    def test_harness_payload_is_deeply_detached_and_read_only(self) -> None:
        source = {
            "nested": {"score": 1},
            "items": ["one"],
        }
        frozen = freeze_native_tasks((NativeTask(task_id="case", payload=source),))[0]
        source["nested"]["score"] = 0
        source["items"].append("two")
        self.assertEqual(
            thaw_json_mapping(frozen.payload),
            {
                "nested": {"score": 1},
                "items": ["one"],
            },
        )
        with self.assertRaises(TypeError):
            frozen.payload["new"] = "value"  # type: ignore[index]
        nested = frozen.payload["nested"]
        with self.assertRaises(TypeError):
            nested["score"] = 2  # type: ignore[index]

    def test_workspace_rejects_rendered_task_subclasses(self) -> None:
        class OverridingRenderedTask(RenderedTask):
            def validate(self):
                return None

        rendered = OverridingRenderedTask(
            task_markdown="task",
            skill_markdown="skill",
            invocation="invoke",
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(
                ResultValidationError,
                "exact RenderedTask",
            ),
        ):
            materialize_rendered_workspace(
                rendered,
                Path(temporary) / "workspace",
                skill_name="target",
            )

    def test_payload_must_be_finite_canonical_json(self) -> None:
        invalid = (
            NativeTask(task_id="nan", payload={"value": float("nan")}),
            NativeTask(task_id="opaque", payload={"value": object()}),
        )
        for task in invalid:
            with (
                self.subTest(task=task.task_id),
                self.assertRaisesRegex(
                    ResultValidationError,
                    "finite canonical JSON",
                ),
            ):
                task.validate()

    def test_long_or_unsafe_ids_have_bounded_deterministic_workspaces(self) -> None:
        long_task = NativeTask(task_id="a" * 121, payload={})
        unsafe_one = NativeTask(task_id="task/one", payload={})
        unsafe_two = NativeTask(task_id="task/one?", payload={})

        self.assertLessEqual(len(long_task.workspace_name.encode("utf-8")), 65)
        self.assertEqual(long_task.workspace_name, long_task.workspace_name)
        self.assertNotEqual(unsafe_one.workspace_name, unsafe_two.workspace_name)
        for task in (long_task, unsafe_one, unsafe_two):
            self.assertNotIn("/", task.workspace_name)

    def test_selection_rejects_workspace_and_requested_id_collisions(self) -> None:
        tasks = (
            NativeTask(task_id="Case", payload={}),
            NativeTask(task_id="case", payload={}),
        )
        with self.assertRaisesRegex(
            ResultValidationError,
            "workspace collision",
        ):
            select_tasks(tasks, limit=None, requested_ids=())

        one = (NativeTask(task_id="one", payload={}),)
        with self.assertRaisesRegex(
            ResultValidationError,
            "duplicate requested",
        ):
            select_tasks(
                one,
                limit=None,
                requested_ids=("one", "one"),
            )

    def test_selection_rejects_invalid_or_ambiguous_limit(self) -> None:
        tasks = (NativeTask(task_id="one", payload={}),)
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ConfigurationError):
                select_tasks(
                    tasks,
                    limit=limit,  # type: ignore[arg-type]
                    requested_ids=(),
                )
        with self.assertRaisesRegex(
            ConfigurationError,
            "mutually exclusive",
        ):
            select_tasks(
                tasks,
                limit=1,
                requested_ids=("one",),
            )

    def test_rendered_attachments_reject_prefix_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rendered = RenderedTask(
                task_markdown="task",
                skill_markdown="skill",
                invocation="invoke",
                attachments=(
                    self._asset(root, "one", "data"),
                    self._asset(root, "two", "data/file.txt"),
                ),
            )
            with self.assertRaisesRegex(
                ResultValidationError,
                "colliding rendered attachment",
            ):
                rendered.validate()

    def test_rendered_attachments_must_belong_to_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = self._asset(root, "frozen", "data/frozen.txt")
            foreign = self._asset(root, "foreign", "data/foreign.txt")
            task = NativeTask(
                task_id="one",
                payload={},
                assets=(frozen,),
            )
            rendered = RenderedTask(
                task_markdown="task",
                skill_markdown="skill",
                invocation="invoke",
                attachments=(foreign,),
            )
            with self.assertRaisesRegex(
                ResultValidationError,
                "not a frozen workspace asset",
            ):
                rendered.validate_for(task)


if __name__ == "__main__":
    unittest.main()
