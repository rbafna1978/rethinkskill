"""RethinkSkill runtime harnesses."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rethinkskill.benchmarks.core import CaseAdapter
from rethinkskill.benchmarks.scoring import Verification
from rethinkskill.errors import ResultValidationError
from rethinkskill.runtime.tasks import (
    NativeTask,
    RenderedTask,
    _dataset_records,
    _first_field,
    _select_tasks,
    _text,
)

_CHOICE_LABELS = tuple("ABCDEFG")

_STRICT_JSON_MARKER = 'Return exactly one JSON object with schema {"answer":"<answer>"}. Do not use XML tags or add other keys or prose.'

_OFFICEQA_STOPWORDS = frozenset(
    {
        "a",
        "all",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "reported",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)

_EMPTY_GUIDANCE = "No additional dynamic guidance was provided."


def target_skill_markdown(skill: str, *, description: str, directive: str) -> str:
    """Render the canonical task-local target skill wrapper."""
    for label, value in (("skill", skill), ("description", description), ("directive", directive)):
        if type(value) is not str:
            raise ResultValidationError(f"target skill {label} must be an exact string")
    if not description.strip() or not directive.strip():
        raise ResultValidationError("target skill description and directive must be non-empty")
    guidance = skill.strip() or _EMPTY_GUIDANCE
    return f'---\nname: "rethinkskill-target"\ndescription: "{description}"\n---\n\n# RethinkSkill Target Guidance\n\n{directive}\n\n## Dynamic guidance\n\n{guidance}\n'


@dataclass(frozen=True, slots=True)
class DeterministicCaseHarness:
    """File-backed tasks evaluated by an existing deterministic case adapter."""

    benchmark: str
    task_kind: str
    input_fields: tuple[str, ...]
    gold_field: str
    gold_aliases: tuple[str, ...]
    adapter: CaseAdapter

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
        records = _dataset_records(source, split)
        tasks = []
        for record in records:
            task_id = _text(record, "id" if "id" in record else "case_id")
            input_value = _first_field(record, self.input_fields, field_role="task input")
            if type(input_value) is not str or not input_value.strip():
                raise ResultValidationError(
                    "native classification task input must be non-empty text"
                )
            gold = _first_field(
                record, (self.gold_field, *self.gold_aliases), field_role=self.gold_field
            )
            if type(gold) is list:
                if not gold or not all(type(item) is str and item.strip() for item in gold):
                    raise ResultValidationError(
                        f"{self.gold_field} must be text or a non-empty text list"
                    )
            elif type(gold) is not str or not gold.strip():
                raise ResultValidationError(
                    f"{self.gold_field} must be text or a non-empty text list"
                )
            label_space = record.get("label_space", record.get("allowed_labels"))
            if label_space is not None and (
                type(label_space) is not list
                or not label_space
                or (not all(type(item) is str and item.strip() for item in label_space))
            ):
                raise ResultValidationError(
                    "optional label_space/allowed_labels must be a non-empty text list"
                )
            tasks.append(
                NativeTask(
                    task_id=task_id,
                    payload={
                        "input": input_value,
                        self.gold_field: gold,
                        "label_space": tuple(label_space or ()),
                    },
                )
            )
        return _select_tasks(tasks, limit=limit, requested_ids=requested_ids)

    def render(self, task: NativeTask, skill: str) -> RenderedTask:
        labels = task.payload.get("label_space", ())
        labels_section = (
            "\n\n# Allowed labels\n\n" + "\n".join(f"- {label}" for label in labels)
            if labels
            else ""
        )
        output_contract = {
            "single_label": "Return exactly one label inside `<answer>...</answer>`.",
            "multi_label": "Return all applicable labels separated by semicolons inside `<answer>...</answer>`.",
            "structured_exact_match": "Return only the requested structured value inside `<answer>...</answer>`; preserve exact syntax.",
        }.get(self.task_kind, "Return the concise final answer inside `<answer>...</answer>`.")
        task_markdown = f"# Task type\n\n{self.task_kind}\n\n# Input\n\n{task.payload['input']}{labels_section}\n\n# Output contract\n\n{output_contract}"
        skill_markdown = target_skill_markdown(
            skill,
            description=f"Dynamic skill for native {self.benchmark} tasks.",
            directive="Solve the task in `task.md` and follow its output contract.",
        )
        rendered = RenderedTask(
            task_markdown=task_markdown,
            skill_markdown=skill_markdown,
            invocation="Read `task.md` and `.agents/skills/rethinkskill-target/SKILL.md`. Solve the task and preserve the required answer tags.",
        )
        rendered.validate()
        return rendered

    def evaluate(self, task: NativeTask, response: str) -> Verification:
        row = {
            "case_id": task.task_id,
            "frozen_response": response,
            self.gold_field: task.payload[self.gold_field],
        }
        return self.adapter(row, Path("."))
