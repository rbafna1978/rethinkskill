"""RethinkSkill benchmarks qa."""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import mimetypes
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rethinkskill.benchmarks.scoring import (
    Verification,
    verify_docvqa,
    verify_livemath,
    verify_officeqa,
    verify_searchqa,
)
from rethinkskill.errors import ResultValidationError
from rethinkskill.runtime.harnesses import (
    _CHOICE_LABELS,
    _OFFICEQA_STOPWORDS,
    _STRICT_JSON_MARKER,
    target_skill_markdown,
)
from rethinkskill.runtime.tasks import (
    NativeAsset,
    NativeTask,
    RenderedTask,
    _aliases,
    _dataset_records,
    _first_field,
    _is_dataset_file,
    _load_json_records,
    _select_tasks,
    _split_roots,
    _text,
    _truncate_context,
    load_dataset_text,
    resolve_asset_below_root,
)
from rethinkskill.utils.serde import strict_json_loads


def _docvqa_dataset_files(source: Path, split: str) -> tuple[Path, ...]:
    resolved = source.expanduser().absolute()
    if resolved.is_symlink():
        raise ResultValidationError(f"DocVQA dataset source must not be a symlink: {resolved}")
    if resolved.is_file():
        if resolved.suffix.lower() not in {".csv", ".json", ".jsonl"}:
            raise ResultValidationError(f"DocVQA dataset must be CSV, JSON, or JSONL: {resolved}")
        return (resolved,)
    roots = _split_roots(resolved, split)
    csv_files = tuple(
        sorted(
            path
            for root in roots
            for path in root.iterdir()
            if path.is_file()
            and (not path.name.startswith("."))
            and (path.suffix.lower() == ".csv")
        )
    )
    if csv_files:
        return csv_files
    json_files = tuple(
        sorted(path for root in roots for path in root.iterdir() if _is_dataset_file(path))
    )
    if not json_files:
        raise ResultValidationError(f"DocVQA dataset contains no CSV/JSON records: {resolved}")
    return json_files


def _docvqa_records(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".csv":
        return list(csv.DictReader(io.StringIO(load_dataset_text(path))))
    return _load_json_records(path)


def _docvqa_answers(value: Any) -> tuple[str, ...]:
    if type(value) is list:
        if not all(type(item) is str for item in value):
            raise ResultValidationError("DocVQA answers must contain only text values")
        answers = tuple(item.strip() for item in value if item.strip())
    elif type(value) is str:
        text = value.strip()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = text
        if type(parsed) is list:
            if not all(type(item) is str for item in parsed):
                raise ResultValidationError("DocVQA answers must contain only text values")
            answers = tuple(item.strip() for item in parsed if item.strip())
        elif type(parsed) is str and parsed.strip():
            answers = (parsed.strip(),)
        else:
            answers = ()
    else:
        raise ResultValidationError("DocVQA answers must be text or a text list")
    if not answers:
        raise ResultValidationError("DocVQA answers must contain at least one text value")
    return answers


@dataclass(frozen=True, slots=True)
class DocVQAHarness:
    """Native DocVQA runner with hash-bound multimodal input."""

    def provenance_files(self, source: Path, *, split: str) -> tuple[Path, ...]:
        return _docvqa_dataset_files(source, split)

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
        del seed
        if asset_root is None:
            raise ResultValidationError(
                "DocVQA native execution requires --asset-root pointing to the frozen image directory"
            )
        root = asset_root.expanduser().resolve()
        tasks: list[NativeTask] = []
        for path in _docvqa_dataset_files(source, split):
            for record in _docvqa_records(path):
                raw_id = _first_field(
                    record, ("questionId", "id", "case_id"), field_role="DocVQA id"
                )
                if type(raw_id) is str:
                    task_id = raw_id.strip()
                elif type(raw_id) is int and raw_id >= 0:
                    task_id = str(raw_id)
                else:
                    raise ResultValidationError(
                        "DocVQA id must be non-empty text or a non-negative integer questionId"
                    )
                question_value = record.get("question")
                question = question_value.strip() if type(question_value) is str else ""
                if not task_id or not question:
                    raise ResultValidationError("DocVQA rows require id and question")
                answers = _docvqa_answers(
                    _first_field(
                        record,
                        ("answers", "answer", "ground_truth", "gold_aliases"),
                        field_role="DocVQA answers",
                    )
                )
                raw_image = _first_field(
                    record, ("image_path", "document_path"), field_role="DocVQA image path"
                )
                if type(raw_image) is not str or not raw_image.strip():
                    raise ResultValidationError(
                        f"DocVQA image path must be non-empty text: {task_id}"
                    )
                image_value = raw_image.strip()
                image = resolve_asset_below_root(root, image_value, label="DocVQA image")
                media_type = mimetypes.guess_type(image.name)[0] or "image/png"
                asset = NativeAsset.freeze(
                    image,
                    target=f"attachments/{image.name}",
                    media_type=media_type,
                    role="document_image",
                )
                tasks.append(
                    NativeTask(
                        task_id=task_id,
                        payload={
                            "question": question,
                            "gold_aliases": answers,
                            "strict_json": _STRICT_JSON_MARKER in question,
                        },
                        assets=(asset,),
                    )
                )
        return _select_tasks(tasks, limit=limit, requested_ids=requested_ids)

    def render(self, task: NativeTask, skill: str) -> RenderedTask:
        image = task.assets[0]
        output_contract = (
            'Return exactly one JSON object: `{"answer":"..."}` with no XML tags, extra keys, or prose.'
            if task.payload["strict_json"]
            else "Return the concise final answer inside `<answer>...</answer>`."
        )
        rendered = RenderedTask(
            task_markdown=f"# Question\n\n{task.payload['question']}\n\n# Document image\n\nInspect the task-local image `{image.target}` before answering.\n\n# Output contract\n\n{output_contract}",
            skill_markdown=target_skill_markdown(
                skill,
                description="Dynamic skill for native DocVQA tasks.",
                directive="Inspect the attached document image carefully.",
            ),
            invocation=f"Read `task.md` and `.agents/skills/rethinkskill-target/SKILL.md`, inspect `{image.target}`, and obey the exact output contract.",
            attachments=task.assets,
        )
        rendered.validate()
        return rendered

    def evaluate(self, task: NativeTask, response: str) -> Verification:
        return verify_docvqa(
            response, task.payload["gold_aliases"], strict_json=task.payload["strict_json"]
        )


def _optional_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key, "")
    if type(value) is not str:
        raise ResultValidationError(f"livemath {key} must be text when supplied")
    return value.strip()


def _choice_label(value: Any) -> str:
    if type(value) is not str:
        raise ResultValidationError("livemath choice labels must be text")
    return value.strip().upper().rstrip(".):")


def _livemath_choices(value: Any) -> list[dict[str, str]]:
    if type(value) is dict:
        candidates = [{"label": label, "text": text} for label, text in value.items()]
    elif type(value) is list:
        candidates = [
            dict(item) if type(item) is dict else {"label": _CHOICE_LABELS[index], "text": item}
            for index, item in enumerate(value)
        ]
    else:
        candidates = []
    choices: list[dict[str, str]] = []
    for index, item in enumerate(candidates):
        if index >= len(_CHOICE_LABELS):
            raise ResultValidationError("livemath supports at most seven choices")
        label_value = item.get("label", _CHOICE_LABELS[index])
        label = _choice_label(label_value)
        text_value = item.get("text", item.get("content"))
        text = text_value.strip() if type(text_value) is str else ""
        if not label or not text:
            raise ResultValidationError("livemath choices require non-empty label and text")
        choices.append({"label": label, "text": text})
    if len(choices) < 2:
        raise ResultValidationError("livemath requires at least two choices")
    if len({choice["label"] for choice in choices}) != len(choices):
        raise ResultValidationError("livemath choice labels are duplicated")
    return choices


def _shuffle_livemath_choices(
    choices: list[dict[str, str]], correct_label: str, *, task_id: str, seed: int
) -> tuple[list[dict[str, str]], dict[str, str]]:
    shuffled = [dict(choice) for choice in choices]
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()
    random.Random(int(digest[:16], 16)).shuffle(shuffled)
    remapped: list[dict[str, str]] = []
    correct: dict[str, str] | None = None
    for index, choice in enumerate(shuffled):
        label = _CHOICE_LABELS[index]
        remapped.append({"label": label, "text": choice["text"]})
        if choice["label"] == correct_label:
            correct = {"label": label, "text": choice["text"]}
    if correct is None:
        raise ResultValidationError(
            f"livemath correct label is absent from choices: {correct_label}"
        )
    return (remapped, correct)


@dataclass(frozen=True, slots=True)
class LiveMathHarness:
    """LiveMathematicianBench schema and seeded choice contract."""

    shuffle_choices: bool = True
    use_theorem: bool = False
    use_sketch: bool = False

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
        del asset_root
        records = _dataset_records(source, split)
        tasks: list[NativeTask] = []
        for index, record in enumerate(records):
            raw_mcq = record.get("mcq")
            if raw_mcq is not None and type(raw_mcq) is not dict:
                raise ResultValidationError("livemath mcq must be an object when supplied")
            mcq = raw_mcq or {}
            question_value = mcq["question"] if "question" in mcq else record.get("question")
            question = question_value.strip() if type(question_value) is str else ""
            if not question:
                raise ResultValidationError("livemath question must be non-empty text")
            choices = _livemath_choices(
                mcq["choices"] if "choices" in mcq else record.get("choices")
            )
            raw_correct = (
                mcq["correct_choice"] if "correct_choice" in mcq else record.get("correct_choice")
            )
            if type(raw_correct) is dict:
                correct_label = _choice_label(raw_correct.get("label"))
            else:
                correct_label = _choice_label(raw_correct)
            raw_month = record.get("month", "")
            if type(raw_month) is not str:
                raise ResultValidationError("livemath month must be text when supplied")
            month = raw_month.strip()
            number = record.get("no", index + 1)
            if "id" in record or "case_id" in record:
                raw_id = record["id"] if "id" in record else record["case_id"]
                if type(raw_id) is not str or not raw_id:
                    raise ResultValidationError("livemath id must be non-empty text when supplied")
                task_id = raw_id
            else:
                if type(number) not in {int, str} or (type(number) is str and (not number.strip())):
                    raise ResultValidationError("livemath no must be an integer or non-empty text")
                task_id = f"{month}:{number}" if month else str(number)
            if self.shuffle_choices:
                choices, correct = _shuffle_livemath_choices(
                    choices, correct_label, task_id=task_id, seed=seed
                )
            else:
                correct_text = next(
                    (choice["text"] for choice in choices if choice["label"] == correct_label), ""
                )
                if not correct_text:
                    raise ResultValidationError(
                        f"livemath correct label is absent from choices: {correct_label}"
                    )
                correct = {"label": correct_label, "text": correct_text}
            tasks.append(
                NativeTask(
                    task_id=task_id,
                    payload={
                        "question": question,
                        "choices": tuple(choices),
                        "correct_choice": correct,
                        "theorem": _optional_text(record, "theorem"),
                        "sketch": _optional_text(record, "sketch"),
                        "strict_json": _STRICT_JSON_MARKER in question,
                    },
                )
            )
        return _select_tasks(tasks, limit=limit, requested_ids=requested_ids)

    def render(self, task: NativeTask, skill: str) -> RenderedTask:
        choices = "\n".join(
            f"{choice['label']}. {choice['text']}" for choice in task.payload["choices"]
        )
        sections = [f"# Question\n\n{task.payload['question']}", f"# Choices\n\n{choices}"]
        if self.use_theorem and task.payload["theorem"]:
            sections.append(f"# Theorem\n\n{task.payload['theorem']}")
        if self.use_sketch and task.payload["sketch"]:
            sections.append(f"# Proof sketch\n\n{task.payload['sketch']}")
        output_contract = (
            'Return exactly one JSON object: `{"answer":"A"}` with the selected label, no XML tags, extra keys, or prose.'
            if task.payload["strict_json"]
            else "Return only the selected choice label inside `<answer>...</answer>`."
        )
        sections.append(f"# Output contract\n\n{output_contract}")
        rendered = RenderedTask(
            task_markdown="\n\n".join(sections),
            skill_markdown=target_skill_markdown(
                skill,
                description="Dynamic skill for native LiveMath tasks.",
                directive="Solve the multiple-choice problem in `task.md`.",
            ),
            invocation="Read `task.md` and `.agents/skills/rethinkskill-target/SKILL.md`. Solve the multiple-choice problem and obey the exact output contract.",
        )
        rendered.validate()
        return rendered

    def evaluate(self, task: NativeTask, response: str) -> Verification:
        return verify_livemath(
            response,
            task.payload["choices"],
            task.payload["correct_choice"],
            strict_json=task.payload["strict_json"],
        )


def _text_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if type(value) is list:
        if not all(type(item) is str for item in value):
            raise ResultValidationError("OfficeQA text lists must contain only text values")
        return tuple(item.strip() for item in value if item.strip())
    if type(value) is not str:
        raise ResultValidationError("OfficeQA text list fields must be text or a text list")
    text = value.strip()
    if not text:
        return ()
    try:
        parsed = strict_json_loads(text)
    except json.JSONDecodeError:
        parsed = None
    if type(parsed) is list:
        if not all(type(item) is str for item in parsed):
            raise ResultValidationError("OfficeQA text lists must contain only text values")
        return tuple(item.strip() for item in parsed if item.strip())
    separator = "\n" if "\n" in text else ","
    return tuple(part.strip() for part in text.split(separator) if part.strip())


def _officeqa_dataset_files(source: Path, split: str) -> tuple[Path, ...]:
    resolved = source.expanduser().absolute()
    if resolved.is_symlink():
        raise ResultValidationError(f"OfficeQA dataset source must not be a symlink: {resolved}")
    if resolved.is_file():
        if resolved.suffix.lower() not in {".csv", ".json", ".jsonl"}:
            raise ResultValidationError(f"OfficeQA dataset must be CSV, JSON, or JSONL: {resolved}")
        return (resolved,)
    files = tuple(
        sorted(
            path
            for root in _split_roots(resolved, split)
            for path in root.iterdir()
            if path.is_file()
            and (not path.name.startswith("."))
            and (path.suffix.lower() in {".csv", ".json", ".jsonl"})
        )
    )
    if not files:
        raise ResultValidationError(f"OfficeQA dataset contains no CSV/JSON records: {resolved}")
    return files


def _officeqa_records(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".csv":
        return list(csv.DictReader(io.StringIO(load_dataset_text(path))))
    return _load_json_records(path)


def _officeqa_terms(question: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall("[a-z0-9.%$-]+", question.lower())
        if token not in _OFFICEQA_STOPWORDS
        and (len(token) > 2 or any(character.isdigit() for character in token))
    )


def _officeqa_evidence(
    question: str,
    assets: Sequence[NativeAsset],
    payloads: Mapping[str, bytes],
    *,
    chunk_chars: int = 2800,
    overlap_chars: int = 400,
    max_chunks: int = 8,
    max_chars: int = 24000,
) -> str:
    terms = _officeqa_terms(question)
    candidates: list[tuple[int, str, int, str]] = []
    step = chunk_chars - overlap_chars
    for asset in assets:
        try:
            text = payloads[asset.target].decode("utf-8", errors="replace")
        except KeyError as exc:
            raise ResultValidationError(
                f"OfficeQA evidence snapshot is missing: {asset.target}"
            ) from exc
        for offset in range(0, max(len(text), 1), step):
            chunk = text[offset : offset + chunk_chars].strip()
            if not chunk:
                continue
            lowered = chunk.lower()
            score = sum(
                lowered.count(term) * (3 if any(character.isdigit() for character in term) else 1)
                for term in terms
            )
            candidates.append((score, Path(asset.target).name, offset, chunk))
    if not candidates:
        raise ResultValidationError("OfficeQA source documents contain no readable text")
    ranked = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))
    selected = ranked[:max_chunks]
    sections: list[str] = []
    used = 0
    for score, name, offset, chunk in selected:
        section = f"## {name} @ character {offset} (retrieval score {score})\n\n{chunk}"
        if used + len(section) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                sections.append(section[:remaining])
            break
        sections.append(section)
        used += len(section)
    return "\n\n".join(sections)


@dataclass(frozen=True, slots=True)
class OfficeQAHarness:
    """Portable OfficeQA runner over explicit frozen local documents."""

    def provenance_files(self, source: Path, *, split: str) -> tuple[Path, ...]:
        return _officeqa_dataset_files(source, split)

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
        del seed
        if asset_root is None:
            raise ResultValidationError(
                "OfficeQA native execution requires --asset-root pointing to the frozen document directory"
            )
        root = asset_root.expanduser().resolve()
        tasks: list[NativeTask] = []
        for path in _officeqa_dataset_files(source, split):
            for record in _officeqa_records(path):
                raw_id = _first_field(record, ("uid", "id", "case_id"), field_role="OfficeQA id")
                question_value = record.get("question")
                gold_value = _first_field(
                    record, ("ground_truth", "answer", "gold_answer"), field_role="OfficeQA answer"
                )
                task_id = raw_id.strip() if type(raw_id) is str else ""
                question = question_value.strip() if type(question_value) is str else ""
                gold = gold_value.strip() if type(gold_value) is str else ""
                if not task_id or not question or (not gold):
                    raise ResultValidationError("OfficeQA rows require id, question, and answer")
                source_files = _text_list(record.get("source_files"))
                if not source_files:
                    raise ResultValidationError(f"OfficeQA row has no source_files: {task_id}")
                frozen_assets = tuple(
                    NativeAsset.freeze_with_snapshot(
                        resolve_asset_below_root(root, source_file, label="OfficeQA source file"),
                        target=f"documents/{Path(source_file).name}",
                        media_type="text/plain",
                        role="evidence_document",
                    )
                    for source_file in source_files
                )
                assets = tuple((asset for asset, _ in frozen_assets))
                evidence = _officeqa_evidence(
                    question,
                    assets,
                    {asset.target: snapshot.payload for asset, snapshot in frozen_assets},
                )
                tasks.append(
                    NativeTask(
                        task_id=task_id,
                        payload={
                            "question": question,
                            "gold_answer": gold,
                            "evidence": evidence,
                            "source_docs": _text_list(record.get("source_docs")),
                            "strict_json": _STRICT_JSON_MARKER in question,
                        },
                        assets=assets,
                    )
                )
        return _select_tasks(tasks, limit=limit, requested_ids=requested_ids)

    def render(self, task: NativeTask, skill: str) -> RenderedTask:
        output_contract = (
            'Return exactly one JSON object: `{"answer":"..."}` with no XML tags, extra keys, or prose.'
            if task.payload["strict_json"]
            else "Return the concise final answer inside `<answer>...</answer>`."
        )
        rendered = RenderedTask(
            task_markdown=f"# Question\n\n{task.payload['question']}\n\n# Deterministically retrieved evidence\n\n{task.payload['evidence']}\n\n# Source files\n\n"
            + "\n".join(f"- `{asset.target}`" for asset in task.assets)
            + f"\n\n# Output contract\n\n{output_contract}",
            skill_markdown=target_skill_markdown(
                skill,
                description="Dynamic skill for native OfficeQA tasks.",
                directive="Answer the question from the frozen local evidence.",
            ),
            invocation="Read `task.md` and `.agents/skills/rethinkskill-target/SKILL.md`. Answer only from the frozen evidence and obey the exact output contract.",
            attachments=task.assets,
        )
        rendered.validate()
        return rendered

    def evaluate(self, task: NativeTask, response: str) -> Verification:
        return verify_officeqa(
            response, task.payload["gold_answer"], strict_json=task.payload["strict_json"]
        )


class SearchQAHarness:
    """Context-grounded SearchQA schema and deterministic verifier binding."""

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
        tasks: list[NativeTask] = []
        for record in records:
            task_id = _text(record, "id" if "id" in record else "case_id")
            task = NativeTask(
                task_id=task_id,
                payload={
                    "question": _text(record, "question"),
                    "context": _text(record, "context", default=""),
                    "gold_aliases": _aliases(record),
                },
            )
            tasks.append(task)
        return _select_tasks(tasks, limit=limit, requested_ids=requested_ids)

    def render(self, task: NativeTask, skill: str) -> RenderedTask:
        context = _truncate_context(task.payload["context"])
        task_markdown = f"# Context\n\n{context}\n\n# Question\n\n{task.payload['question']}\n\n# Output contract\n\nReturn the concise final answer inside `<answer>...</answer>`."
        skill_markdown = target_skill_markdown(
            skill,
            description="Dynamic skill for the current native QA task.",
            directive="Ground the answer in `task.md` and follow its output contract.",
        )
        rendered = RenderedTask(
            task_markdown=task_markdown,
            skill_markdown=skill_markdown,
            invocation="Read `task.md` and `.agents/skills/rethinkskill-target/SKILL.md`. Answer the question and preserve the required answer tags.",
        )
        rendered.validate()
        return rendered

    def evaluate(self, task: NativeTask, response: str) -> Verification:
        return verify_searchqa(response, task.payload["gold_aliases"])
