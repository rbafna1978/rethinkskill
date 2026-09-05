"""RethinkSkill providers common."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rethinkskill.errors import ConfigurationError
from rethinkskill.providers.transport import provider_process_environment
from rethinkskill.runtime.tasks import RenderedTask
from rethinkskill.runtime.types import ModelOutcome
from rethinkskill.utils.process import (
    ProcessResult,
    combined_secret_values,
    execute,
    launch_failure_summary,
    redact_text,
    snapshot_launch_failure,
)
from rethinkskill.utils.serde import freeze_json_mapping, strict_json_loads


def _merge_metadata(
    base: dict[str, object], metadata: Mapping[str, object] | None, *, label: str
) -> dict[str, object]:
    if metadata is None:
        return base
    overlap = set(base).intersection(metadata)
    if overlap:
        raise ConfigurationError(f"{label} metadata cannot replace core fields: {sorted(overlap)}")
    return {**base, **metadata}


def finalize_cli_outcome(
    *,
    provider_label: str,
    result: ProcessResult,
    argv: Sequence[str],
    stdout: str,
    stderr: str,
    redactions: Sequence[str],
    response: str,
    schema_failure: str,
    timeout_seconds: int,
    transport_fallback: str,
    transport_failure: str = "",
    raw_metadata: Mapping[str, object] | None = None,
    process_metadata: Mapping[str, object] | None = None,
) -> ModelOutcome:
    """Apply one provider-neutral CLI terminal-state and evidence contract."""
    if type(result) is not ProcessResult or result.launch_failed:
        raise ConfigurationError("CLI outcome requires an exact launched ProcessResult")
    for label, value in (
        ("provider label", provider_label),
        ("stdout", stdout),
        ("stderr", stderr),
        ("response", response),
        ("schema failure", schema_failure),
        ("transport fallback", transport_fallback),
        ("transport failure", transport_failure),
    ):
        if type(value) is not str:
            raise ConfigurationError(f"CLI {label} must be an exact string")
    if not provider_label.strip() or not transport_fallback.strip():
        raise ConfigurationError("CLI provider label and transport fallback must be non-empty")
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ConfigurationError("CLI timeout_seconds must be a positive integer")
    completed = (
        result.returncode == 0
        and (not result.timed_out)
        and bool(response.strip())
        and (not schema_failure)
        and (not transport_failure)
    )
    if completed:
        failure_class = None
        failure = ""
    elif result.timed_out:
        failure_class = "target_timeout"
        failure = f"native {provider_label} target exceeded {timeout_seconds} seconds"
    elif result.returncode != 0 or transport_failure:
        failure_class = "transport_error"
        failure = (transport_failure or stderr.strip() or stdout.strip() or transport_fallback)[
            -4000:
        ]
    else:
        failure_class = "response_schema_error"
        failure = (schema_failure or f"{provider_label} response schema validation failed")[-4000:]
    raw_payload = _merge_metadata(
        {"stdout": stdout, "stderr": stderr}, raw_metadata, label="CLI raw"
    )
    process_payload = _merge_metadata(
        {"argv": list(argv), **result.to_public_dict(redactions=redactions)},
        process_metadata,
        label="CLI process",
    )
    return ModelOutcome(
        status="COMPLETED" if completed else "FAILED",
        response=response,
        raw=json.dumps(raw_payload, ensure_ascii=False, sort_keys=True, allow_nan=False),
        process=process_payload,
        attempted_calls=1,
        completed_calls=1 if completed else 0,
        failure_class=failure_class,
        failure=failure,
    )

def embedded_prompt(rendered: RenderedTask) -> str:
    """Render identical benchmark content for CLIs and direct chat APIs."""
    return f"# Skill\n\n{rendered.skill_markdown}\n\n# Task\n\n{rendered.task_markdown}\n\n# Invocation\n\n{rendered.invocation}\n"


def probe_cli_contract(
    launcher: Path,
    *,
    provider: str,
    required_flags: Sequence[str],
    help_args: Sequence[str] = ("--help",),
) -> dict[str, object]:
    """Check a CLI contract without contacting a model."""
    environment = provider_process_environment(provider)
    result = execute(
        (str(launcher), *help_args), cwd=Path.cwd(), environment=environment, timeout_seconds=30
    )
    text = redact_text(
        f"{result.stdout}\n{result.stderr}", combined_secret_values(environment, os.environ)
    )
    missing = [flag for flag in required_flags if flag not in text]
    if result.returncode != 0 or result.timed_out or missing:
        diagnostic = text.strip()[-2000:] or launch_failure_summary(result)
        raise ConfigurationError(
            f"{provider} CLI readiness probe failed; missing_flags={missing}; diagnostic={diagnostic or 'none'}"
        )
    return {
        "status": f"RETHINKSKILL_{provider.upper()}_CLI_READY",
        "required_flags": list(required_flags),
        "help_args": list(help_args),
        "model_calls": 0,
    }


def freeze_zero_call_readiness(value: Mapping[str, object]) -> Mapping[str, object]:
    """Detach one CLI readiness report and require zero-call provenance."""
    try:
        frozen = freeze_json_mapping(value)
    except Exception as exc:
        raise ConfigurationError(
            f"provider readiness snapshot is invalid: {type(exc).__name__}"
        ) from exc
    declarations = 0
    pending: list[object] = [frozen]
    while pending:
        current = pending.pop()
        if isinstance(current, Mapping):
            for key, nested in current.items():
                if key == "model_calls":
                    declarations += 1
                    if type(nested) is not int or nested != 0:
                        raise ConfigurationError("provider readiness must declare zero model calls")
                pending.append(nested)
        elif isinstance(current, tuple):
            pending.extend(current)
    if declarations == 0:
        raise ConfigurationError("provider readiness must declare zero model calls")
    return frozen


def parse_json_object(value: str, *, provider: str) -> Mapping[str, Any]:
    try:
        parsed = strict_json_loads(value)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{provider} returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ConfigurationError(f"{provider} returned a non-object JSON value")
    return parsed


def sanitized_process_streams(
    result: ProcessResult, environment: Mapping[str, str]
) -> tuple[str, str, tuple[str, ...]]:
    """Redact secret-like environment values before parsing or persistence."""
    redactions = combined_secret_values(environment, os.environ)
    return (
        redact_text(result.stdout, redactions),
        redact_text(result.stderr, redactions),
        redactions,
    )


def launch_failure_outcome(argv: Sequence[str], exc: OSError) -> ModelOutcome:
    """Represent a CLI process that failed before a target call was attempted."""
    failure_type, failure_errno = snapshot_launch_failure(exc)
    failure = f"process launch failed: {failure_type}"
    if failure_errno is not None:
        failure += f"; errno={failure_errno}"
    return ModelOutcome(
        status="FAILED",
        response="",
        raw="",
        process={
            "argv": list(argv),
            "launched": False,
            "returncode": None,
            "timed_out": False,
            "launch_failure": {"type": failure_type, "errno": failure_errno},
        },
        attempted_calls=0,
        completed_calls=0,
        failure_class="transport_error",
        failure=failure,
    )


def launch_failure_result_outcome(result: ProcessResult) -> ModelOutcome:
    """Normalize a sanitized process result whose child never started."""
    if type(result) is not ProcessResult or not result.launch_failed or result.invocation is None:
        raise ConfigurationError("launch failure outcome requires a bound failed ProcessResult")
    failure = f"process launch failed: {launch_failure_summary(result)}"
    return ModelOutcome(
        status="FAILED",
        response="",
        raw="",
        process={"argv": list(result.invocation.argv), **result.to_public_dict()},
        attempted_calls=0,
        completed_calls=0,
        failure_class="transport_error",
        failure=failure,
    )
