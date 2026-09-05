"""RethinkSkill providers codex."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rethinkskill.errors import AuthorizationRequiredError, ConfigurationError
from rethinkskill.providers.common import (
    finalize_cli_outcome,
    freeze_zero_call_readiness,
    launch_failure_outcome,
    launch_failure_result_outcome,
    probe_cli_contract,
    sanitized_process_streams,
)
from rethinkskill.providers.transport import (
    TransportConfig,
    TransportKind,
    launcher_path,
    provider_environment_policy,
    provider_process_environment,
    validate_transport,
)
from rethinkskill.runtime.tasks import RenderedTask
from rethinkskill.runtime.types import ModelOutcome
from rethinkskill.utils.integrity import safe_exception_type
from rethinkskill.utils.process import execute, redact_text
from rethinkskill.utils.serde import strict_json_loads, thaw_json_mapping

_DISABLED_FEATURES = (
    "apps",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "computer_use",
    "image_generation",
    "plugins",
    "standalone_web_search",
)

_REQUIRED_FLAGS = (
    "--config",
    "--disable",
    "--strict-config",
    "--model",
    "--sandbox",
    "--cd",
    "--skip-git-repo-check",
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "--color",
    "--json",
    "--output-last-message",
)


def audit_codex_jsonl(value: str) -> dict[str, object]:
    event_types: list[str] = []
    completed_turns = 0
    usage: Mapping[str, Any] | None = None
    parse_errors: list[str] = []
    for line_number, line in enumerate(value.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = strict_json_loads(line)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"line {line_number}: {exc}")
            continue
        if not isinstance(event, dict):
            parse_errors.append(f"line {line_number}: expected object")
            continue
        event_type = str(event.get("type") or "")
        event_types.append(event_type)
        if event_type == "turn.completed":
            completed_turns += 1
            observed_usage = event.get("usage")
            if isinstance(observed_usage, dict):
                usage = observed_usage
    return {
        "event_types": event_types,
        "completed_turns": completed_turns,
        "usage": dict(usage or {}),
        "parse_errors": parse_errors,
    }


@dataclass(frozen=True, slots=True)
class CodexCliExecutor:
    launcher: Path
    model: str
    reasoning_effort: str
    sandbox: str
    readiness: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "readiness", freeze_zero_call_readiness(self.readiness))

    @classmethod
    def prepare(cls, config: TransportConfig, *, authorized: bool) -> CodexCliExecutor:
        if config.kind is not TransportKind.CODEX:
            raise ConfigurationError("expected a Codex transport")
        validate_transport(config)
        if not authorized:
            raise AuthorizationRequiredError("native execution requires --authorize-model-calls")
        launcher = launcher_path(config, "codex")
        readiness = probe_cli_contract(
            launcher, provider="codex", required_flags=_REQUIRED_FLAGS, help_args=("exec", "--help")
        )
        return cls(
            launcher=launcher,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            sandbox=config.sandbox,
            readiness=readiness,
        )

    def public_manifest(self) -> Mapping[str, object]:
        value: dict[str, object] = {
            "kind": "codex",
            "launcher": str(self.launcher),
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "sandbox": self.sandbox,
            "readiness": thaw_json_mapping(self.readiness),
            "environment_policy": provider_environment_policy("codex"),
            "native_executor": True,
        }
        return value

    def execute(
        self, rendered: RenderedTask, *, workspace: Path, timeout_seconds: int
    ) -> ModelOutcome:
        last_message = workspace / "codex_last_message.txt"
        argv = [
            str(self.launcher),
            "exec",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-C",
            str(workspace),
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--strict-config",
            "--json",
            "-s",
            self.sandbox,
            "-c",
            'approval_policy="never"',
            "-c",
            'web_search="disabled"',
            "-c",
            'shell_environment_policy={inherit="none",set={PATH="/usr/local/bin:/usr/bin:/bin"}}',
        ]
        for feature in _DISABLED_FEATURES:
            argv.extend(("--disable", feature))
        if self.reasoning_effort:
            argv.extend(("-c", f'model_reasoning_effort="{self.reasoning_effort}"'))
        for attachment in rendered.attachments:
            if attachment.media_type.startswith("image/"):
                argv.extend(("-i", str(workspace / attachment.target)))
        argv.extend(
            ("-m", self.model, "--output-last-message", str(last_message), rendered.invocation)
        )
        environment = provider_process_environment("codex")
        try:
            result = execute(
                argv, cwd=workspace, environment=environment, timeout_seconds=timeout_seconds
            )
        except OSError as exc:
            return launch_failure_outcome(argv, exc)
        if result.launch_failed:
            return launch_failure_result_outcome(result)
        stdout, stderr, redactions = sanitized_process_streams(result, environment)
        last_message_failure = ""
        try:
            raw_response = (
                redact_text(last_message.read_text(encoding="utf-8"), redactions)
                if last_message.is_file()
                else ""
            )
        except OSError as exc:
            raw_response = ""
            last_message_failure = safe_exception_type(exc)
        audit = audit_codex_jsonl(stdout)
        schema_failure = ""
        if audit["parse_errors"]:
            schema_failure = f"Codex JSONL event stream is malformed: {audit['parse_errors']}"
        elif audit["completed_turns"] != 1:
            schema_failure = f"Codex JSONL event stream requires exactly one turn.completed event; observed={audit['completed_turns']}"
        elif last_message_failure:
            schema_failure = f"Codex last-message artifact is unreadable: {last_message_failure}"
        elif not raw_response.strip():
            schema_failure = "Codex returned no non-empty last message"
        return finalize_cli_outcome(
            provider_label="Codex",
            result=result,
            argv=argv,
            stdout=stdout,
            stderr=stderr,
            redactions=redactions,
            response=raw_response,
            schema_failure=schema_failure,
            timeout_seconds=timeout_seconds,
            transport_fallback="native Codex target returned no completed response",
            raw_metadata={"event_audit": audit},
            process_metadata={"event_audit": audit},
        )
