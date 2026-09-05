"""RethinkSkill providers claude code."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rethinkskill.errors import AuthorizationRequiredError, ConfigurationError
from rethinkskill.providers.common import (
    embedded_prompt,
    finalize_cli_outcome,
    freeze_zero_call_readiness,
    launch_failure_outcome,
    launch_failure_result_outcome,
    parse_json_object,
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
from rethinkskill.utils.process import execute
from rethinkskill.utils.serde import thaw_json_mapping

_REQUIRED_FLAGS = (
    "--print",
    "--output-format",
    "--model",
    "--permission-mode",
    "--tools",
    "--strict-mcp-config",
    "--disable-slash-commands",
    "--no-chrome",
    "--setting-sources",
    "--no-session-persistence",
    "--effort",
)


@dataclass(frozen=True, slots=True)
class ClaudeCodeCliExecutor:
    launcher: Path
    model: str
    effort: str
    readiness: Mapping[str, object]
    base_url: str | None = None
    auth_token_env: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "readiness", freeze_zero_call_readiness(self.readiness))

    @classmethod
    def prepare(cls, config: TransportConfig, *, authorized: bool) -> ClaudeCodeCliExecutor:
        if config.kind is not TransportKind.CLAUDE_CODE:
            raise ConfigurationError("expected a Claude Code transport")
        validate_transport(config)
        if not authorized:
            raise AuthorizationRequiredError("native execution requires --authorize-model-calls")
        if config.claude_auth_token_env and (not os.environ.get(config.claude_auth_token_env, "")):
            raise AuthorizationRequiredError(
                f"Claude credential environment variable is empty: {config.claude_auth_token_env}"
            )
        launcher = launcher_path(config, "claude")
        readiness = probe_cli_contract(
            launcher, provider="claude_code", required_flags=_REQUIRED_FLAGS
        )
        return cls(
            launcher=launcher,
            model=config.model,
            effort=config.claude_effort,
            readiness=readiness,
            base_url=config.claude_base_url,
            auth_token_env=config.claude_auth_token_env,
        )

    def public_manifest(self) -> Mapping[str, object]:
        value: dict[str, object] = {
            "kind": "claude-code",
            "launcher": str(self.launcher),
            "model": self.model,
            "effort": self.effort,
            "tool_policy": "disabled_except_task_local_attachment_read",
            "readiness": thaw_json_mapping(self.readiness),
            "environment_policy": provider_environment_policy("claude-code"),
            "native_executor": True,
        }
        if self.base_url:
            value["gateway"] = {
                "base_url": self.base_url.rstrip("/"),
                "auth_token_env": self.auth_token_env,
            }
        return value

    def execute(
        self, rendered: RenderedTask, *, workspace: Path, timeout_seconds: int
    ) -> ModelOutcome:
        attachment_tools = "Read" if rendered.attachments else ""
        argv = [
            str(self.launcher),
            "--print",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--permission-mode",
            "dontAsk",
            "--tools",
            attachment_tools,
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--no-chrome",
            embedded_prompt(rendered),
        ]
        environment = provider_process_environment("claude-code")
        environment.pop("ANTHROPIC_BASE_URL", None)
        environment.pop("ANTHROPIC_AUTH_TOKEN", None)
        if self.base_url:
            assert self.auth_token_env is not None
            auth_token = os.environ.get(self.auth_token_env, "")
            if not auth_token:
                return ModelOutcome(
                    status="FAILED",
                    response="",
                    raw="",
                    process={"argv": argv, "returncode": None, "timed_out": False},
                    attempted_calls=0,
                    completed_calls=0,
                    failure_class="authorization_error",
                    failure=f"Claude credential environment variable became empty: {self.auth_token_env}",
                )
            environment.pop(self.auth_token_env, None)
            environment["ANTHROPIC_BASE_URL"] = self.base_url.rstrip("/")
            environment["ANTHROPIC_AUTH_TOKEN"] = auth_token
            environment["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
        environment["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
        try:
            result = execute(
                argv, cwd=workspace, environment=environment, timeout_seconds=timeout_seconds
            )
        except OSError as exc:
            return launch_failure_outcome(argv, exc)
        if result.launch_failed:
            return launch_failure_result_outcome(result)
        stdout, stderr, redactions = sanitized_process_streams(result, environment)
        response = ""
        parse_failure = ""
        if stdout.strip():
            try:
                payload = parse_json_object(stdout, provider="Claude Code")
                response = str(payload.get("result") or "")
            except ConfigurationError as exc:
                parse_failure = str(exc)
        if not parse_failure and (not response.strip()):
            parse_failure = "Claude Code response has no non-empty result"
        return finalize_cli_outcome(
            provider_label="Claude Code",
            result=result,
            argv=argv,
            stdout=stdout,
            stderr=stderr,
            redactions=redactions,
            response=response,
            schema_failure=parse_failure,
            timeout_seconds=timeout_seconds,
            transport_fallback="native Claude Code target returned no response",
        )
