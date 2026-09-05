"""RethinkSkill providers gemini cli."""

from __future__ import annotations

import json
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
from rethinkskill.utils.serde import atomic_write_json, thaw_json_mapping

_REQUIRED_FLAGS = (
    "--prompt",
    "--model",
    "--output-format",
    "--approval-mode",
    "--extensions",
    "--skip-trust",
)


@dataclass(frozen=True, slots=True)
class GeminiCliExecutor:
    launcher: Path
    model: str
    readiness: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "readiness", freeze_zero_call_readiness(self.readiness))

    @classmethod
    def prepare(cls, config: TransportConfig, *, authorized: bool) -> GeminiCliExecutor:
        if config.kind is not TransportKind.GEMINI_CLI:
            raise ConfigurationError("expected a Gemini CLI transport")
        validate_transport(config)
        if not authorized:
            raise AuthorizationRequiredError("native execution requires --authorize-model-calls")
        launcher = launcher_path(config, "gemini")
        readiness = probe_cli_contract(launcher, provider="gemini", required_flags=_REQUIRED_FLAGS)
        return cls(
            launcher=launcher,
            model=config.model,
            readiness=readiness,
        )

    def public_manifest(self) -> Mapping[str, object]:
        value: dict[str, object] = {
            "kind": "gemini-cli",
            "launcher": str(self.launcher),
            "model": self.model,
            "tool_policy": "disabled_except_task_local_attachment_read",
            "readiness": thaw_json_mapping(self.readiness),
            "environment_policy": provider_environment_policy("gemini-cli"),
            "native_executor": True,
        }
        return value

    def execute(
        self, rendered: RenderedTask, *, workspace: Path, timeout_seconds: int
    ) -> ModelOutcome:
        atomic_write_json(
            workspace / ".gemini/settings.json",
            {
                "tools": {"core": ["read_file"] if rendered.attachments else []},
                "privacy": {"usageStatisticsEnabled": False},
            },
        )
        argv = [
            str(self.launcher),
            "--prompt",
            embedded_prompt(rendered),
            "--model",
            self.model,
            "--output-format",
            "json",
            "--approval-mode",
            "default",
            "--extensions",
            "none",
            "--skip-trust",
        ]
        environment = provider_process_environment("gemini-cli")
        environment["GEMINI_SANDBOX"] = "false"
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
        provider_error = ""
        parse_failure = ""
        if stdout.strip():
            try:
                payload = parse_json_object(stdout, provider="Gemini CLI")
                response = str(payload.get("response") or "")
                error = payload.get("error")
                if error:
                    provider_error = json.dumps(
                        error, ensure_ascii=False, sort_keys=True, allow_nan=False
                    )
            except ConfigurationError as exc:
                parse_failure = str(exc)
        if not parse_failure and (not provider_error) and (not response.strip()):
            parse_failure = "Gemini CLI response has no non-empty response"
        return finalize_cli_outcome(
            provider_label="Gemini CLI",
            result=result,
            argv=argv,
            stdout=stdout,
            stderr=stderr,
            redactions=redactions,
            response=response,
            schema_failure=parse_failure,
            timeout_seconds=timeout_seconds,
            transport_fallback="native Gemini CLI target returned no response",
            transport_failure=provider_error,
        )
