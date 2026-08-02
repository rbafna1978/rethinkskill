"""RethinkSkill providers openai compatible."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rethinkskill.errors import AuthorizationRequiredError, ConfigurationError
from rethinkskill.providers.common import embedded_prompt
from rethinkskill.providers.transport import TransportConfig, TransportKind, validate_transport
from rethinkskill.runtime.tasks import RenderedTask
from rethinkskill.runtime.types import ModelOutcome
from rethinkskill.utils.integrity import safe_exception_type
from rethinkskill.utils.process import combined_secret_values, redact_text
from rethinkskill.utils.serde import strict_json_loads


def chat_completions_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def _message_content(rendered: RenderedTask, workspace: Path) -> str | list[dict[str, object]]:
    prompt = embedded_prompt(rendered)
    images = [
        attachment
        for attachment in rendered.attachments
        if attachment.media_type.startswith("image/")
    ]
    if not images:
        return prompt
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for attachment in images:
        path = workspace / attachment.target
        if not path.is_file():
            raise ValueError(f"materialized image attachment is absent: {path}")
        media_type = attachment.media_type or mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}}
        )
    return content


@dataclass(frozen=True, slots=True)
class OpenAICompatibleExecutor:
    model: str
    base_url: str
    api_key_env: str
    reasoning_effort: str

    @classmethod
    def prepare(cls, config: TransportConfig, *, authorized: bool) -> OpenAICompatibleExecutor:
        if config.kind is not TransportKind.OPENAI_COMPATIBLE:
            raise ConfigurationError("expected an OpenAI-compatible transport")
        validate_transport(config)
        if not authorized:
            raise AuthorizationRequiredError("native execution requires --authorize-model-calls")
        assert config.api_key_env is not None
        if not os.environ.get(config.api_key_env):
            raise AuthorizationRequiredError(
                f"API credential environment variable is empty: {config.api_key_env}"
            )
        return cls(
            model=config.model,
            base_url=str(config.api_base_url).rstrip("/"),
            api_key_env=config.api_key_env,
            reasoning_effort=config.reasoning_effort,
        )

    def public_manifest(self) -> Mapping[str, object]:
        return {
            "kind": "openai-compatible",
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "reasoning_effort": self.reasoning_effort,
            "native_executor": True,
        }

    def execute(
        self, rendered: RenderedTask, *, workspace: Path, timeout_seconds: int
    ) -> ModelOutcome:
        endpoint = chat_completions_url(self.base_url)
        key = os.environ.get(self.api_key_env, "")
        if not key:
            return ModelOutcome(
                status="FAILED",
                response="",
                raw="",
                process={"endpoint": endpoint, "credential_env": self.api_key_env},
                attempted_calls=0,
                completed_calls=0,
                failure_class="authorization_error",
                failure=f"API credential environment variable became empty: {self.api_key_env}",
            )
        redactions = tuple(
            sorted({key, *combined_secret_values(os.environ)}, key=len, reverse=True)
        )
        try:
            content = _message_content(rendered, workspace)
        except (OSError, ValueError) as exc:
            return ModelOutcome(
                status="FAILED",
                response="",
                raw="",
                process={"endpoint": endpoint, "credential_env": self.api_key_env},
                attempted_calls=0,
                completed_calls=0,
                failure_class="missing_artifact",
                failure=f"OpenAI-compatible input artifact is unreadable: {safe_exception_type(exc)}",
            )
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "rethinkskill/1.0",
            },
            method="POST",
        )
        status_code = None
        raw = ""
        response = ""
        failure_class = None
        failure = ""
        try:
            with urlopen(request, timeout=timeout_seconds) as handle:
                status_code = int(handle.status)
                raw = redact_text(handle.read().decode("utf-8", errors="replace"), redactions)
            parsed = strict_json_loads(raw)
            choices = parsed.get("choices") if isinstance(parsed, dict) else None
            message = (
                choices[0].get("message")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict)
                else None
            )
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise ValueError("response has no choices[0].message.content")
            response = content
        except HTTPError as exc:
            status_code = exc.code
            raw = redact_text(exc.read().decode("utf-8", errors="replace"), redactions)
            failure_class = "provider_http_error"
            failure = f"HTTP {exc.code}: {raw[-3000:]}"
        except (URLError, TimeoutError, OSError) as exc:
            failure_class = (
                "target_timeout"
                if isinstance(exc, (TimeoutError, socket.timeout))
                else "transport_error"
            )
            failure = f"OpenAI-compatible request failed: {safe_exception_type(exc)}"
        except (json.JSONDecodeError, ValueError) as exc:
            failure_class = "response_schema_error"
            failure = f"OpenAI-compatible response schema is invalid: {safe_exception_type(exc)}"
        response = redact_text(response, redactions)
        failure = redact_text(failure, redactions)
        completed = failure_class is None and bool(response.strip())
        return ModelOutcome(
            status="COMPLETED" if completed else "FAILED",
            response=response,
            raw=raw,
            process={
                "endpoint": endpoint,
                "http_status": status_code,
                "credential_env": self.api_key_env,
                "request_body_bytes": len(request.data or b""),
            },
            attempted_calls=1,
            completed_calls=1 if completed else 0,
            failure_class=failure_class,
            failure=failure,
        )
