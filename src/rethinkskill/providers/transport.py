"""RethinkSkill providers transport."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from rethinkskill.errors import ConfigurationError, MissingArtifactError


class TransportKind(str, Enum):
    CODEX = "codex"
    CLAUDE_CODE = "claude-code"
    GEMINI_CLI = "gemini-cli"
    OPENAI_COMPATIBLE = "openai-compatible"


_PROVIDER_NAME = re.compile("^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


def validate_provider_name(value: object) -> str:
    """Validate and return one stable provider registration name."""
    if type(value) is TransportKind:
        return value.value
    if type(value) is not str or not _PROVIDER_NAME.fullmatch(value):
        raise ConfigurationError(f"invalid provider name: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class TransportConfig:
    kind: TransportKind
    model: str
    reasoning_effort: str = "medium"
    launcher: str | None = None
    sandbox: str = "read-only"
    api_base_url: str | None = None
    api_key_env: str | None = None
    claude_effort: str = "medium"
    claude_base_url: str | None = None
    claude_auth_token_env: str | None = None
    provider: str | None = None

    @property
    def provider_name(self) -> str:
        """Return the selected provider registration name."""
        return self.provider or self.kind.value

    def validate_static(self) -> None:
        validate_transport(self)

    def public_manifest(self) -> dict[str, object]:
        return transport_manifest(self)


_COMMON_NAMES = (
    "ALL_PROXY",
    "CURL_CA_BUNDLE",
    "CUDA_VISIBLE_DEVICES",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "NO_PROXY",
    "OMP_NUM_THREADS",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TEMP",
    "TERM",
    "TOKENIZERS_PARALLELISM",
    "TMP",
    "TMPDIR",
    "TZ",
    "USER",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "all_proxy",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)

_PROVIDER_PREFIXES = {
    "claude-code": ("ANTHROPIC_", "CLAUDE_"),
    "codex": ("CODEX_", "OPENAI_"),
    "gemini-cli": ("CLOUDSDK_", "GCLOUD_", "GEMINI_", "GOOGLE_"),
    "openai-compatible": ("OPENAI_",),
}

_ALIASES = {"claude_code": "claude-code", "gemini": "gemini-cli", "gemini_cli": "gemini-cli"}

_DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"


def _provider_name(provider: str) -> str:
    normalized = _ALIASES.get(provider, provider)
    if normalized not in _PROVIDER_PREFIXES:
        raise ConfigurationError(f"unsupported provider environment policy: {provider}")
    return normalized


def base_process_environment(*, source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy only non-provider process settings needed by local launchers."""
    observed = os.environ if source is None else source
    environment = {name: observed[name] for name in _COMMON_NAMES if name in observed}
    environment.setdefault("PATH", _DEFAULT_PATH)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def provider_process_environment(
    provider: str,
    *,
    source: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build one provider-specific allowlisted child environment."""
    normalized = _provider_name(provider)
    observed = os.environ if source is None else source
    environment = base_process_environment(source=observed)
    prefixes = _PROVIDER_PREFIXES[normalized]
    for name, value in observed.items():
        if name.startswith(prefixes):
            environment[name] = value
    if overrides:
        for name, value in overrides.items():
            if not isinstance(name, str) or not name:
                raise ConfigurationError(
                    "provider environment override names must be non-empty text"
                )
            if not isinstance(value, str):
                raise ConfigurationError(f"provider environment override must be text: {name}")
            environment[name] = value
    return environment


def provider_environment_policy(provider: str) -> dict[str, object]:
    """Return a credential-free description of the child environment policy."""
    normalized = _provider_name(provider)
    return {
        "inherit": "allowlist",
        "common_names": list(_COMMON_NAMES),
        "provider_prefixes": list(_PROVIDER_PREFIXES[normalized]),
        "python_user_site": False,
        "ambient_pythonpath": False,
        "ambient_dynamic_loader": False,
    }


def launcher_path(config: TransportConfig, default: str) -> Path:
    value = config.launcher or default
    located = value if Path(value).is_absolute() else shutil.which(value)
    if not located or not Path(located).is_file():
        raise MissingArtifactError(f"transport launcher is absent: {value}")
    return Path(located).resolve()


_ENVIRONMENT_NAME = re.compile("^[A-Za-z_][A-Za-z0-9_]*$")

_REASONING_EFFORTS = {"", "low", "medium", "high", "xhigh", "max"}


def _validate_https_url(value: str, *, label: str) -> None:
    if (
        type(value) is not str
        or not value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ConfigurationError(f"{label} must be an absolute HTTPS URL")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ConfigurationError(f"{label} must be an absolute HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigurationError(f"{label} cannot contain credentials, query, or fragment")


def _validate_environment_name(value: str, *, label: str) -> None:
    if type(value) is not str or not _ENVIRONMENT_NAME.fullmatch(value):
        raise ConfigurationError(f"invalid {label}: {value!r}")


def validate_transport(config: TransportConfig) -> None:
    if type(config) is not TransportConfig:
        raise ConfigurationError("transport config must be an exact TransportConfig")
    if type(config.kind) is not TransportKind:
        raise ConfigurationError(f"invalid transport kind: {config.kind!r}")
    if config.provider is not None:
        validate_provider_name(config.provider)
    if (
        type(config.model) is not str
        or not config.model.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in config.model)
    ):
        raise ConfigurationError("model must be non-empty")
    if (
        type(config.reasoning_effort) is not str
        or config.reasoning_effort not in _REASONING_EFFORTS
    ):
        raise ConfigurationError(f"invalid reasoning effort: {config.reasoning_effort}")
    if config.launcher is not None and (
        type(config.launcher) is not str or not config.launcher.strip() or "\x00" in config.launcher
    ):
        raise ConfigurationError("launcher must be non-empty text or None")
    if type(config.sandbox) is not str:
        raise ConfigurationError("sandbox must be text")
    for label, value in (
        ("API base URL", config.api_base_url),
        ("API credential environment name", config.api_key_env),
        ("Claude gateway URL", config.claude_base_url),
        ("Claude credential environment name", config.claude_auth_token_env),
    ):
        if value is not None and type(value) is not str:
            raise ConfigurationError(f"{label} must be text or None")
    if (
        type(config.claude_effort) is not str
        or not config.claude_effort.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in config.claude_effort)
    ):
        raise ConfigurationError("Claude effort must be non-empty text")
    if config.kind is TransportKind.CODEX:
        if config.sandbox != "read-only":
            raise ConfigurationError("Codex target sandbox must be read-only")
    elif config.kind is TransportKind.OPENAI_COMPATIBLE:
        if not config.api_base_url or not config.api_key_env:
            raise ConfigurationError(
                "openai-compatible transport requires api_base_url and api_key_env"
            )
        _validate_https_url(config.api_base_url, label="API base URL")
        _validate_environment_name(config.api_key_env, label="API credential environment name")
    elif config.kind is TransportKind.CLAUDE_CODE:
        if bool(config.claude_base_url) != bool(config.claude_auth_token_env):
            raise ConfigurationError(
                "Claude gateway requires both claude_base_url and claude_auth_token_env"
            )
        if config.claude_base_url:
            _validate_https_url(config.claude_base_url, label="Claude gateway URL")
            assert config.claude_auth_token_env is not None
            _validate_environment_name(
                config.claude_auth_token_env, label="Claude credential environment name"
            )
    elif config.kind is TransportKind.GEMINI_CLI:
        if config.sandbox != "read-only":
            raise ConfigurationError("Gemini CLI requires the fixed local execution policy")


def transport_manifest(config: TransportConfig) -> dict[str, object]:
    validate_transport(config)
    value: dict[str, object] = {
        "kind": config.provider_name,
        "transport_kind": config.kind.value,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
    }
    if config.kind is TransportKind.CODEX:
        value.update({"launcher": config.launcher or "codex", "sandbox": config.sandbox})
    elif config.kind is TransportKind.CLAUDE_CODE:
        value.update({"launcher": config.launcher or "claude", "effort": config.claude_effort})
        if config.claude_base_url:
            value["gateway"] = {
                "base_url": config.claude_base_url.rstrip("/"),
                "auth_token_env": config.claude_auth_token_env,
            }
    elif config.kind is TransportKind.GEMINI_CLI:
        value.update(
            {
                "launcher": config.launcher or "gemini",
                "tool_policy": "disabled_except_task_local_attachment_read",
            }
        )
    else:
        value.update(
            {"base_url": str(config.api_base_url).rstrip("/"), "api_key_env": config.api_key_env}
        )
    return value
