"""RethinkSkill providers builtin."""

from __future__ import annotations

from rethinkskill.providers.catalog import ProviderInterface, ProviderSpec
from rethinkskill.providers.claude_code import ClaudeCodeCliExecutor
from rethinkskill.providers.codex import CodexCliExecutor
from rethinkskill.providers.gemini_cli import GeminiCliExecutor
from rethinkskill.providers.openai_compatible import OpenAICompatibleExecutor
from rethinkskill.providers.transport import TransportKind


def claude_code_provider_spec() -> ProviderSpec:
    return ProviderSpec(
        kind=TransportKind.CLAUDE_CODE,
        executor_type=ClaudeCodeCliExecutor,
        interface=ProviderInterface.CLI,
        default_launcher="claude",
        credential_mode="session_or_explicit_gateway_env",
    )


def codex_provider_spec() -> ProviderSpec:
    return ProviderSpec(
        kind=TransportKind.CODEX,
        executor_type=CodexCliExecutor,
        interface=ProviderInterface.CLI,
        default_launcher="codex",
        credential_mode="codex_session",
    )


def gemini_cli_provider_spec() -> ProviderSpec:
    return ProviderSpec(
        kind=TransportKind.GEMINI_CLI,
        executor_type=GeminiCliExecutor,
        interface=ProviderInterface.CLI,
        default_launcher="gemini",
        credential_mode="gemini_cli_environment",
    )


def openai_compatible_provider_spec() -> ProviderSpec:
    return ProviderSpec(
        kind=TransportKind.OPENAI_COMPATIBLE,
        executor_type=OpenAICompatibleExecutor,
        interface=ProviderInterface.HTTP,
        credential_mode="explicit_api_key_env",
    )
