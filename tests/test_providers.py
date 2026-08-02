from __future__ import annotations

import unittest

from rethinkskill.errors import ConfigurationError
from rethinkskill.providers.transport import (
    base_process_environment,
    provider_environment_policy,
    provider_process_environment,
)


class TransportEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "HOME": "/safe/home",
            "PATH": "/safe/bin",
            "HTTPS_PROXY": "https://proxy.example",
            "PYTHONPATH": "/host/injection",
            "LD_PRELOAD": "/host/injection.so",
            "UNRELATED_API_KEY": "unrelated-secret",
            "OPENAI_API_KEY": "openai-secret",
            "CODEX_HOME": "/safe/codex",
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "CLAUDE_CONFIG_DIR": "/safe/claude",
            "GEMINI_API_KEY": "gemini-secret",
            "GOOGLE_APPLICATION_CREDENTIALS": "/safe/google.json",
        }

    def test_base_environment_excludes_credentials_and_injection(self) -> None:
        environment = base_process_environment(source=self.source)
        self.assertEqual(environment["HOME"], "/safe/home")
        self.assertEqual(environment["PATH"], "/safe/bin")
        self.assertNotIn("UNRELATED_API_KEY", environment)
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    def test_codex_environment_inherits_only_codex_credentials(self) -> None:
        environment = provider_process_environment(
            "codex",
            source=self.source,
        )
        self.assertEqual(environment["OPENAI_API_KEY"], "openai-secret")
        self.assertEqual(environment["CODEX_HOME"], "/safe/codex")
        self.assertNotIn("ANTHROPIC_API_KEY", environment)
        self.assertNotIn("GEMINI_API_KEY", environment)
        self.assertNotIn("UNRELATED_API_KEY", environment)

    def test_claude_environment_inherits_only_claude_credentials(self) -> None:
        environment = provider_process_environment(
            "claude_code",
            source=self.source,
        )
        self.assertEqual(
            environment["ANTHROPIC_API_KEY"],
            "anthropic-secret",
        )
        self.assertEqual(
            environment["CLAUDE_CONFIG_DIR"],
            "/safe/claude",
        )
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("GEMINI_API_KEY", environment)

    def test_gemini_environment_inherits_only_google_credentials(self) -> None:
        environment = provider_process_environment(
            "gemini-cli",
            source=self.source,
        )
        self.assertEqual(environment["GEMINI_API_KEY"], "gemini-secret")
        self.assertEqual(
            environment["GOOGLE_APPLICATION_CREDENTIALS"],
            "/safe/google.json",
        )
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("ANTHROPIC_API_KEY", environment)

    def test_explicit_overrides_do_not_mutate_source(self) -> None:
        environment = provider_process_environment(
            "openai-compatible",
            source=self.source,
            overrides={"TARGET_API_KEY": "mapped-secret"},
        )
        self.assertEqual(environment["TARGET_API_KEY"], "mapped-secret")
        self.assertNotIn("TARGET_API_KEY", self.source)

    def test_public_policy_contains_names_not_values(self) -> None:
        policy = provider_environment_policy("codex")
        self.assertEqual(policy["inherit"], "allowlist")
        self.assertIn("OPENAI_", policy["provider_prefixes"])
        self.assertNotIn("openai-secret", repr(policy))
        self.assertFalse(policy["ambient_pythonpath"])
        self.assertFalse(policy["ambient_dynamic_loader"])

    def test_unknown_provider_policy_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError,
            "unsupported provider environment policy",
        ):
            provider_process_environment("unknown", source=self.source)


if __name__ == "__main__":
    unittest.main()
