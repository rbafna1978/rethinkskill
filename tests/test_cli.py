from __future__ import annotations

import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from rethinkskill.cli import build_parser, main, parser_command_names


class CliTests(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[int, dict[str, object], str]:
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(list(arguments))
        stream = stdout.getvalue() if code == 0 else stderr.getvalue()
        return code, json.loads(stream), stderr.getvalue()

    def test_public_command_surface_is_small_and_explicit(self) -> None:
        self.assertEqual(
            parser_command_names(build_parser()),
            (
                "benchmark-catalog",
                "provider-catalog",
                "provider-preflight",
                "optimizer-catalog",
                "plan-protocol",
                "native-preflight",
                "native-run",
                "native-evolution-preflight",
                "native-evolve",
                "validate-run",
                "validate-results",
                "recovery-plan",
                "evaluate-cases",
                "validate-evaluation",
            ),
        )

    def test_benchmark_catalog_is_unified_and_zero_call(self) -> None:
        code, value, stderr = self.invoke("benchmark-catalog")
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(value["status"], "RETHINKSKILL_CAPABILITY_CATALOG")
        self.assertEqual(value["counts"]["total"], 20)
        self.assertEqual(value["model_calls"], 0)

    def test_provider_catalog_is_zero_call_and_provider_neutral(self) -> None:
        code, value, _ = self.invoke("provider-catalog")
        self.assertEqual(code, 0)
        self.assertEqual(
            [provider["kind"] for provider in value["providers"]],
            ["codex", "claude-code", "gemini-cli", "openai-compatible"],
        )
        self.assertEqual(value["model_calls"], 0)

    def test_optimizer_catalog_is_zero_call(self) -> None:
        code, value, _ = self.invoke("optimizer-catalog")
        self.assertEqual(code, 0)
        self.assertEqual(value["count"], 1)
        self.assertEqual(value["model_calls"], 0)

    def test_protocol_plan_is_zero_call(self) -> None:
        code, value, _ = self.invoke("plan-protocol", "--benchmark", "searchqa")
        self.assertEqual(code, 0)
        self.assertEqual(value["status"], "RETHINKSKILL_PROTOCOL_PLAN_ZERO_CALL")
        self.assertEqual(value["model_calls"], 0)

    def test_native_run_requires_explicit_authorization(self) -> None:
        code, value, _ = self.invoke(
            "native-run",
            "--benchmark",
            "searchqa",
            "--dataset",
            "examples/live_smoke/qa",
            "--skill",
            "examples/live_smoke/skill.md",
            "--out-root",
            "runs/cli-unauthorized-test",
            "--split",
            "test",
            "--limit",
            "1",
            "--transport",
            "openai-compatible",
            "--model",
            "test-model",
            "--api-base-url",
            "https://example.com/v1",
            "--api-key-env",
            "TEST_API_KEY",
        )
        self.assertEqual(code, 2)
        self.assertEqual(value["error_type"], "AuthorizationRequiredError")
        self.assertEqual(value["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
