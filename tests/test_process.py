from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rethinkskill.utils.process import ProcessResult, execute, secret_values


class ProcessReceiptTests(unittest.TestCase):
    def test_secret_like_environment_values_are_redacted(self) -> None:
        result = ProcessResult(
            returncode=1,
            timed_out=False,
            elapsed_seconds=0.25,
            stdout="token=super-secret-value",
            stderr="safe",
        )
        redactions = secret_values(
            {
                "TARGET_API_KEY": "super-secret-value",
                "PARALLELISM": "20",
            }
        )
        public = result.to_public_dict(redactions=redactions)
        self.assertNotIn("super-secret-value", public["stdout_tail"])
        self.assertIn("[REDACTED]", public["stdout_tail"])

    def test_short_or_non_secret_values_are_not_treated_as_credentials(self) -> None:
        self.assertEqual(
            secret_values({"TARGET_API_KEY": "short", "ORDINARY": "long-value"}),
            (),
        )

    def test_execute_rejects_invalid_timeout_before_launch(self) -> None:
        for timeout in (0, -1, True, 1.5):
            with (
                self.subTest(timeout=timeout),
                patch("rethinkskill.utils.process.subprocess.Popen") as popen,
            ):
                with self.assertRaises(ValueError):
                    execute(
                        ("command",),
                        cwd=Path.cwd(),
                        environment={},
                        timeout_seconds=timeout,  # type: ignore[arg-type]
                    )
                popen.assert_not_called()

    def test_timeout_cleanup_tolerates_child_exit_signal_race(self) -> None:
        process = Mock(pid=123, returncode=143)
        process.communicate.side_effect = (
            subprocess.TimeoutExpired(("command",), 1),
            ("partial stdout", "partial stderr"),
        )
        with (
            patch(
                "rethinkskill.utils.process.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "rethinkskill.utils.process.os.killpg",
                side_effect=ProcessLookupError,
            ) as killpg,
        ):
            result = execute(
                ("command",),
                cwd=Path.cwd(),
                environment={},
                timeout_seconds=1,
            )
        self.assertTrue(result.timed_out)
        self.assertEqual(result.stdout, "partial stdout")
        self.assertEqual(killpg.call_count, 1)

    def test_timeout_cleanup_escalation_tolerates_signal_races(self) -> None:
        process = Mock(pid=456, returncode=137)
        process.communicate.side_effect = (
            subprocess.TimeoutExpired(("command",), 1),
            subprocess.TimeoutExpired(("command",), 10),
            ("", "killed"),
        )
        with (
            patch(
                "rethinkskill.utils.process.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "rethinkskill.utils.process.os.killpg",
                side_effect=ProcessLookupError,
            ) as killpg,
        ):
            result = execute(
                ("command",),
                cwd=Path.cwd(),
                environment={},
                timeout_seconds=1,
            )
        self.assertTrue(result.timed_out)
        self.assertEqual(result.stderr, "killed")
        self.assertEqual(killpg.call_count, 2)
