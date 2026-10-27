"""FOCUS fork: retry behaviour of the OpenAI-compatible executor."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from rethinkskill.providers import openai_compatible
from rethinkskill.providers.openai_compatible import OpenAICompatibleExecutor
from rethinkskill.runtime.tasks import RenderedTask
from rethinkskill.runtime.types import freeze_model_outcome

FAKE_KEY = "sk_focus_fake_key_for_tests_only"
OK_BODY = json.dumps({"choices": [{"message": {"content": "```python\npass\n```"}}]})
EMPTY_BODY = json.dumps({"choices": [{"message": {"content": ""}}]})


class _ScriptedHandler(BaseHTTPRequestHandler):
    script: list[tuple[int, str]] = []
    seen: list[str] = []

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).seen.append(self.headers.get("Authorization", ""))
        code, body = type(self).script.pop(0)
        payload = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        pass


class OpenAIRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        _ScriptedHandler.script = []
        _ScriptedHandler.seen = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.sleeps: list[float] = []
        patch_sleep = mock.patch.object(openai_compatible, "_sleep", self.sleeps.append)
        patch_sleep.start()
        self.addCleanup(patch_sleep.stop)
        patch_env = mock.patch.dict(
            os.environ, {"FOCUS_TEST_KEY": FAKE_KEY, "RETHINKSKILL_RETRY_DELAYS": "1,2,3"}
        )
        patch_env.start()
        self.addCleanup(patch_env.stop)
        self.workspace = Path(tempfile.mkdtemp())
        self.executor = OpenAICompatibleExecutor(
            model="fake-model",
            base_url=f"http://127.0.0.1:{self.server.server_address[1]}/v1",
            api_key_env="FOCUS_TEST_KEY",
            reasoning_effort="",
        )
        self.rendered = RenderedTask(
            task_markdown="# Task\n\ndo it", skill_markdown="# Skill\n\nbe good", invocation="go"
        )

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _run(self, script: list[tuple[int, str]]):
        _ScriptedHandler.script = list(script)
        outcome = self.executor.execute(self.rendered, workspace=self.workspace, timeout_seconds=10)
        return freeze_model_outcome(outcome)

    def test_retries_5xx_then_succeeds(self) -> None:
        outcome = self._run([(503, "{}"), (503, "{}"), (200, OK_BODY)])
        self.assertEqual(outcome.status, "COMPLETED")
        self.assertEqual(outcome.process["retries"], 2)
        self.assertEqual(self.sleeps, [1.0, 2.0])
        self.assertEqual(outcome.attempted_calls, 1)
        self.assertEqual(outcome.completed_calls, 1)
        self.assertEqual(len(_ScriptedHandler.seen), 3)

    def test_429_is_retried(self) -> None:
        outcome = self._run([(429, "{}"), (200, OK_BODY)])
        self.assertEqual(outcome.status, "COMPLETED")
        self.assertEqual(outcome.process["retries"], 1)

    def test_400_is_not_retried(self) -> None:
        outcome = self._run([(400, '{"error":"bad"}')])
        self.assertEqual(outcome.status, "FAILED")
        self.assertEqual(outcome.failure_class, "provider_http_error")
        self.assertEqual(outcome.process["retries"], 0)
        self.assertEqual(self.sleeps, [])
        self.assertEqual(len(_ScriptedHandler.seen), 1)

    def test_gives_up_after_three_retries(self) -> None:
        outcome = self._run([(503, "{}")] * 4)
        self.assertEqual(outcome.status, "FAILED")
        self.assertEqual(outcome.failure_class, "provider_http_error")
        self.assertEqual(outcome.process["http_status"], 503)
        self.assertEqual(outcome.process["retries"], 3)
        self.assertEqual(self.sleeps, [1.0, 2.0, 3.0])
        self.assertEqual(len(_ScriptedHandler.seen), 4)

    def test_empty_content_is_retried(self) -> None:
        outcome = self._run([(200, EMPTY_BODY), (200, OK_BODY)])
        self.assertEqual(outcome.status, "COMPLETED")
        self.assertEqual(outcome.process["retries"], 1)

    def test_success_first_try_has_no_retries(self) -> None:
        outcome = self._run([(200, OK_BODY)])
        self.assertEqual(outcome.status, "COMPLETED")
        self.assertEqual(outcome.process["retries"], 0)
        self.assertEqual(self.sleeps, [])

    def test_key_never_in_evidence(self) -> None:
        outcome = self._run([(500, json.dumps({"echo": FAKE_KEY})), (200, OK_BODY)])
        self.assertEqual(_ScriptedHandler.seen[0], f"Bearer {FAKE_KEY}")
        dumped = json.dumps(outcome.public())
        self.assertNotIn(FAKE_KEY, dumped)
        failed = self._run([(400, json.dumps({"echo": FAKE_KEY}))])
        self.assertNotIn(FAKE_KEY, json.dumps(failed.public()))

    def test_bad_delay_setting_is_rejected(self) -> None:
        from rethinkskill.errors import ConfigurationError

        with mock.patch.dict(os.environ, {"RETHINKSKILL_RETRY_DELAYS": "ten"}):
            with self.assertRaises(ConfigurationError):
                self._run([(200, OK_BODY)])


if __name__ == "__main__":
    unittest.main()
