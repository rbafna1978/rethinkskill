from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from rethinkskill.errors import ConfigurationError
from rethinkskill.utils.fs import OutputLock


class OutputLockTests(unittest.TestCase):
    def test_lock_records_owner_with_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            lock = OutputLock(output)
            with lock:
                self.assertIsNotNone(lock.handle)
                self.assertEqual(
                    lock.path.read_text(encoding="utf-8").split()[0],
                    f"pid={os.getpid()}",
                )
                self.assertEqual(
                    stat.S_IMODE(lock.path.stat().st_mode),
                    0o600,
                )
            self.assertIsNone(lock.handle)

    def test_lock_rejects_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            victim = root / "victim.txt"
            victim.write_text("preserve", encoding="utf-8")
            lock = OutputLock(output)
            lock.path.symlink_to(victim)

            with self.assertRaisesRegex(ConfigurationError, "symlink"):
                with lock:
                    pass

            self.assertEqual(victim.read_text(encoding="utf-8"), "preserve")
            self.assertIsNone(lock.handle)

    def test_contended_lock_closes_failed_handle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            first = OutputLock(output)
            second = OutputLock(output)
            with first:
                with self.assertRaisesRegex(
                    ConfigurationError,
                    "another runner owns",
                ):
                    with second:
                        pass
                self.assertIsNone(second.handle)
                self.assertIsNotNone(first.handle)

    def test_lock_rejects_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            victim = root / "victim.txt"
            victim.write_text("preserve", encoding="utf-8")
            lock = OutputLock(output)
            os.link(victim, lock.path)

            with self.assertRaisesRegex(ConfigurationError, "regular file"):
                with lock:
                    pass

            self.assertEqual(victim.read_text(encoding="utf-8"), "preserve")
            self.assertIsNone(lock.handle)
