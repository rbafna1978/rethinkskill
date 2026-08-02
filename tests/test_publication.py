from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.verify_publication import verify_publication


class PublicationVerifierTests(unittest.TestCase):
    def test_current_tracked_publication_surface_is_clean(self) -> None:
        root = Path(__file__).resolve().parents[1]
        if not (root / ".git").exists():
            self.skipTest("requires the authoritative Git-tracked file set")
        report = verify_publication(root)
        self.assertEqual(
            report["status"],
            "RETHINKSKILL_PUBLICATION_SURFACE_VALIDATED",
        )
        self.assertEqual(report["model_calls"], 0)

    def test_secret_is_rejected_without_echoing_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            secret = "sk-" + "A" * 24
            path = root / "module.py"
            path.write_text(f'TOKEN = "{secret}"\n', encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "openai_style_secret:module.py",
            ) as raised:
                verify_publication(root)
            self.assertNotIn(secret, str(raised.exception))

    def test_untracked_git_file_is_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(
                ("git", "init", "-q", str(root)),
                check=True,
            )
            safe = root / "safe.py"
            safe.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(
                ("git", "-C", str(root), "add", "safe.py"),
                check=True,
            )
            secret = root / "untracked.py"
            secret.write_text(
                'TOKEN = "sk-' + "B" * 24 + '"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "openai_style_secret:untracked.py",
            ):
                verify_publication(root)

    def test_host_specific_absolute_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "config.json"
            host_path = "/" + "Users/example/private/input.json"
            path.write_text(
                f'{{"data": "{host_path}"}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "macos_user_path:config.json",
            ):
                verify_publication(root)

    def test_protected_evidence_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "final_results" / "receipt.json"
            path.parent.mkdir()
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "protected_path:final_results/receipt.json",
            ):
                verify_publication(
                    root,
                    files=(path,),
                )


if __name__ == "__main__":
    unittest.main()
