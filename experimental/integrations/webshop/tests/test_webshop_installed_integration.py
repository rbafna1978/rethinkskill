from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from rethinkskill_webshop.runtime import SubprocessWebShopFactory


@unittest.skipUnless(
    os.environ.get("RETHINKSKILL_WEBSHOP_ROOT")
    and os.environ.get("RETHINKSKILL_WEBSHOP_PYTHON")
    and os.environ.get("JAVA_HOME"),
    "configure the official WebShop checkout, Python, and Java runtimes",
)
class InstalledWebShopIntegrationTests(unittest.TestCase):
    def test_official_text_environment_reset(self) -> None:
        factory = SubprocessWebShopFactory.from_environment()
        manifest = factory.dependency_manifest()
        self.assertTrue(manifest["ready"], manifest)
        with tempfile.TemporaryDirectory() as temporary:
            episode = factory.open(
                session=0,
                num_products=1000,
                verifier_workspace=Path(temporary),
                timeout_seconds=180,
            )
            try:
                state = episode.reset()
                self.assertTrue(state.observation.strip())
                self.assertIn("clickables", state.available_actions)
                self.assertFalse(state.done)
                self.assertEqual(state.reward, 0.0)
            finally:
                episode.close()


if __name__ == "__main__":
    unittest.main()
