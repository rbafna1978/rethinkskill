from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import rethinkskill_webshop.environment_probe as probe


class _Distribution:
    def __init__(
        self,
        root: Path,
        *,
        name: str = "Fixture_Package",
        version: str = "1.2.3",
        files: tuple[str, ...] = ("fixture/__init__.py",),
    ) -> None:
        self.root = root
        self.metadata = {"Name": name}
        self.version = version
        self.files = files

    def locate_file(self, item: object) -> Path:
        return self.root / str(item)


class WebShopEnvironmentProbeTests(unittest.TestCase):
    def test_probe_source_parses_as_python_3_8(self) -> None:
        source = Path(probe.__file__).read_text(encoding="utf-8")
        ast.parse(source, filename=probe.__file__, feature_version=(3, 8))

    def test_environment_identity_hashes_installed_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture/__init__.py"
            source.parent.mkdir()
            source.write_text("VALUE = 1\n", encoding="utf-8")
            distribution = _Distribution(root)
            first = probe._environment_identity(
                (distribution,),
                allowed_roots=(root,),
            )
            source.write_text("VALUE = 2\n", encoding="utf-8")
            second = probe._environment_identity(
                (distribution,),
                allowed_roots=(root,),
            )
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["distribution_count"], 1)
        self.assertEqual(
            first["distributions"][0]["normalized_name"],
            "fixture-package",
        )
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_environment_identity_rejects_editable_source_indirection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct_url = root / "fixture-1.0.dist-info/direct_url.json"
            direct_url.parent.mkdir()
            direct_url.write_text(
                json.dumps(
                    {
                        "url": "file:///private/source",
                        "dir_info": {"editable": True},
                    }
                ),
                encoding="utf-8",
            )
            distribution = _Distribution(
                root,
                files=("fixture-1.0.dist-info/direct_url.json",),
            )
            with self.assertRaisesRegex(ValueError, "editable"):
                probe._environment_identity(
                    (distribution,),
                    allowed_roots=(root,),
                )

    def test_environment_identity_rejects_pth_and_symlink_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pth = root / "fixture.pth"
            pth.write_text("/private/source\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "indirect"):
                probe._environment_identity(
                    (_Distribution(root, files=("fixture.pth",)),),
                    allowed_roots=(root,),
                )
            target = root / "target.py"
            target.write_text("VALUE = 1\n", encoding="utf-8")
            link = root / "fixture/__init__.py"
            link.parent.mkdir()
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                probe._environment_identity(
                    (_Distribution(root),),
                    allowed_roots=(root,),
                )

    def test_environment_identity_hashes_non_external_pth_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "fixture"
            package.mkdir()
            pth = root / "distutils-precedence.pth"
            pth.write_text(
                "# frozen startup behavior\n"
                "import _distutils_hack; _distutils_hack.add_shim()\n"
                "fixture\n",
                encoding="utf-8",
            )
            distribution = _Distribution(
                root,
                files=("distutils-precedence.pth",),
            )
            identity = probe._environment_identity(
                (distribution,),
                allowed_roots=(root,),
            )
        self.assertEqual(identity["distribution_count"], 1)
        self.assertEqual(identity["files"], 1)

    def test_environment_identity_deduplicates_identical_record_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            distribution = _Distribution(
                root,
                files=("fixture.py", "fixture.py"),
            )
            identity = probe._environment_identity(
                (distribution,),
                allowed_roots=(root,),
            )
        self.assertEqual(identity["files"], 1)

    def test_environment_identity_rejects_duplicate_normalized_names(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "one.py"
            second = root / "two.py"
            first.write_text("1\n", encoding="utf-8")
            second.write_text("2\n", encoding="utf-8")
            distributions = (
                _Distribution(
                    root,
                    name="fixture_package",
                    files=("one.py",),
                ),
                _Distribution(
                    root,
                    name="fixture-package",
                    files=("two.py",),
                ),
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                probe._environment_identity(
                    distributions,
                    allowed_roots=(root,),
                )

    def test_environment_identity_rejects_inventory_outside_runtime_roots(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "environment"
            outside = root / "outside"
            allowed.mkdir()
            source = outside / "fixture.py"
            source.parent.mkdir()
            source.write_text("VALUE = 1\n", encoding="utf-8")
            distribution = _Distribution(
                outside,
                files=("fixture.py",),
            )
            with self.assertRaisesRegex(ValueError, "outside"):
                probe._environment_identity(
                    (distribution,),
                    allowed_roots=(allowed,),
                )


if __name__ == "__main__":
    unittest.main()
