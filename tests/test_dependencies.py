from __future__ import annotations

import unittest
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from rethinkskill.errors import ConfigurationError
from rethinkskill.runtime.types import (
    distribution_file_identity,
    numeric_version,
    package_tree_identity,
    probe_distribution,
)


class DependencyProbeTests(unittest.TestCase):
    def test_package_tree_identity_is_portable_and_ignores_caches(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "adapter"
            root.mkdir()
            module = root / "runtime.py"
            module.write_text("value = 1\n", encoding="utf-8")
            first = package_tree_identity(module)
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "runtime.pyc").write_bytes(b"cache")
            second = package_tree_identity(module)
            module.write_text("value = 2\n", encoding="utf-8")
            third = package_tree_identity(module)
        self.assertEqual(first, second)
        self.assertNotEqual(first["sha256"], third["sha256"])
        self.assertNotIn("root", first)

    def test_distribution_identity_hashes_declared_files(self) -> None:
        class FakeDistribution:
            version = "1.2.3"
            files = (Path("package/b.py"), Path("package/a.py"))

            def __init__(self, root: Path) -> None:
                self.root = root

            def locate_file(self, item: Path) -> Path:
                return self.root / item

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "package").mkdir()
            (root / "package/a.py").write_text("a = 1\n", encoding="utf-8")
            (root / "package/b.py").write_text("b = 1\n", encoding="utf-8")
            installed = FakeDistribution(root)
            with mock.patch(
                "rethinkskill.runtime.types.metadata.distribution",
                return_value=installed,
            ):
                first = distribution_file_identity(
                    "example",
                    allowed_roots=(root,),
                )
                (root / "package/b.py").write_text(
                    "b = 2\n",
                    encoding="utf-8",
                )
                second = distribution_file_identity(
                    "example",
                    allowed_roots=(root,),
                )
        self.assertEqual(first["files"], 2)
        self.assertEqual(first["version"], "1.2.3")
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertNotIn(str(root), repr(first))

    def test_distribution_identity_rejects_symlinked_files(self) -> None:
        class FakeDistribution:
            version = "1"
            files = (Path("package/value.py"),)

            def __init__(self, root: Path) -> None:
                self.root = root

            def locate_file(self, item: Path) -> Path:
                return self.root / item

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "package").mkdir()
            target = root / "target.py"
            target.write_text("value = 1\n", encoding="utf-8")
            (root / "package/value.py").symlink_to(target)
            with (
                mock.patch(
                    "rethinkskill.runtime.types.metadata.distribution",
                    return_value=FakeDistribution(root),
                ),
                self.assertRaisesRegex(ConfigurationError, "must not be a symlink"),
            ):
                distribution_file_identity("example")

    def test_distribution_identity_allows_bounded_console_script_paths(
        self,
    ) -> None:
        class FakeDistribution:
            version = "1"
            files = (Path("../../../bin/example-tool"),)

            def __init__(self, root: Path) -> None:
                self.root = root

            def locate_file(self, item: Path) -> Path:
                return self.root / item

        with TemporaryDirectory() as temporary:
            environment = Path(temporary) / "environment"
            site_packages = environment / "lib/python/site-packages"
            scripts = environment / "bin"
            site_packages.mkdir(parents=True)
            scripts.mkdir()
            (scripts / "example-tool").write_text(
                "#!/usr/bin/env python\n",
                encoding="utf-8",
            )
            installed = FakeDistribution(site_packages)
            with (
                mock.patch(
                    "rethinkskill.runtime.types.metadata.distribution",
                    return_value=installed,
                ),
                mock.patch(
                    "rethinkskill.utils.release._installed_environment_roots",
                    return_value=(site_packages, scripts),
                ),
            ):
                identity = distribution_file_identity(
                    "example",
                )
        self.assertEqual(identity["files"], 1)
        self.assertNotIn(str(environment), repr(identity))

    def test_distribution_identity_rejects_parent_path_outside_environment(
        self,
    ) -> None:
        class FakeDistribution:
            version = "1"
            files = (Path("../../../../outside.py"),)

            def __init__(self, root: Path) -> None:
                self.root = root

            def locate_file(self, item: Path) -> Path:
                return self.root / item

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            site_packages = root / "environment/lib/python/site-packages"
            site_packages.mkdir(parents=True)
            (root / "outside.py").write_text("value = 1\n", encoding="utf-8")
            with (
                mock.patch(
                    "rethinkskill.runtime.types.metadata.distribution",
                    return_value=FakeDistribution(site_packages),
                ),
                self.assertRaisesRegex(ConfigurationError, "outside"),
            ):
                distribution_file_identity(
                    "example",
                    allowed_roots=(site_packages,),
                )

    def test_distribution_identity_rejects_editable_path_indirection(self) -> None:
        class FakeDistribution:
            version = "1"
            files = (Path("__editable__.example.pth"),)

            def __init__(self, root: Path) -> None:
                self.root = root

            def locate_file(self, item: Path) -> Path:
                return self.root / item

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "__editable__.example.pth").write_text(
                "/external/source\n",
                encoding="utf-8",
            )
            with (
                mock.patch(
                    "rethinkskill.runtime.types.metadata.distribution",
                    return_value=FakeDistribution(root),
                ),
                self.assertRaisesRegex(ConfigurationError, "indirect or editable"),
            ):
                distribution_file_identity("example")

    def test_numeric_release_prefix(self) -> None:
        self.assertEqual(numeric_version("3.1.5"), (3, 1, 5))
        self.assertEqual(numeric_version("0.4.2+local"), (0, 4, 2))
        self.assertIsNone(numeric_version("release"))

    @mock.patch(
        "rethinkskill.runtime.types.metadata.version",
        return_value="3.1.5",
    )
    @mock.patch(
        "rethinkskill.runtime.types.util.find_spec",
        return_value=object(),
    )
    def test_probe_requires_both_module_and_compatible_distribution(
        self,
        _find_spec,
        _version,
    ) -> None:
        value = probe_distribution(
            distribution="example",
            import_name="example",
            requirement=">=3.1,<4",
            accepts=lambda version: numeric_version(version) >= (3, 1),
        )
        self.assertTrue(value["ready"])
        self.assertEqual(value["version"], "3.1.5")
        self.assertEqual(value["requirement"], ">=3.1,<4")

    @mock.patch(
        "rethinkskill.runtime.types.metadata.version",
        side_effect=metadata.PackageNotFoundError,
    )
    @mock.patch(
        "rethinkskill.runtime.types.util.find_spec",
        return_value=None,
    )
    def test_absent_distribution_is_not_ready(
        self,
        _find_spec,
        _version,
    ) -> None:
        value = probe_distribution(
            distribution="missing",
            import_name="missing",
            requirement="==1",
            accepts=lambda version: version == "1",
        )
        self.assertFalse(value["ready"])
        self.assertFalse(value["module_available"])
        self.assertIsNone(value["version"])


if __name__ == "__main__":
    unittest.main()
