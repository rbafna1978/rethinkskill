#!/usr/bin/env python3
"""Validate built archives against their source packages without model calls."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class Package:
    name: str
    source: Path
    module: str
    artifacts: Path


def _single(directory: Path, suffix: str) -> Path:
    matches = sorted(directory.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise ValueError(f"expected one {suffix} archive in {directory}; found {len(matches)}")
    return matches[0]


def _portable(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"archive contains an unsafe path: {name}")
    return path


def _source_files(package: Package) -> dict[str, bytes]:
    source_root = package.source / "src" / package.module
    if not source_root.is_dir():
        raise ValueError(f"package source is absent: {source_root}")
    files = {
        path.relative_to(package.source / "src").as_posix(): path.read_bytes()
        for path in sorted(source_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    if not files:
        raise ValueError(f"package contains no Python source: {package.name}")
    return files


def _sdist_payloads(path: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            portable = _portable(member.name)
            if member.issym() or member.islnk() or (not member.isfile() and not member.isdir()):
                raise ValueError(f"sdist contains a link or special file: {member.name}")
            if not member.isfile():
                continue
            relative = PurePosixPath(*portable.parts[1:])
            if not relative.parts:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"sdist member is unreadable: {member.name}")
            payloads[relative.as_posix()] = handle.read()
    return payloads


def _wheel_payloads(path: Path) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            portable = _portable(info.filename)
            if info.is_dir():
                continue
            payloads[portable.as_posix()] = archive.read(info)
    return payloads


def _forbidden(paths: set[str]) -> None:
    forbidden = (
        ".git/",
        "final_results/",
        "manuscript/",
        "runs/",
        "paper_data/",
        "__pycache__/",
    )
    for name in paths:
        normalized = f"/{name}"
        protected = any(
            normalized.startswith(f"/{prefix}") or f"/{prefix}" in normalized
            for prefix in forbidden
        )
        if protected:
            raise ValueError(f"archive contains a protected path: {name}")


def _verify_source_identity(
    payloads: dict[str, bytes], source_files: dict[str, bytes], *, prefix: str
) -> None:
    missing: list[str] = []
    drifted: list[str] = []
    for relative, expected in source_files.items():
        archived = f"{prefix}{relative}"
        if archived not in payloads:
            missing.append(archived)
        elif payloads[archived] != expected:
            drifted.append(archived)
    if missing or drifted:
        raise ValueError(f"archive source mismatch; missing={missing[:5]}; drifted={drifted[:5]}")


def _report(path: Path, payloads: dict[str, bytes]) -> dict[str, object]:
    _forbidden(set(payloads))
    return {
        "path": str(path),
        "files": len(payloads),
        "bytes": sum(len(payload) for payload in payloads.values()),
    }


def verify_distributions(packages: tuple[Package, ...]) -> dict[str, object]:
    archives: dict[str, dict[str, object]] = {}
    for package in packages:
        source_files = _source_files(package)
        sdist = _single(package.artifacts, ".tar.gz")
        wheel = _single(package.artifacts, ".whl")
        sdist_payloads = _sdist_payloads(sdist)
        wheel_payloads = _wheel_payloads(wheel)
        _verify_source_identity(sdist_payloads, source_files, prefix="src/")
        _verify_source_identity(wheel_payloads, source_files, prefix="")
        for required in ("pyproject.toml", "README.md"):
            if required not in sdist_payloads:
                raise ValueError(f"{package.name} sdist omits {required}")
        archives[f"{package.name}_sdist"] = _report(sdist, sdist_payloads)
        archives[f"{package.name}_wheel"] = _report(wheel, wheel_payloads)
    return {
        "schema_version": 1,
        "status": "RETHINKSKILL_DISTRIBUTIONS_VALIDATED",
        "archives": archives,
        "model_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("main", "alfworld", "spreadsheetbench", "skillrl", "mce", "webshop"):
        parser.add_argument(f"{name}_directory", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    packages = (
        Package("main", root, "rethinkskill", args.main_directory),
        Package(
            "alfworld",
            root / "integrations/alfworld",
            "rethinkskill_alfworld",
            args.alfworld_directory,
        ),
        Package(
            "spreadsheetbench",
            root / "integrations/spreadsheetbench",
            "rethinkskill_spreadsheetbench",
            args.spreadsheetbench_directory,
        ),
        Package(
            "skillrl",
            root / "experimental/integrations/skillrl",
            "rethinkskill_skillrl",
            args.skillrl_directory,
        ),
        Package(
            "mce",
            root / "experimental/integrations/mce",
            "rethinkskill_mce",
            args.mce_directory,
        ),
        Package(
            "webshop",
            root / "experimental/integrations/webshop",
            "rethinkskill_webshop",
            args.webshop_directory,
        ),
    )
    try:
        value = verify_distributions(packages)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        value = {
            "schema_version": 1,
            "status": "RETHINKSKILL_DISTRIBUTIONS_INVALID",
            "error": str(exc),
            "model_calls": 0,
        }
        print(json.dumps(value, indent=2, sort_keys=True))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
