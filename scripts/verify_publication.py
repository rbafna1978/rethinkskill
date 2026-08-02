#!/usr/bin/env python3
"""Fail closed on secrets, host paths, or protected evidence in public files."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

FORBIDDEN_NAMES = frozenset({".DS_Store", ".env"})
FORBIDDEN_SUFFIXES = frozenset({".key", ".p12", ".pem"})
PROTECTED_PARTS = frozenset(
    {
        ".local_secrets",
        "archive",
        "final_results",
        "manuscript",
        "research",
        "runs",
    }
)
CONTENT_RULES = (
    (
        "openai_style_secret",
        re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "github_token",
        re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "aws_access_key",
        re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    ),
    (
        "private_key",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "macos_user_path",
        re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
    ),
    (
        "linux_user_path",
        re.compile(rb"/home/[A-Za-z0-9._-]+/"),
    ),
    (
        "windows_user_path",
        re.compile(rb"[A-Za-z]:\\Users\\[^\\\r\n]+\\"),
    ),
)


def _git_publication_files(root: Path) -> tuple[Path, ...] | None:
    if not (root / ".git").exists():
        return None
    completed = subprocess.run(
        (
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError("unable to enumerate Git publication files")
    return tuple(root / raw.decode("utf-8") for raw in completed.stdout.split(b"\0") if raw)


def _publication_files(root: Path) -> tuple[Path, ...]:
    git_files = _git_publication_files(root)
    if git_files is not None:
        return git_files
    return tuple(
        sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
        )
    )


def verify_publication(
    root: Path,
    *,
    files: Iterable[Path] | None = None,
) -> dict[str, object]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"publication root is absent: {resolved}")
    candidates = tuple(files) if files is not None else _publication_files(resolved)
    findings: list[str] = []
    scanned = 0
    for path in candidates:
        candidate = path if path.is_absolute() else resolved / path
        normalized = candidate.parent.resolve() / candidate.name
        try:
            relative = normalized.relative_to(resolved)
        except ValueError:
            findings.append(f"outside_root:{normalized}")
            continue
        if candidate.name in FORBIDDEN_NAMES or candidate.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden_file:{relative.as_posix()}")
            continue
        protected = PROTECTED_PARTS.intersection(relative.parts)
        if protected:
            findings.append(f"protected_path:{relative.as_posix()}:{sorted(protected)[0]}")
            continue
        if candidate.is_symlink():
            findings.append(f"symlink:{relative.as_posix()}")
            continue
        if not candidate.is_file():
            findings.append(f"missing_tracked_file:{relative.as_posix()}")
            continue
        payload = candidate.read_bytes()
        scanned += 1
        if b"\0" in payload:
            continue
        for name, pattern in CONTENT_RULES:
            if pattern.search(payload):
                findings.append(f"{name}:{relative.as_posix()}")
    if findings:
        raise ValueError("publication surface rejected: " + ", ".join(sorted(findings)))
    return {
        "schema_version": 1,
        "status": "RETHINKSKILL_PUBLICATION_SURFACE_VALIDATED",
        "root": str(resolved),
        "files_scanned": scanned,
        "model_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path.cwd(),
    )
    args = parser.parse_args()
    try:
        report = verify_publication(args.root)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "RETHINKSKILL_PUBLICATION_SURFACE_INVALID",
                    "error": str(exc),
                    "model_calls": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
