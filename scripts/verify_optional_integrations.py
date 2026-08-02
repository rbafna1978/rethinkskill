#!/usr/bin/env python3
"""Install built wheels in isolation and validate their plugin registrations."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CORE_CHECK = r"""
import json
import os
from pathlib import Path
import rethinkskill
from rethinkskill.benchmarks.capabilities import capability_catalog
from rethinkskill.evolution.optimizer import optimizer_catalog
from rethinkskill.providers.catalog import provider_catalog

site = Path(os.environ["RETHINKSKILL_WHEEL_SITE"]).resolve()
assert Path(rethinkskill.__file__).resolve().is_relative_to(site)
capabilities = capability_catalog().manifest()
assert capabilities["counts"] == {
    "total": 20,
    "deterministic_scorer": 5,
    "adapter_available": 4,
    "runtime_ready": 4,
    "native_runnable": 4,
    "scoring_only": 1,
    "deterministic_only": 1,
    "external_adapter": 0,
    "external_only": 3,
    "declared_only": 12,
    "incomplete_adapter": 0,
    "unavailable": 15,
}, capabilities["counts"]
assert provider_catalog().manifest()["count"] == 4
assert optimizer_catalog().manifest()["count"] == 1
print(json.dumps(capabilities["counts"], sort_keys=True))
"""


PLUGIN_CHECK = r"""
import json
import os
from pathlib import Path
import rethinkskill
import rethinkskill_alfworld
import rethinkskill_mce
import rethinkskill_skillrl
import rethinkskill_spreadsheetbench
import rethinkskill_webshop
from rethinkskill.benchmarks.capabilities import capability_catalog
from rethinkskill_alfworld.harness import AlfWorldHarness
from rethinkskill_mce.plugin import BENCHMARKS as MCE_BENCHMARKS
from rethinkskill_skillrl.plugin import BENCHMARKS as SKILLRL_BENCHMARKS
from rethinkskill_spreadsheetbench.harness import SpreadsheetBenchHarness
from rethinkskill_webshop.harness import WebShopHarness

site = Path(os.environ["RETHINKSKILL_WHEEL_SITE"]).resolve()
for package in (
    rethinkskill,
    rethinkskill_alfworld,
    rethinkskill_mce,
    rethinkskill_skillrl,
    rethinkskill_spreadsheetbench,
    rethinkskill_webshop,
):
    assert Path(package.__file__).resolve().is_relative_to(site), package.__file__
value = capability_catalog(
    load_benchmark_plugins=True,
    load_harness_plugins=True,
).manifest()
counts = value["counts"]
assert counts["total"] == 20, counts
assert counts["deterministic_scorer"] == 17, counts
assert counts["adapter_available"] == 19, counts
assert counts["external_adapter"] == 2, counts
assert counts["external_only"] == 1, counts
assert counts["declared_only"] == 0, counts
assert counts["runtime_ready"] == 16, counts
assert counts["native_runnable"] == 16, counts
assert counts["scoring_only"] == 0, counts
assert counts["deterministic_only"] == 0, counts
assert counts["unavailable"] == 1, counts
for name in (*SKILLRL_BENCHMARKS, *MCE_BENCHMARKS, "spreadsheetbench", "alfworld", "webshop"):
    capability = next(item for item in value["capabilities"] if item["name"] == name)
    assert capability["native_registration"]["source"] == "entry_point", capability
assert AlfWorldHarness.__module__ == "rethinkskill_alfworld.harness"
assert SpreadsheetBenchHarness.__module__ == "rethinkskill_spreadsheetbench.harness"
assert WebShopHarness.__module__ == "rethinkskill_webshop.harness"
print(json.dumps(counts, sort_keys=True))
"""


def _single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {directory}; found {len(matches)}")
    return matches[0]


def _run(
    python: Path,
    *arguments: str,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> str:
    completed = subprocess.run(
        (str(python), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=cwd,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def verify_optional_integrations(directories: tuple[Path, ...]) -> dict[str, object]:
    patterns = (
        "rethinkskill-*.whl",
        "rethinkskill_alfworld-*.whl",
        "rethinkskill_spreadsheetbench-*.whl",
        "rethinkskill_skillrl-*.whl",
        "rethinkskill_mce-*.whl",
        "rethinkskill_webshop-*.whl",
    )
    wheels = tuple(
        _single(directory, pattern)
        for directory, pattern in zip(directories, patterns, strict=True)
    )
    with tempfile.TemporaryDirectory(prefix="rethinkskill-wheel-check-") as temporary:
        work = Path(temporary)
        site = work / "site-packages"
        python = Path(sys.executable).resolve()
        _run(
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            "--target",
            str(site),
            str(wheels[0]),
        )
        child_environment = {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(site),
            "RETHINKSKILL_WHEEL_SITE": str(site),
        }
        for name in ("LANG", "LC_ALL", "PATH", "SYSTEMROOT"):
            if name in os.environ:
                child_environment[name] = os.environ[name]
        core = json.loads(
            _run(
                python,
                "-S",
                "-c",
                CORE_CHECK,
                cwd=work,
                environment=child_environment,
            )
        )
        _run(
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            "--target",
            str(site),
            *(str(wheel) for wheel in wheels[1:]),
        )
        plugins = json.loads(
            _run(
                python,
                "-S",
                "-c",
                PLUGIN_CHECK,
                cwd=work,
                environment=child_environment,
            )
        )
    return {
        "schema_version": 1,
        "status": "RETHINKSKILL_OPTIONAL_INTEGRATIONS_VALIDATED",
        "wheels": [str(wheel) for wheel in wheels],
        "core_counts": core,
        "plugin_counts": plugins,
        "model_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("main", "alfworld", "spreadsheetbench", "skillrl", "mce", "webshop"):
        parser.add_argument(f"{name}_directory", type=Path)
    args = parser.parse_args()
    names = ("main", "alfworld", "spreadsheetbench", "skillrl", "mce", "webshop")
    directories = tuple(getattr(args, f"{name}_directory") for name in names)
    try:
        value = verify_optional_integrations(directories)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "RETHINKSKILL_OPTIONAL_INTEGRATIONS_INVALID",
                    "error": str(exc),
                    "model_calls": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
