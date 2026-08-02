"""RethinkSkill cli parser."""

from __future__ import annotations

import argparse
from pathlib import Path

from rethinkskill.providers.transport import TransportKind

_REASONING_EFFORTS = ("", "low", "medium", "high", "xhigh", "max")

_CLAUDE_EFFORTS = ("low", "medium", "high", "max")


def add_native_transport_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transport",
        default=TransportKind.CODEX.value,
        metavar="PROVIDER",
        help="Built-in or explicitly loaded provider registration name.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", choices=_REASONING_EFFORTS, default="medium")
    parser.add_argument("--launcher")
    parser.add_argument("--api-base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--claude-base-url")
    parser.add_argument("--claude-auth-token-env")
    parser.add_argument("--claude-effort", choices=_CLAUDE_EFFORTS, default="medium")


def add_named_native_transport_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    option = prefix.replace("_", "-")
    parser.add_argument(
        f"--{option}-transport",
        default=TransportKind.CODEX.value,
        metavar="PROVIDER",
        help="Built-in or explicitly loaded provider registration name.",
    )
    parser.add_argument(f"--{option}-model", required=True)
    parser.add_argument(
        f"--{option}-reasoning-effort", choices=_REASONING_EFFORTS, default="medium"
    )
    parser.add_argument(f"--{option}-launcher")
    parser.add_argument(f"--{option}-api-base-url")
    parser.add_argument(f"--{option}-api-key-env")
    parser.add_argument(f"--{option}-claude-base-url")
    parser.add_argument(f"--{option}-claude-auth-token-env")
    parser.add_argument(f"--{option}-claude-effort", choices=_CLAUDE_EFFORTS, default="medium")


def add_catalog_commands(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser(
        "benchmark-catalog",
        help="List benchmark scoring, execution, ownership, and fidelity support.",
    )
    subparsers.add_parser(
        "provider-catalog", help="List native model providers with zero model calls."
    )
    provider_preflight = subparsers.add_parser(
        "provider-preflight",
        help="Run configured built-in provider readiness probes with zero model calls.",
    )
    add_native_transport_arguments(provider_preflight)
    provider_preflight.add_argument(
        "--authorize-provider-probes",
        action="store_true",
        help="Authorize launcher/configuration probes, never model inference.",
    )
    subparsers.add_parser(
        "optimizer-catalog",
        help="List centrally reviewed optimizer strategies with zero model calls.",
    )
    protocol = subparsers.add_parser(
        "plan-protocol", help="Render the provider-independent SkillOpt-Lite stage plan."
    )
    protocol.add_argument("--benchmark", required=True)
    protocol.add_argument(
        "--protocol", type=Path, default=Path("configs/protocols/skillopt_lite.json")
    )


def add_result_commands(subparsers: argparse._SubParsersAction) -> None:
    validate_run = subparsers.add_parser(
        "validate-run", help="Replay terminal run evidence with zero model calls."
    )
    validate_run.add_argument("run_root", type=Path)
    validate = subparsers.add_parser(
        "validate-results", help="Validate a results.jsonl ledger without modifying it."
    )
    validate.add_argument("results", type=Path)
    validate.add_argument("--expected-ids", type=Path)
    recovery = subparsers.add_parser(
        "recovery-plan", help="List only confirmed infrastructure-invalid IDs; perform no mutation."
    )
    recovery.add_argument("results", type=Path)
    recovery.add_argument("--expected-ids", type=Path)
    evaluate = subparsers.add_parser(
        "evaluate-cases", help="Run deterministic benchmark verifiers with zero model calls."
    )
    evaluate.add_argument("--benchmark", required=True)
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    validate_evaluation = subparsers.add_parser(
        "validate-evaluation",
        help="Replay deterministic evaluation evidence with zero model calls.",
    )
    validate_evaluation.add_argument("manifest", type=Path)


def add_native_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=900)


def add_native_evolution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="val")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--train-ids-file", type=Path)
    parser.add_argument("--validation-ids-file", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--arm", choices=("normal", "fail_only", "success_only"), default="normal")
    parser.add_argument("--hard-dead-band", type=float, default=0.01)
    parser.add_argument("--soft-rescue-delta", type=float)
    parser.add_argument("--max-skill-chars", type=int, default=80000)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--optimizer-strategy", default="model-skill")
    add_named_native_transport_arguments(parser, "target")
    add_named_native_transport_arguments(parser, "optimizer")


def add_native_execution_commands(subparsers: argparse._SubParsersAction) -> None:
    native_preflight = subparsers.add_parser(
        "native-preflight", help="Resolve native dataset tasks without model calls."
    )
    add_native_run_arguments(native_preflight)
    native_run = subparsers.add_parser(
        "native-run", help="Execute one benchmark through its native harness."
    )
    add_native_run_arguments(native_run)
    add_native_transport_arguments(native_run)
    native_run.add_argument("--authorize-model-calls", action="store_true")


def add_native_evolution_commands(subparsers: argparse._SubParsersAction) -> None:
    native_evolution_preflight = subparsers.add_parser(
        "native-evolution-preflight",
        help="Freeze native evolution inputs and transports with zero model calls.",
    )
    add_native_evolution_arguments(native_evolution_preflight)
    native_evolve = subparsers.add_parser(
        "native-evolve", help="Run bounded skill evolution through native benchmark harnesses."
    )
    add_native_evolution_arguments(native_evolve)
    native_evolve.add_argument("--authorize-model-calls", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rethinkskill",
        description="Research code for controlled multi-round agent-skill evolution.",
    )
    parser.add_argument("--repository", type=Path)
    parser.add_argument(
        "--load-benchmark-plugins",
        action="store_true",
        help="Explicitly import installed rethinkskill.benchmarks entry points.",
    )
    parser.add_argument(
        "--load-harness-plugins",
        action="store_true",
        help="Explicitly import installed rethinkskill.harnesses entry points.",
    )
    parser.add_argument(
        "--load-provider-plugins",
        action="store_true",
        help="Explicitly import installed rethinkskill.providers entry points.",
    )
    parser.add_argument(
        "--load-optimizer-plugins",
        action="store_true",
        help="Explicitly import installed rethinkskill.optimizers entry points.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_catalog_commands(subparsers)
    add_native_execution_commands(subparsers)
    add_native_evolution_commands(subparsers)
    add_result_commands(subparsers)
    return parser


def parser_command_names(parser: argparse.ArgumentParser | None = None) -> tuple[str, ...]:
    """Return public subcommands in their declared order for routing tests."""
    selected = parser or build_parser()
    for action in selected._actions:
        if isinstance(action, argparse._SubParsersAction):
            return tuple(action.choices)
    return ()
