"""RethinkSkill cli commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from rethinkskill.benchmarks.capabilities import capability_catalog
from rethinkskill.errors import AuthorizationRequiredError, ConfigurationError
from rethinkskill.evaluation.core import evaluate_cases
from rethinkskill.evaluation.results import load_ids, recovery_plan, validate_ledger
from rethinkskill.evidence.evaluation import validate_evaluation_evidence
from rethinkskill.evidence.run import validate_run_evidence
from rethinkskill.evolution.loop import EvolutionOptions, execute_evolution, plan_evolution
from rethinkskill.evolution.native import (
    NativeEvaluationSelection,
    NativeSkillEvaluator,
    freeze_native_evolution_inputs,
)
from rethinkskill.evolution.optimizer import optimizer_catalog
from rethinkskill.evolution.protocol import GatePolicy, ProtocolSpec
from rethinkskill.providers.catalog import ProviderCatalog, prepare_executor, provider_catalog
from rethinkskill.providers.transport import TransportConfig
from rethinkskill.runtime.runner import NativeRunOptions, execute_native_plan, plan_native_run
from rethinkskill.utils.fs import Repository
from rethinkskill.utils.serde import thaw_json_mapping


@dataclass(frozen=True, slots=True)
class CommandResult:
    """One JSON-serializable command payload plus its process exit code."""

    payload: object
    exit_code: int = 0

    def validate(self) -> None:
        if type(self.exit_code) is not int or self.exit_code < 0 or self.exit_code > 255:
            raise ConfigurationError(f"command exit code must be in [0, 255]: {self.exit_code!r}")


CommandHandler = Callable[[argparse.Namespace], CommandResult]


class CommandRouter:
    """Fail-closed, side-effect-free mapping from command names to handlers."""

    def __init__(self, handlers: Iterable[tuple[str, CommandHandler]] = ()):
        self._handlers: dict[str, CommandHandler] = {}
        for name, handler in handlers:
            self.register(name, handler)

    def register(self, name: str, handler: CommandHandler) -> None:
        if not name or name != name.strip() or any(character.isspace() for character in name):
            raise ConfigurationError(f"invalid CLI command registration: {name!r}")
        if not callable(handler):
            raise ConfigurationError(f"CLI command handler is not callable: {name}")
        if name in self._handlers:
            raise ConfigurationError(f"duplicate CLI command registration: {name}")
        self._handlers[name] = handler

    def names(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def dispatch(self, args: argparse.Namespace) -> CommandResult:
        command = getattr(args, "command", None)
        try:
            handler = self._handlers[command]
        except KeyError as exc:
            raise ConfigurationError(f"no CLI command handler registered for {command!r}") from exc
        result = handler(args)
        if type(result) is not CommandResult:
            raise ConfigurationError(f"CLI command handler returned an invalid result: {command}")
        result.validate()
        return result


def native_transport(args: argparse.Namespace, *, catalog: ProviderCatalog) -> TransportConfig:
    """Build the maintained native provider contract."""
    spec = catalog.resolve(args.transport)
    config = TransportConfig(
        kind=spec.kind,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        launcher=args.launcher or spec.default_launcher,
        sandbox="read-only",
        api_base_url=args.api_base_url,
        api_key_env=args.api_key_env,
        claude_effort=args.claude_effort,
        claude_base_url=args.claude_base_url,
        claude_auth_token_env=args.claude_auth_token_env,
        provider=None if spec.identifier == spec.kind.value else spec.identifier,
    )
    catalog.validate_config(config, role="target")
    return config


def named_native_transport(
    args: argparse.Namespace, prefix: str, *, catalog: ProviderCatalog
) -> TransportConfig:
    spec = catalog.resolve(getattr(args, f"{prefix}_transport"))
    config = TransportConfig(
        kind=spec.kind,
        model=getattr(args, f"{prefix}_model"),
        reasoning_effort=getattr(args, f"{prefix}_reasoning_effort"),
        launcher=getattr(args, f"{prefix}_launcher") or spec.default_launcher,
        sandbox="read-only",
        api_base_url=getattr(args, f"{prefix}_api_base_url"),
        api_key_env=getattr(args, f"{prefix}_api_key_env"),
        claude_effort=getattr(args, f"{prefix}_claude_effort"),
        claude_base_url=getattr(args, f"{prefix}_claude_base_url"),
        claude_auth_token_env=getattr(args, f"{prefix}_claude_auth_token_env"),
        provider=None if spec.identifier == spec.kind.value else spec.identifier,
    )
    catalog.validate_config(config, role=prefix)
    return config


def validate_run_command(args: argparse.Namespace) -> CommandResult:
    return CommandResult(validate_run_evidence(args.run_root))


def command_router() -> CommandRouter:
    """Return all public CLI commands in parser order."""
    return CommandRouter(
        (
            ("benchmark-catalog", benchmark_catalog_command),
            ("provider-catalog", provider_catalog_command),
            ("provider-preflight", provider_preflight_command),
            ("optimizer-catalog", optimizer_catalog_command),
            ("plan-protocol", protocol_plan_command),
            ("native-preflight", native_command),
            ("native-run", native_command),
            ("native-evolution-preflight", native_evolution_command),
            ("native-evolve", native_evolution_command),
            ("validate-run", validate_run_command),
            ("validate-results", results_command),
            ("recovery-plan", results_command),
            ("evaluate-cases", evaluate_cases_command),
            ("validate-evaluation", validate_evaluation_command),
        )
    )


def _capabilities(args: argparse.Namespace):
    return capability_catalog(
        load_benchmark_plugins=args.load_benchmark_plugins,
        load_harness_plugins=args.load_harness_plugins,
    )


def evaluate_cases_command(args: argparse.Namespace) -> CommandResult:
    repository = Repository.discover(args.repository)
    run_root = repository.resolve_relative("runs", must_exist=False)
    output = repository.validate_new_output(args.output, run_root)
    return CommandResult(
        evaluate_cases(
            benchmark=args.benchmark,
            input_path=args.input.expanduser().resolve(),
            output_path=output,
            catalog=_capabilities(args),
        )
    )


def validate_evaluation_command(args: argparse.Namespace) -> CommandResult:
    return CommandResult(validate_evaluation_evidence(args.manifest, catalog=_capabilities(args)))


def benchmark_catalog_command(args: argparse.Namespace) -> CommandResult:
    return CommandResult(_capabilities(args).manifest())


def provider_catalog_command(args: argparse.Namespace) -> CommandResult:
    return CommandResult(
        provider_catalog(load_plugins=getattr(args, "load_provider_plugins", False)).manifest()
    )


def provider_preflight_command(args: argparse.Namespace) -> CommandResult:
    """Prepare one built-in provider and publish bounded zero-call readiness."""
    if args.authorize_provider_probes is not True:
        raise AuthorizationRequiredError("provider-preflight requires --authorize-provider-probes")
    providers = provider_catalog(load_plugins=getattr(args, "load_provider_plugins", False))
    config = native_transport(args, catalog=providers)
    spec = providers.resolve(config.provider_name)
    if spec.identifier != spec.kind.value:
        raise ConfigurationError("provider-preflight accepts built-in provider registrations only")
    executor = prepare_executor(config, authorized=True, catalog=providers, role="target")
    scopes = {
        "codex": "launcher_flags",
        "claude-code": "launcher_flags",
        "gemini-cli": "launcher_flags",
        "openai-compatible": "configuration_and_credential_presence",
    }
    return CommandResult(
        {
            "schema_version": 1,
            "status": "RETHINKSKILL_PROVIDER_PREFLIGHT_READY",
            "provider": spec.identifier,
            "transport_kind": spec.kind.value,
            "interface": spec.interface.value,
            "readiness_scope": scopes[spec.kind.value],
            "model_availability_verified": False,
            "endpoint_connectivity_verified": False,
            "authorization_scope": "provider_readiness_probes_only",
            "executor": executor.public_manifest(),
            "model_calls": 0,
        }
    )


def optimizer_catalog_command(args: argparse.Namespace) -> CommandResult:
    from rethinkskill.evolution.optimizer import optimizer_catalog

    return CommandResult(
        optimizer_catalog(load_plugins=getattr(args, "load_optimizer_plugins", False)).manifest()
    )


def protocol_plan_command(args: argparse.Namespace) -> CommandResult:
    repository = Repository.discover(args.repository)
    source = args.protocol
    if not source.is_absolute():
        source = repository.root / source
    return CommandResult(ProtocolSpec.load(source.resolve()).plan(args.benchmark))


def results_command(args: argparse.Namespace) -> CommandResult:
    expected = load_ids(args.expected_ids) if args.expected_ids else None
    report = validate_ledger(args.results, expected_ids=expected)
    if args.command == "recovery-plan":
        return CommandResult(recovery_plan(report))
    return CommandResult(report.to_dict(), 0 if report.status == "STRUCTURALLY_VALID" else 2)


_ARM_FEEDBACK = {
    "normal": ("failure", "success"),
    "fail_only": ("failure",),
    "success_only": ("success",),
}


def native_evolution_command(args: argparse.Namespace) -> CommandResult:
    repository = Repository.discover(args.repository)
    providers = provider_catalog(load_plugins=args.load_provider_plugins)
    target_config = named_native_transport(args, "target", catalog=providers)
    optimizer_config = named_native_transport(args, "optimizer", catalog=providers)
    optimizers = optimizer_catalog(load_plugins=args.load_optimizer_plugins)
    optimizer_selection = optimizers.selection_manifest(args.optimizer_strategy)
    catalog = capability_catalog(
        load_benchmark_plugins=args.load_benchmark_plugins,
        load_harness_plugins=args.load_harness_plugins,
    )
    categories = _ARM_FEEDBACK[args.arm]
    evolution_plan = plan_evolution(
        repository=repository,
        benchmark=args.benchmark,
        initial_skill=args.skill,
        output_root=args.out_root,
        gate=GatePolicy(
            hard_dead_band=args.hard_dead_band, soft_rescue_delta=args.soft_rescue_delta
        ),
        options=EvolutionOptions(
            rounds=args.rounds,
            arm=args.arm,
            feedback_categories=categories,
            train_split="train",
            validation_split="validation",
            max_skill_chars=args.max_skill_chars,
        ),
    )
    frozen = freeze_native_evolution_inputs(
        repository=repository,
        catalog=catalog,
        benchmark=args.benchmark,
        dataset=args.dataset,
        initial_skill=args.skill,
        output_root=evolution_plan.output_root,
        selections={
            "train": NativeEvaluationSelection(
                split=args.train_split, limit=args.train_limit, ids_file=args.train_ids_file
            ),
            "validation": NativeEvaluationSelection(
                split=args.validation_split,
                limit=args.validation_limit,
                ids_file=args.validation_ids_file,
            ),
        },
        seed=args.seed,
        asset_root=args.asset_root,
        timeout_seconds=args.timeout_seconds,
    )
    if args.command == "native-evolution-preflight":
        providers.validate_config(target_config, role="target")
        providers.validate_config(optimizer_config, role="optimizer")
        return CommandResult(
            {
                **thaw_json_mapping(evolution_plan.preflight),
                "status": "RETHINKSKILL_NATIVE_EVOLUTION_PREFLIGHT_PASS",
                "native_inputs": thaw_json_mapping(frozen.manifest),
                "target_transport": target_config.public_manifest(),
                "optimizer_transport": optimizer_config.public_manifest(),
                "optimizer_strategy": optimizer_selection,
            }
        )
    target_executor = prepare_executor(
        target_config, authorized=args.authorize_model_calls, catalog=providers, role="target"
    )
    optimizer_executor = prepare_executor(
        optimizer_config, authorized=args.authorize_model_calls, catalog=providers, role="optimizer"
    )
    evaluator = NativeSkillEvaluator(
        repository=repository,
        catalog=catalog,
        frozen=frozen,
        output_root=evolution_plan.output_root,
        executor=target_executor,
        timeout_seconds=args.timeout_seconds,
    )
    optimizer = optimizers.build(
        args.optimizer_strategy,
        executor=optimizer_executor,
        output_root=evolution_plan.output_root,
        timeout_seconds=args.timeout_seconds,
    )
    receipt = execute_evolution(
        evolution_plan,
        evaluator=evaluator,
        optimizer=optimizer,
        authorized=args.authorize_model_calls,
    )
    return CommandResult(
        receipt, 0 if receipt["status"] == "RETHINKSKILL_EVOLUTION_VALIDATED" else 1
    )


def native_command(args: argparse.Namespace) -> CommandResult:
    repository = Repository.discover(args.repository)
    providers = provider_catalog(load_plugins=args.load_provider_plugins)
    transport = native_transport(args, catalog=providers) if args.command == "native-run" else None
    catalog = capability_catalog(
        load_benchmark_plugins=args.load_benchmark_plugins,
        load_harness_plugins=args.load_harness_plugins,
    )
    plan = plan_native_run(
        repository=repository,
        catalog=catalog,
        benchmark=args.benchmark,
        dataset=args.dataset,
        skill=args.skill,
        output_root=args.out_root,
        options=NativeRunOptions(
            split=args.split,
            limit=args.limit,
            ids_file=args.ids_file,
            seed=args.seed,
            asset_root=args.asset_root,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    if args.command == "native-preflight":
        return CommandResult(plan.preflight)
    assert transport is not None
    executor = prepare_executor(
        transport, authorized=args.authorize_model_calls, catalog=providers, role="target"
    )
    receipt = execute_native_plan(plan, executor=executor, authorized=args.authorize_model_calls)
    return CommandResult(
        receipt, 0 if receipt["status"] == "RETHINKSKILL_NATIVE_EXECUTION_VALIDATED" else 1
    )
