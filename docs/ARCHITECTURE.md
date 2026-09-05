# Architecture

RethinkSkill is organized by research concepts. The maintained Python package
has eight public areas:

```text
src/rethinkskill/
├── benchmarks/    benchmark declarations, loaders, harnesses, and scorers
├── providers/     OpenAI-compatible API, Codex, Claude Code, and Gemini CLI
├── runtime/       task freezing, workspaces, execution, and call accounting
├── evolution/     feedback views, validation gates, and round transitions
├── evaluation/    deterministic case evaluation and result ledgers
├── evidence/      offline validation of run, evaluation, and evolution outputs
├── integrations/  built-in and external capability declarations
├── utils/         file snapshots, serialization, process, filesystem, and plugins
└── cli/           the `rethinkskill` command-line interface
```

The two paper-specific environment packages live in `integrations/`:

- `rethinkskill-spreadsheetbench` owns workbook execution.
- `rethinkskill-alfworld` owns the ALFWorld environment harness.

Optional examples under `experimental/integrations/` demonstrate how the same
interfaces can be extended for SkillRL, MCE, WebShop, and MCP-Atlas without
making them part of the six-benchmark paper release.

## Execution flow

```text
configuration
    ↓
benchmark + provider selection
    ↓
zero-call preflight and input freezing
    ↓ explicit --authorize-model-calls
task execution → deterministic or harness-owned evaluation
    ↓
results ledger + receipt
    ↓
offline evidence validation
```

`native-preflight` resolves the selected benchmark, dataset, skill, task IDs,
provider configuration, and output location without contacting a model. The
live command repeats the frozen-input checks immediately before execution and
requires explicit authorization. A run writes only to a new output directory.

## Evolution flow

All three feedback arms use the same evolution loop:

```text
current skill → train outcomes → feedback view → candidate proposal
                                             ↓
                                  validation gate
                                             ↓
                            next skill + best checkpoint
```

The feedback view is `normal`, `fail_only`, or `success_only`. Candidate
acceptance determines the next-round skill; strict best-checkpoint selection
is recorded separately. The optimizer and target can use different providers
through the same provider contract.

## Catalogs and plugins

The benchmark catalog joins four independent facts: declaration, deterministic
scorer, execution harness, and runtime readiness. The provider and optimizer
catalogs use the same explicit-registration pattern. Installed plugins are not
loaded unless the corresponding `--load-*-plugins` flag is supplied.

Third-party packages register entry points in one or more groups:

```toml
[project.entry-points."rethinkskill.benchmarks"]
example = "example_package:BENCHMARKS"

[project.entry-points."rethinkskill.harnesses"]
example = "example_package:HARNESSES"
```

Duplicate names, incomplete registrations, and mismatched ownership metadata
fail closed.

## Evidence boundary

Configuration, execution outputs, and scientific evidence are separate. This
repository contains code and synthetic test fixtures. Formal trajectories,
results, ledgers, receipts, and manuscript sources are not part of the public
Git tree and are never reconstructed by the validation commands.
