# Contributing to RethinkSkill

## Scope

Contributions should preserve the separation between maintained source code
and experimental evidence. Do not commit formal results, raw trajectories,
model conversations, ledgers, receipts, manuscript files, credentials, or
local run outputs.

Benchmark and provider extensions must use the public registries and explicit
plugin-loading boundaries. Optional benchmark environments belong in
independently installable packages under `integrations/`; the core wheel must
remain importable without them.

## Development checks

Install the development environment and run the zero-model checks:

```bash
python -m pip install -e ".[dev,spreadsheet]"
python -m pytest
ruff check src tests integrations experimental scripts
python scripts/verify_publication.py
```

For release-facing changes, also build every distribution and run:

```bash
python scripts/verify_distributions.py \
  dist/main dist/alfworld dist/spreadsheetbench dist/skillrl dist/mce \
  dist/webshop
python scripts/verify_optional_integrations.py \
  dist/main dist/alfworld dist/spreadsheetbench dist/skillrl dist/mce \
  dist/webshop
```

Tests must not contact models, external APIs, or mutate formal artifact trees.
Tests for installed external environments must skip clearly when those
environments are absent.

## Provenance requirements

Any new execution path must freeze its effective inputs before execution,
validate the frozen plan immediately before the side effect, record stable
hashes and call accounting, and fail closed on drift. Historical evidence must
remain historical and must not be rewritten or reinterpreted as a new result.
