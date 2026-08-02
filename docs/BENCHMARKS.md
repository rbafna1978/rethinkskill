# Benchmark support

The catalog declares 20 benchmark capabilities. Six correspond to the paper
release; fourteen are optional external extensions.

## Paper release

| Benchmark | Scoring | Execution |
| --- | --- | --- |
| `searchqa` | Exact match, token F1, substring | Built-in QA harness |
| `officeqa` | Answer match | Built-in document QA harness |
| `docvqa` | Answer match | Built-in visual QA harness |
| `livemath` | Multiple-choice accuracy | Built-in seeded-choice harness |
| `spreadsheetbench` | Answer-cell comparison | `rethinkskill-spreadsheetbench` package |
| `alfworld` | Terminal environment success | `rethinkskill-alfworld` package |

The first five are the primary benchmarks reported in the paper. ALFWorld is
reported as an appendix experiment. Dataset files and licensed environment
assets are not redistributed by this repository.

## External extensions

| Package or boundary | Benchmarks | Support |
| --- | --- | --- |
| SkillRL | `nq`, `triviaqa`, `popqa`, `hotpotqa`, `2wiki`, `musique`, `bamboogle` | Optional offline QA scorer and harness |
| MCE | `finer`, `uspto50k`, `symptom2disease`, `lawbench-charge`, `aegis2` | Optional file-backed scorer and harness |
| WebShop | `webshop` | Optional bridge to a separately installed official runtime |
| MCP-Atlas | `mcp-atlas` | Declared boundary only; no runnable adapter |

These packages demonstrate extensibility. Their presence does not claim that
the paper evaluated all 20 benchmarks or that an offline adapter reproduces an
upstream system's complete experimental environment.

## Capability terms

- `deterministic_scorer`: frozen responses can be scored without model calls.
- `adapter_available`: a dataset-to-task or environment harness is installed.
- `runtime_ready`: required local files and dependencies pass readiness checks.
- `native_runnable`: both an adapter and its required runtime are available.
- `scoring_only`: scoring is available but task execution is not.
- `external_only`: the benchmark is intentionally represented only as an
  external boundary.

Inspect the current machine rather than inferring support from package names:

```bash
rethinkskill benchmark-catalog
rethinkskill --load-benchmark-plugins --load-harness-plugins benchmark-catalog
```

## Deterministic evaluation

`evaluate-cases` consumes immutable JSONL rows with unique `case_id` values.
It snapshots the input and any scorer-declared files, writes deterministic
verdicts, and records a manifest. `validate-evaluation` replays that manifest
without contacting a model.

Environment benchmarks use harness-owned terminal outcomes instead of
pretending that a frozen text response is an authoritative score.

## Adding a benchmark

A benchmark extension provides a `BenchmarkSpec`, an optional harness, and
entry-point registration. Scoring and execution are registered separately so
that a scorer never implies that the corresponding dataset or environment is
installed. See the small packages in `experimental/integrations/` for working
examples.
