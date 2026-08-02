# Live provider smoke fixtures

These synthetic one-item datasets validate native provider, harness, parser,
scorer, ledger, and receipt integration. They are not official benchmark
splits and must never be reported as research results.

The `qa/` fixture exercises the core context-grounded QA path. Experimental
suite fixtures live with their owning package under `experimental/` so they
are not mistaken for core benchmark examples.

`alfworld/` contains only a one-record manifest pointing to a standard
relative path inside a separately installed ALFWorld corpus. It does not
redistribute the game, trajectory, PDDL, or grammar assets. Supply that corpus
with `--asset-root`; the fixture is suitable for `native-preflight` and a
strictly bounded environment smoke, not a benchmark result.

Before authorizing any synthetic live run, probe the exact provider through
the same catalog/configuration path used by `native-run`:

```bash
rethinkskill provider-preflight \
  --transport openai-compatible \
  --model MODEL \
  --api-base-url https://provider.example/v1 \
  --api-key-env PROVIDER_API_KEY \
  --authorize-provider-probes

rethinkskill provider-preflight \
  --transport codex --model MODEL --authorize-provider-probes

rethinkskill provider-preflight \
  --transport claude-code --model MODEL --authorize-provider-probes

rethinkskill provider-preflight \
  --transport gemini-cli --model MODEL --authorize-provider-probes
```

These probes make zero inference calls. They do not prove endpoint
connectivity or model availability. A live smoke must use one synthetic row,
`--limit 1`, a fresh `--out-root`, and `--authorize-model-calls`; its receipt
is transport evidence, never a benchmark result.
