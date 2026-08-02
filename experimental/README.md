# Experimental extensions

This directory contains opt-in integrations that demonstrate how
`rethinkskill` can be extended beyond the six benchmarks used by the core
research release.

Experimental integrations are catalogued with `tier=external`. They are not
installed with the core package, are not
part of the core benchmark support claim, and must not be interpreted as
reproductions of upstream paper results. Each implemented package uses the
same public benchmark and harness entry-point contracts as an independent
third-party extension.

| Integration | State | What is implemented | What is not claimed |
| --- | --- | --- | --- |
| [`skillrl`](integrations/skillrl/) | External offline adapter | Seven deterministic QA scorers and context-grounded file harnesses | SkillRL live retrieval, training, or paper-result reproduction |
| [`mce`](integrations/mce/) | External dataset-bound adapter | Five deterministic prediction scorers and file-backed harnesses | Official preprocessing, bundled datasets, or paper-result reproduction |
| [`webshop`](integrations/webshop/) | External-runtime adapter | Bounded bridge to a separately installed official WebShop environment | Bundled WebShop assets or an out-of-the-box official run |
| [`mcp-atlas`](integrations/mcp-atlas/) | External design only | Explicit external capability boundary and future adapter requirements | A runnable adapter or local scorer |

## Extension contract

An extension package may register benchmark scorers through the
`rethinkskill.benchmarks` entry-point group and native harnesses through
`rethinkskill.harnesses`. Plugin loading is explicit:

```bash
rethinkskill --load-benchmark-plugins --load-harness-plugins benchmark-catalog
```

The core remains usable without this directory. Experimental packages are
built and tested independently so that their dependencies and fidelity claims
do not silently expand the core installation.

## Development installation

Install the core first, then select only the extension being studied:

```bash
python -m pip install -e .
python -m pip install -e experimental/integrations/skillrl
```

Formal use must additionally freeze the upstream dataset or environment,
selected task IDs, package versions, and runtime evidence required by the
integration-specific README.
