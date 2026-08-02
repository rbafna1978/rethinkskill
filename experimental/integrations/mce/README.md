# Experimental integration: rethinkskill-mce

This package is an opt-in external extension example. It is catalogued with
`tier=external` and is not part of the six-benchmark core research release.

Optional deterministic scorers and file-backed harnesses for the five Meta
Context Engineering capabilities declared by rethinkskill: FinER, USPTO-50K,
Symptom2Disease, LawBench-Charge, and AEGIS2.

This package evaluates frozen responses against supplied labels. It does not
bundle official datasets, reproduce official preprocessing, canonicalize
molecules, or claim paper-result reproduction. Dataset preparation and source
identity remain part of the experiment provenance.

Synthetic one-row transport fixtures are kept under `examples/live_smoke/`.
They validate adapter wiring only and are not official benchmark data or
research results.

Install the core and this package, then explicitly opt in to both plugin groups:

```python
from rethinkskill.benchmarks.capabilities import capability_catalog

catalog = capability_catalog(
    load_benchmark_plugins=True,
    load_harness_plugins=True,
)
```

Plugin loading is explicit. A core-only installation retains all five names as
`declared_only` capabilities and reports their scorer and harness as absent.
