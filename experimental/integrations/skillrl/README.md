# Experimental integration: rethinkskill-skillrl

This package is an opt-in external extension example. It is catalogued with
`tier=external` and is not part of the six-benchmark core research release.

Optional deterministic scorer and context-grounded file harness for the seven
QA capabilities declared by the `skillrl-offline` integration in rethinkskill:
NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, and Bamboogle.

This package evaluates frozen responses against supplied aliases and can run
file-backed, context-grounded QA tasks. It does **not** reproduce SkillRL's live
retrieval environment, search backend, training procedure, or paper results.

Install the core and this package, then explicitly opt in to both plugin groups:

```python
from rethinkskill.benchmarks.capabilities import capability_catalog

catalog = capability_catalog(
    load_benchmark_plugins=True,
    load_harness_plugins=True,
)
```

Plugin loading is explicit. Core-only installation still declares all seven
capabilities, but truthfully reports that their scorer and harness are absent.
