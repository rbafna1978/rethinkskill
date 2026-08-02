"""Entry-point registration for the optional ALFWorld harness."""

from rethinkskill.benchmarks.harness_catalog import (
    EvaluationAuthority,
    NativeHarnessSpec,
)
from rethinkskill_alfworld.harness import AlfWorldHarness


def spec() -> NativeHarnessSpec:
    return NativeHarnessSpec(
        name="alfworld",
        harness=AlfWorldHarness(),
        execution_scope=(
            "official installed ALFWorld text episode; bounded observation/"
            "action interaction; terminal environment won-state evaluation; "
            "external frozen corpus"
        ),
        source="alfworld==0.4.2 installed environment",
        evaluation_authority=EvaluationAuthority.HARNESS,
    )


__all__ = ["spec"]
