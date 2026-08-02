"""Entry-point registration for the optional WebShop harness."""

from rethinkskill.benchmarks.harness_catalog import (
    EvaluationAuthority,
    NativeHarnessSpec,
)
from rethinkskill_webshop.harness import WebShopHarness


def spec() -> NativeHarnessSpec:
    return NativeHarnessSpec(
        name="webshop",
        harness=WebShopHarness(),
        execution_scope=(
            "official WebShop text Gym environment in an independently "
            "configured subprocess; bounded search/click interaction; "
            "official terminal reward"
        ),
        source="princeton-nlp/WebShop official environment",
        evaluation_authority=EvaluationAuthority.HARNESS,
    )


__all__ = ["spec"]
