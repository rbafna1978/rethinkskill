"""RethinkSkill integrations definitions."""

from __future__ import annotations

from rethinkskill.integrations.catalog import (
    AdapterDelivery,
    ExecutionFidelity,
    IntegrationSpec,
    IntegrationTier,
)


def builtin_integration_specs() -> tuple[IntegrationSpec, ...]:
    """Return maintained integration boundaries in deterministic order."""
    return (
        IntegrationSpec(
            name="original-qa",
            tier=IntegrationTier.FIRST_PARTY,
            fidelity=ExecutionFidelity.PAPER_NATIVE,
            benchmarks=("searchqa", "officeqa", "docvqa", "livemath"),
            description="Default first-party adapters for the original rethinkskill question-answering benchmarks.",
        ),
        IntegrationSpec(
            name="spreadsheetbench",
            tier=IntegrationTier.FIRST_PARTY_OPTIONAL,
            fidelity=ExecutionFidelity.PAPER_NATIVE,
            benchmarks=("spreadsheetbench",),
            description="First-party workbook code-generation and artifact-verification integration.",
            delivery=AdapterDelivery.OPTIONAL_PACKAGE,
            adapter_distribution="rethinkskill-spreadsheetbench",
            optional_dependencies=("openpyxl>=3.1,<4",),
        ),
        IntegrationSpec(
            name="alfworld",
            tier=IntegrationTier.FIRST_PARTY_OPTIONAL,
            fidelity=ExecutionFidelity.PAPER_NATIVE,
            benchmarks=("alfworld",),
            description="Optional first-party interactive ALFWorld integration backed by an installed environment and separately materialized corpus.",
            delivery=AdapterDelivery.OPTIONAL_PACKAGE,
            adapter_distribution="rethinkskill-alfworld",
            optional_dependencies=("alfworld==0.4.2",),
            external_requirements=("official_corpus",),
        ),
        IntegrationSpec(
            name="skillrl-offline",
            tier=IntegrationTier.EXTERNAL,
            fidelity=ExecutionFidelity.OFFLINE_PROXY,
            benchmarks=("nq", "triviaqa", "popqa", "hotpotqa", "2wiki", "musique", "bamboogle"),
            description="External context-grounded offline scorer and harness package for the SkillRL QA suite; not a reproduction of live retrieval environments.",
            delivery=AdapterDelivery.OPTIONAL_PACKAGE,
            adapter_distribution="rethinkskill-skillrl",
        ),
        IntegrationSpec(
            name="mce",
            tier=IntegrationTier.EXTERNAL,
            fidelity=ExecutionFidelity.DATASET_BOUND,
            benchmarks=("finer", "uspto50k", "symptom2disease", "lawbench-charge", "aegis2"),
            description="External file-backed adapters for the Meta Context Engineering suite with dataset preprocessing kept provenance-bound.",
            delivery=AdapterDelivery.OPTIONAL_PACKAGE,
            adapter_distribution="rethinkskill-mce",
        ),
        IntegrationSpec(
            name="webshop",
            tier=IntegrationTier.EXTERNAL,
            fidelity=ExecutionFidelity.EXTERNAL_REQUIRED,
            benchmarks=("webshop",),
            description="Declared WebShop capability with an opt-in official environment adapter under experimental/integrations/webshop.",
            delivery=AdapterDelivery.OPTIONAL_PACKAGE,
            adapter_distribution="rethinkskill-webshop",
            external_requirements=(
                "official_checkout",
                "external_python_3_8",
                "official_data_and_search_index",
            ),
        ),
        IntegrationSpec(
            name="mcp-atlas",
            tier=IntegrationTier.EXTERNAL,
            fidelity=ExecutionFidelity.EXTERNAL_REQUIRED,
            benchmarks=("mcp-atlas",),
            description="Declared-only MCP-Atlas capability whose official external environment is not bundled.",
            delivery=AdapterDelivery.DECLARED_ONLY,
            adapter_distribution=None,
            external_requirements=(
                "official_external_environment",
                "typescript_agent_harness",
                "llm_as_judge",
                "mcp_server_credentials_and_seed_data",
            ),
        ),
    )
