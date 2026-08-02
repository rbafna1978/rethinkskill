"""File-backed context-grounded harness for the SkillRL offline proxy."""

from rethinkskill.benchmarks.qa import SearchQAHarness


class SkillRLOfflineHarness(SearchQAHarness):
    """Explicit package-owned name for the shared QA task schema."""


__all__ = ["SkillRLOfflineHarness"]
