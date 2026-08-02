"""RethinkSkill errors."""

from __future__ import annotations


class RethinkSkillError(RuntimeError):
    """Base class for actionable, user-facing failures."""


class ConfigurationError(RethinkSkillError):
    """The experiment registry or a user override is invalid."""


class ProtectedPathError(RethinkSkillError):
    """A requested write would enter an immutable evidence or manuscript tree."""


class MissingArtifactError(RethinkSkillError):
    """A required local artifact is not materialized."""


class AuthorizationRequiredError(RethinkSkillError):
    """A live model execution lacks explicit operator authorization."""


class ResultValidationError(RethinkSkillError):
    """A result ledger violates the expected row contract."""
