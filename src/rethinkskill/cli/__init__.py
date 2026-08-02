"""Command-line interface for RethinkSkill."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence

from rethinkskill.cli.commands import command_router
from rethinkskill.cli.parser import build_parser, parser_command_names
from rethinkskill.errors import RethinkSkillError
from rethinkskill.utils.serde import thaw_json_value


def _json(value: object, *, stream: object | None = None) -> None:
    print(
        json.dumps(
            thaw_json_value(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        file=stream or sys.stdout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = command_router().dispatch(args)
        _json(result.payload)
        return result.exit_code
    except RethinkSkillError as exc:
        _json(
            {
                "schema_version": 1,
                "status": "RETHINKSKILL_ERROR",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "model_calls": 0,
            },
            stream=sys.stderr,
        )
        return 2


__all__ = ["build_parser", "main", "parser_command_names"]
