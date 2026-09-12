"""Golden ``--help`` snapshots for the root app and every command/subcommand.

These exist to gate the F03 split of ``cli/app.py`` into per-command-group
modules (see docs/audit-2026-09-11.xml, finding F03): the split must be a pure
move with no change to options, defaults, or help text, and this is the check
that proves it. The snapshots are generated from the *current* CLI (before or
after the split — whichever is on disk) and must stay byte-identical across
the refactor.

Regenerate after an intentional CLI change with::

    REGULONADO_REGEN_GOLDEN=1 uv run --no-sync pytest tests/test_cli_help.py
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import pytest
from regulonado.cli.app import app
from typer.main import get_command
from typer.testing import CliRunner

runner = CliRunner()
DATA_DIR = Path(__file__).parent / "data" / "cli_help"

# Fixed terminal width/colour so the rich-rendered help text (box borders,
# wrapping) is identical regardless of the environment the tests run in.
HELP_ENV = {"COLUMNS": "100", "LINES": "40", "NO_COLOR": "1", "TERM": "dumb"}


def _collect_cases() -> list[tuple[str, list[str]]]:
    """Walk the click command tree built from ``app`` and list every path.

    Returns ``(slug, args)`` pairs: ``args`` is the command path to pass to
    ``CliRunner.invoke`` (before appending ``--help``), and ``slug`` is that
    path joined with ``_`` for use as a golden-file name. The root app itself
    is included as ``("root", [])``.
    """
    root_command = get_command(app)
    cases: list[tuple[str, list[str]]] = [("root", [])]

    def walk(command: click.Command, path: list[str]) -> None:
        if isinstance(command, click.Group):
            for name, sub_command in command.commands.items():
                sub_path = [*path, name]
                cases.append(("_".join(sub_path), sub_path))
                walk(sub_command, sub_path)

    walk(root_command, [])
    return cases


CASES = _collect_cases()


@pytest.mark.parametrize("slug,args", CASES, ids=[slug for slug, _ in CASES])
def test_help_snapshot(slug: str, args: list[str]) -> None:
    result = runner.invoke(app, [*args, "--help"], env=HELP_ENV)
    assert result.exit_code == 0, result.output

    golden_path = DATA_DIR / f"{slug}.txt"
    if os.environ.get("REGULONADO_REGEN_GOLDEN") == "1":
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(result.output)

    assert golden_path.exists(), (
        f"Missing golden file {golden_path}; regenerate with "
        f"REGULONADO_REGEN_GOLDEN=1 uv run --no-sync pytest {__file__}"
    )
    assert result.output == golden_path.read_text()
