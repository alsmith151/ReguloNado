"""Interactive prompt primitive, matching `seqnado config`'s behaviour.

When SeqNado is installed we use its ``get_user_input`` directly so the two
generators are literally the same code. The fallback below reproduces the same
signature and rendering (``"{prompt} (default: {x}): "``, re-prompt on invalid
input) so the experience does not change based on whether an optional dependency
happens to be present.
"""

from __future__ import annotations

import os
from typing import Sequence

_TRUE = {"yes", "y", "true", "1"}
_FALSE = {"no", "n", "false", "0"}


def _fallback_get_user_input(
    prompt: str,
    default: str | None = None,
    is_boolean: bool = False,
    choices: Sequence[str] | None = None,
    is_path: bool = False,
    required: bool = True,
    multi_select: bool = False,
):
    """Local reimplementation of ``seqnado.config.user_input.get_user_input``."""
    while True:
        if choices:
            suffix = f"({', '.join(choices)})" if multi_select else f"({'/'.join(choices)})"
        elif default is not None:
            suffix = f"(default: {default})"
        else:
            suffix = ""

        answer = input(f"{prompt} {suffix}: ").strip()

        if not answer:
            if default is not None:
                answer = str(default)
            elif not required:
                return None
            else:
                print("Input cannot be empty. Please try again.")
                continue

        if is_boolean:
            lowered = answer.lower()
            if lowered in _TRUE:
                return True
            if lowered in _FALSE:
                return False
            print("Invalid boolean value. Please enter yes/no, y/n, true/false, or 1/0.")
            continue

        if multi_select and choices:
            selections = [item.strip() for item in answer.split(",") if item.strip()]
            invalid = [item for item in selections if item not in choices]
            if invalid:
                print(f"Invalid choices: {', '.join(invalid)}")
                print(f"Please choose from: {', '.join(choices)}")
                continue
            if not selections:
                print("No valid selections made. Please try again.")
                continue
            return selections

        if choices and answer not in choices:
            print(f"Invalid choice. Please choose from: {', '.join(choices)}")
            continue

        if is_path and answer and not os.path.exists(answer):
            print(f"The path '{answer}' does not exist. Please try again.")
            continue

        return answer


def get_user_input(*args, **kwargs):
    """Prompt the user, preferring SeqNado's implementation when available."""
    try:
        from seqnado.config.user_input import get_user_input as _seqnado_prompt
    except ImportError:
        return _fallback_get_user_input(*args, **kwargs)
    return _seqnado_prompt(*args, **kwargs)


def ask(
    prompt: str,
    default=None,
    *,
    is_boolean: bool = False,
    choices: Sequence[str] | None = None,
    is_path: bool = False,
    required: bool = True,
    multi_select: bool = False,
    interactive: bool = True,
):
    """Prompt, or return the default unchanged when running non-interactively."""
    if not interactive:
        return default
    return get_user_input(
        prompt,
        default=None if default is None else str(default),
        is_boolean=is_boolean,
        choices=list(choices) if choices else None,
        is_path=is_path,
        required=required,
        multi_select=multi_select,
    )


def ask_int(prompt: str, default: int, *, interactive: bool = True) -> int:
    """Prompt for an integer, re-prompting until one is given."""
    while True:
        answer = ask(prompt, default, interactive=interactive)
        try:
            return int(str(answer))
        except (TypeError, ValueError):
            if not interactive:
                raise
            print(f"'{answer}' is not a whole number. Please try again.")
