"""Compatibility launcher for the training runner."""

from .training.runner import *  # noqa: F401,F403
from .training.runner import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
