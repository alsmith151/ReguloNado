"""Compatibility launcher for ``python -m regulonado``."""

from regulonado.cli.app import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
