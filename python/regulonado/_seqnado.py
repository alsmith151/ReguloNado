"""Optional SeqNado integration.

SeqNado is an optional dependency (``pip install regulonado[seqnado]``). Every
code path that needs its project API goes through :func:`require_seqnado_project`
so that a missing — or too old — install produces one actionable message instead
of an ImportError in one place and an AttributeError in another.

The project API (``seqnado.open_project`` and friends) landed after the 1.0.7
release, so an already-installed SeqNado may import cleanly and still lack it;
the capability check below covers that case explicitly.
"""

from __future__ import annotations

from typing import Any

_INSTALL_HINT = (
    "SeqNado project support requires SeqNado >= 1.1.\n"
    "  pip install 'regulonado[seqnado]'\n"
    "Until 1.1 is published, install from the development branch:\n"
    "  pip install 'seqnado @ git+https://github.com/Milne-Group/SeqNado@develop'"
)


class SeqNadoUnavailableError(RuntimeError):
    """Raised when SeqNado's project API is needed but not usable."""


def seqnado_available() -> bool:
    """Return True when SeqNado is installed and exposes the project API."""
    try:
        require_seqnado()
    except SeqNadoUnavailableError:
        return False
    return True


def require_seqnado() -> Any:
    """Import SeqNado and verify it exposes the project API.

    Returns the ``seqnado`` module. Raises :class:`SeqNadoUnavailableError` with
    installation instructions otherwise.
    """
    try:
        import seqnado
    except ImportError as exc:
        raise SeqNadoUnavailableError(f"SeqNado is not installed.\n{_INSTALL_HINT}") from exc

    if not hasattr(seqnado, "open_project"):
        try:
            from importlib.metadata import version as _version

            installed = _version("seqnado")
        except (ImportError, ModuleNotFoundError):
            installed = getattr(seqnado, "__version__", "unknown")
        # Report where the module came from: the usual cause of a SeqNado that
        # imports but has no API is a directory named 'seqnado' shadowing the
        # installed package.
        location = getattr(seqnado, "__file__", None) or getattr(seqnado, "__path__", "?")
        raise SeqNadoUnavailableError(
            f"The SeqNado at {location} (version {installed}) has no project API "
            f"('seqnado.open_project').\n{_INSTALL_HINT}"
        )
    return seqnado


def open_project(output_dir, **kwargs) -> Any:
    """Open a SeqNado output directory, single or multiomics.

    Thin wrapper over ``seqnado.open_project`` that routes the missing-dependency
    case through :class:`SeqNadoUnavailableError`.
    """
    from pathlib import Path

    seqnado = require_seqnado()
    path = Path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"SeqNado output directory not found: {path}")
    return seqnado.open_project(path, **kwargs)


def is_multi_project(project: Any) -> bool:
    """True when ``project`` is a multiomics wrapper rather than a single project."""
    return hasattr(project, "assays") and hasattr(project, "items")
