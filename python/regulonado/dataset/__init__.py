"""Dataset construction and signal transforms."""

from . import build as _build

# Preserve the historical facade, including private test/workflow helpers that
# callers imported from ``regulonado.dataset`` before the module split.
globals().update({name: value for name, value in vars(_build).items() if name != "__name__"})
__all__ = [name for name in vars(_build) if not name.startswith("__")]
