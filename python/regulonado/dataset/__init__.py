"""Dataset construction and signal transforms."""

from . import build as _build
from . import discovery as _discovery

# Preserve the historical facade, including private test/workflow helpers that
# callers imported from ``regulonado.dataset`` before the module split.
for _module in (_build, _discovery):
    globals().update({name: value for name, value in vars(_module).items() if name != "__name__"})
__all__ = sorted(
    {name for module in (_build, _discovery) for name in vars(module) if not name.startswith("__")}
)
