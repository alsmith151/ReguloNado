from __future__ import annotations

import torch


class _IdentityTransform:
    name = "identity"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y


class _PowerTransform:
    def __init__(self, power: float) -> None:
        self.name = "power"
        self._power = power
        self._inv_power = 1.0 / power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp_min(x.float(), 0.0) ** self._power

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.clamp_min(y.float(), 0.0) ** self._inv_power


class _Log1pTransform:
    name = "log1p"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp_min(x.float(), 0.0))

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.clamp_min(torch.expm1(y.float()), 0.0)


_REGISTRY: dict[str, object] = {
    "identity": _IdentityTransform(),
    "log1p": _Log1pTransform(),
}


def get_transform(
    name: str, power_value: float | None = None
) -> _IdentityTransform | _PowerTransform | _Log1pTransform:
    if name == "power":
        if power_value is None:
            raise ValueError("target transform 'power' requires power_value")
        return _PowerTransform(power_value)
    if name.startswith("power_"):
        legacy_power = float(name.removeprefix("power_").replace("_", "."))
        return _PowerTransform(power_value if power_value is not None else legacy_power)
    if name not in _REGISTRY:
        raise ValueError(f"Unknown target transform {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]  # type: ignore[return-value]
