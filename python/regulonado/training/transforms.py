from __future__ import annotations

import torch


class _IdentityTransform:
    """Identity transform: no normalization."""

    name = "identity"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply identity transformation (no-op).

        Parameters
        ----------
        x : torch.Tensor
            Input signal values.

        Returns
        -------
        torch.Tensor
            Input unchanged.
        """
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply identity inverse transformation (no-op).

        Parameters
        ----------
        y : torch.Tensor
            Transformed signal values.

        Returns
        -------
        torch.Tensor
            Input unchanged.
        """
        return y


class _PowerTransform:
    """Power law transform: x → x^p.

    Applies clipping at 0 to ensure numerical stability.

    Parameters
    ----------
    power : float
        Exponent p. Common values: 0.5 (square root), 2.0 (squaring).
    """

    def __init__(self, power: float) -> None:
        self.name = "power"
        self._power = power
        self._inv_power = 1.0 / power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply power transform.

        Parameters
        ----------
        x : torch.Tensor
            Input signal values.

        Returns
        -------
        torch.Tensor
            Transformed values: clamp(x, min=0) ^ power.
        """
        return torch.clamp_min(x.float(), 0.0) ** self._power

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse power transform.

        Parameters
        ----------
        y : torch.Tensor
            Transformed signal values.

        Returns
        -------
        torch.Tensor
            Original values: clamp(y, min=0) ^ (1/power).
        """
        return torch.clamp_min(y.float(), 0.0) ** self._inv_power


class _Log1pTransform:
    """Log1p transform: x → log(1 + x).

    Stabilizes large signal values while preserving small values. Clipping
    ensures numerical stability for negative inputs.
    """

    name = "log1p"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply log1p transform.

        Parameters
        ----------
        x : torch.Tensor
            Input signal values.

        Returns
        -------
        torch.Tensor
            Transformed values: log(1 + clamp(x, min=0)).
        """
        return torch.log1p(torch.clamp_min(x.float(), 0.0))

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse log1p transform.

        Parameters
        ----------
        y : torch.Tensor
            Transformed signal values.

        Returns
        -------
        torch.Tensor
            Original values: clamp(exp(y) - 1, min=0).
        """
        return torch.clamp_min(torch.expm1(y.float()), 0.0)


_REGISTRY: dict[str, object] = {
    "identity": _IdentityTransform(),
    "log1p": _Log1pTransform(),
}


def get_transform(
    name: str, power_value: float | None = None
) -> _IdentityTransform | _PowerTransform | _Log1pTransform:
    """Get a target value transform by name.

    Returns a stateless transform object for normalizing genomic signal values.
    Supported transforms: identity (no-op), log1p (log(1 + x)), and power
    (x^p). Power can be specified as a parameter or in the name string
    (legacy support for "power_0_5" → 0.5).

    Parameters
    ----------
    name : str
        Transform name: "identity", "log1p", "power", or "power_<value>"
        (e.g., "power_0_5" for square root).
    power_value : float | None, optional
        Exponent for power transform. Required if name=="power". Ignored if
        name starts with "power_".

    Returns
    -------
    _IdentityTransform | _PowerTransform | _Log1pTransform
        Transform object with forward(x) and inverse(y) methods.

    Raises
    ------
    ValueError
        If name is unrecognized or if name=="power" without power_value.

    Examples
    --------
    >>> t = get_transform("log1p")
    >>> x = torch.tensor([1.0, 10.0])
    >>> y = t.forward(x)
    >>> x_recovered = t.inverse(y)
    """
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
