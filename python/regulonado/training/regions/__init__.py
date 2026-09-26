"""Region-level count head training on cached backbone embeddings.

See :mod:`regulonado.training.regions.model` for the model and config,
:mod:`regulonado.training.regions.loss` for the count likelihood, and
:mod:`regulonado.training.regions.metrics` for evaluation metrics.
"""

from regulonado.training.regions.loss import COUNT_NOISE_MODELS, CountLikelihoodLoss
from regulonado.training.regions.metrics import (
    GroupedCountMetrics,
    PerTaskCorrelationMetrics,
    group_count_rates,
)
from regulonado.training.regions.model import (
    AttentionPool,
    CountHead,
    RegionCountConfig,
    RegionCountModel,
    RegionCountOutput,
)

__all__ = [
    "COUNT_NOISE_MODELS",
    "AttentionPool",
    "CountHead",
    "CountLikelihoodLoss",
    "GroupedCountMetrics",
    "PerTaskCorrelationMetrics",
    "RegionCountConfig",
    "RegionCountModel",
    "RegionCountOutput",
    "group_count_rates",
]
