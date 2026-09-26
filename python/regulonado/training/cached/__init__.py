"""Region-level count head training on cached backbone embeddings.

See :mod:`regulonado.training.cached.model` for the model and config,
:mod:`regulonado.training.cached.loss` for the count likelihood, and
:mod:`regulonado.training.cached.metrics` for evaluation metrics.
"""

from regulonado.training.cached.loss import COUNT_NOISE_MODELS, CountLikelihoodLoss
from regulonado.training.cached.metrics import (
    GroupedCountMetrics,
    PerTaskCorrelationMetrics,
    group_count_rates,
)
from regulonado.training.cached.model import (
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
