"""Array-in prediction for region-count models: the region-model counterpart to
:mod:`regulonado.design.predictor`.

``RegionSequencePredictor``/``RegionFoldEnsemble`` mirror ``SequencePredictor``/``FoldEnsemble``
attribute-for-attribute so the rest of the design/attribution machinery (``FoldEnsemble.crop_bp``
consumers, ``TrackReadout``-shaped readouts, ``ism_scan``/``grad_scan``) works against either
kind of ensemble unchanged. The one output axis a region model has that a track model doesn't
need — the per-group scalar rate -- is reshaped to a degenerate one-bin axis so it fits the same
``(F, B, T, N)`` contract everywhere else in the codebase.

``RegionContrastReadout`` is the region analogue of ``design.attribution.TrackReadout``: instead
of reading one (group of) track columns over a bin window, it reads one group column, optionally
centred against the mean of every group -- the same contrast the training-time
``contrast_pearson_<group>`` selection metric is built from (see
``training/regions/metrics.py:227 _contrast``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from regulonado.design.predictor import FoldEnsemble
from regulonado.design.search import _to_numpy

logger = logging.getLogger(__name__)

__all__ = ["RegionContrastReadout", "RegionFoldEnsemble", "RegionSequencePredictor"]


class RegionSequencePredictor:
    """Array-in predictor for one region-model fold: ``(B, 4, context_length) -> (B, n_groups, 1)``.

    The scalar per group is ``softplus(eta)``, computed directly from the model's ``eta`` (the
    per-group log rate returned by ``RegionCountModel.group_log_rates``) rather than via
    ``exp`` then ``log1p`` -- that round trip loses precision and overflows float32 once
    ``eta > 88``, and ``softplus`` is exactly the same quantity
    ``training/regions/metrics.py``'s ``_contrast`` centres across groups to build the
    ``contrast_pearson_<group>`` selection metric this predictor exists to optimise.

    Attribute names deliberately match ``design.predictor.SequencePredictor`` so
    ``design.predictor.FoldEnsemble``'s machinery (and anything built against it, like
    ``design.attribution.TrackReadout``) works unchanged against a region ensemble:
    ``context_length``, ``n_pred_bins`` (always 1 -- one degenerate bin per group), ``bin_size``,
    ``track_names``. ``track_names`` returning ``group_names`` is a deliberate pun: the "track"
    axis of a region model IS its group axis, so downstream ``track_{i}`` columns end up labelled
    with group names for free, with no changes anywhere else.

    Region-specific attributes: ``group_names`` (same list as ``track_names``, under its own
    name for callers that want to be explicit), ``window`` (the ``TrunkWindow`` the live trunk
    was built with), ``crop_bp`` (bp trimmed from each side of the context before the region's
    pooled bins start), and ``snap_bp`` (the *effective*, i.e. pooled, bin width -- not
    derivable from ``context_length``/``n_pred_bins``/``bin_size`` the way it is for a per-bin
    track model, since ``n_pred_bins`` is always 1 here).
    """

    def __init__(
        self,
        checkpoint_dir,
        dataset_dir=None,
        device: str | None = None,
        batch_size: int = 1,
    ) -> None:
        from regulonado.training.regions.inference import load_region_count_model

        if dataset_dir is not None:
            logger.info(
                f"RegionSequencePredictor ignores dataset_dir={dataset_dir!r} (accepted only "
                "for FoldSpec signature compatibility with SequencePredictor)"
            )

        self.model = load_region_count_model(checkpoint_dir, device=device)
        self.model.eval()

        if not hasattr(self.model, "trunk"):
            raise ValueError(
                f"{checkpoint_dir} has no live trunk; RegionSequencePredictor needs a live-trunk "
                "region-count checkpoint (load_region_count_model raises for cached-embeddings "
                "heads, so this should not happen in practice)"
            )
        window = self.model.trunk.window
        self.window = window
        self.context_length = int(window.input_length)
        self.n_pred_bins = 1
        self.bin_size = int(window.k * window.pool_factor * window.bin_size)
        self.crop_bp = window.output_offset_bp + window.first_bin * window.bin_size
        self.snap_bp = window.bin_size * window.pool_factor

        config = self.model.config
        n_groups = int(config.n_groups)
        self.group_names = list(
            config.group_names or [f"group{i}" for i in range(n_groups)]
        )
        # Deliberate pun -- see the class docstring.
        self.track_names = self.group_names

        first_param = next(self.model.parameters())
        self.device = str(first_param.device)
        self.dtype = first_param.dtype
        self.batch_size = batch_size

    def __call__(self, one_hot_batch):
        import numpy as np
        import torch

        if isinstance(one_hot_batch, np.ndarray):
            one_hot_batch = torch.from_numpy(one_hot_batch)
        assert one_hot_batch.shape[-1] == self.context_length, (
            f"one_hot_batch has length {one_hot_batch.shape[-1]} on its last axis but this "
            f"model's input_length is {self.context_length}"
        )

        outputs = []
        start = 0
        with torch.inference_mode():
            while start < one_hot_batch.shape[0]:
                width = min(self.batch_size, one_hot_batch.shape[0] - start)
                chunk = one_hot_batch[start : start + width].to(
                    device=self.device, dtype=torch.float32
                )
                try:
                    eta = self.model.group_log_rates(sequence=chunk)  # (w, n_groups)
                    scalar = torch.nn.functional.softplus(eta)
                    outputs.append(scalar.unsqueeze(-1))  # (w, n_groups, 1)
                    start += width
                except RuntimeError as exc:
                    message = str(exc).lower()
                    recoverable = (
                        "integer out of range" in message
                        or "out of memory" in message
                        or "max_pool1d" in message
                    )
                    if not recoverable or width <= 1:
                        raise RuntimeError(
                            "Region oracle inference failed at batch size 1; check "
                            "checkpoint/model geometry and input context length."
                        ) from exc
                    self.batch_size = max(1, width // 2)
                    outputs.clear()
                    start = 0
                    if self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
        return torch.cat(outputs, dim=0)

    def to(self, device: str) -> "RegionSequencePredictor":
        """Move the already-loaded model to ``device`` in place (no disk I/O)."""
        self.model.to(device)
        self.device = device
        self.dtype = next(self.model.parameters()).dtype
        return self

    def gradient(
        self,
        one_hot_context,
        track_indices: Sequence[int],
        bins,
        reduction: Literal["mean", "max", "topk"] = "mean",
        topk_bins: int = 10,
    ):
        """d(centred contrast)/d(input) for one context, in one forward+backward pass.

        Unlike ``__call__``, this runs with autograd enabled (no ``inference_mode``). ``bins``
        must be ``slice(0, 1)`` -- the region model has no bin axis to select a window from, so
        anything else is a caller bug rather than something to silently coerce. ``reduction`` and
        ``topk_bins`` are likewise meaningless here (there is nothing to reduce over) and are
        ignored rather than accepted-then-silently-dropped, matching that this signature exists
        only so ``design.attribution.grad_scan`` can call a region ensemble exactly as it calls a
        track ensemble.

        The scored quantity is ``softplus(eta)`` centred against the mean over every group, then
        averaged over ``track_indices`` (the selected group(s)) -- the same contrast
        ``RegionContrastReadout`` computes, so the gradient approximates the same thing ISM would
        measure against that readout.

        Returns ``(grad (4, L) float32 numpy, score float)``.
        """
        import numpy as np
        import torch

        if bins != slice(0, 1):
            raise ValueError(
                f"RegionSequencePredictor.gradient only supports bins=slice(0, 1) (one "
                f"degenerate bin per group); got {bins!r}"
            )
        if isinstance(one_hot_context, np.ndarray):
            one_hot_context = torch.from_numpy(one_hot_context)
        x = one_hot_context.to(device=self.device, dtype=torch.float32).unsqueeze(0)
        x.requires_grad_(True)
        with torch.enable_grad():
            eta = self.model.group_log_rates(sequence=x)  # (1, n_groups)
            scalar = torch.nn.functional.softplus(eta)
            centred = scalar - scalar.mean(dim=-1, keepdim=True)
            score = centred[:, list(track_indices)].mean(dim=-1)
            score.backward()
        grad = x.grad[0].detach().to(torch.float32).cpu().numpy()
        return grad, float(score.detach().to(torch.float32).cpu())


class RegionFoldEnsemble(FoldEnsemble):
    """Runs the same one-hot batch through several independently trained region-model folds.

    A thin subclass: ``__init__``/``predict``/``gradient`` are inherited byte-for-byte from
    ``design.predictor.FoldEnsemble`` because ``RegionSequencePredictor``'s attribute names
    match ``SequencePredictor``'s exactly (see its docstring). Only the per-fold loader and the
    consistency check are region-specific -- ``_load`` must build a ``RegionSequencePredictor``,
    and ``_assert_consistent`` must additionally require the same ``TrunkWindow`` across folds:
    mismatched windows would silently score each fold's designs against a different crop of the
    context, with no shape error to catch it. ``crop_bp``/``snap_bp``/``group_names`` are also
    overridden/added here because ``snap_bp`` (the pooled bin width) isn't derivable from
    ``context_length``/``n_pred_bins``/``bin_size`` the way the base class's generic ``crop_bp``
    formula assumes -- both must be carried explicitly from a fold's own predictor instead.
    """

    def _load(self, spec, *, device: str | None) -> RegionSequencePredictor:
        return RegionSequencePredictor(
            spec.checkpoint_dir, spec.dataset_dir, device, self.batch_size
        )

    def _assert_consistent(self, predictors: list[RegionSequencePredictor]) -> None:
        super()._assert_consistent(predictors)
        first_spec, first = self._specs[0], predictors[0]
        first_label = first_spec.name or str(first_spec.checkpoint_dir)
        first_window = first.window.as_dict()
        for spec, predictor in zip(self._specs[1:], predictors[1:]):
            label = spec.name or str(spec.checkpoint_dir)
            window = predictor.window.as_dict()
            if window != first_window:
                differing = sorted(
                    key for key in first_window if first_window[key] != window.get(key)
                )
                raise ValueError(
                    f"Fold {label!r} has a different window than {first_label!r} "
                    f"(differs in {differing}): {window} vs {first_window}"
                )

    @property
    def group_names(self) -> list[str]:
        return self._predictors[0].group_names

    @property
    def crop_bp(self) -> int:
        """Overrides the base class's generic ``(context_length - n_pred_bins*bin_size)//2``,
        which is meaningless here since ``n_pred_bins`` is always 1 -- carried from the fold's
        own predictor instead."""
        return self._predictors[0].crop_bp

    @property
    def snap_bp(self) -> int:
        return self._predictors[0].snap_bp


@dataclass(slots=True)
class RegionContrastReadout:
    """Scalar readout of one group's score, ensembled across folds -- the region-model analogue
    of ``design.attribution.TrackReadout``.

    ``ensemble.predict(one_hot_batch)`` must return ``(F, B, n_groups, 1)`` (a
    ``RegionSequencePredictor``/``RegionFoldEnsemble``, or a duck-typed stand-in). With
    ``centre=True`` (the default), each fold's per-group score is centred against the mean over
    every group before ``group_index`` is selected -- this is exactly the contrast
    ``objective.contrast_energy`` optimises (see that function's docstring for the algebraic
    identity), so a design search using one and an ISM sweep reading the other agree about what
    "specific" means. ``centre=False`` is a diagnostic escape hatch for absolute accessibility
    (how much signal a group gets in isolation) rather than specificity relative to the rest.

    Deliberately takes no ``bins``/``bin_reduction`` -- a region model has a single degenerate
    bin per group, so those parameters would be meaningless, and silently accepting-then-ignoring
    them would be a trap for a caller expecting ``TrackReadout``'s windowing semantics.

    ``__call__`` returns ``(scores (B,), per_fold_scores (F, B))``, the same contract as
    ``TrackReadout``, so ``ism_scan``/``grad_scan`` compose with it unchanged.
    """

    ensemble: Any
    group_index: int
    fold_reduction: Literal["mean", "median"] = "mean"
    centre: bool = True

    def __call__(self, one_hot_batch):
        import numpy as np

        preds = _to_numpy(self.ensemble.predict(one_hot_batch))[..., 0]  # (F, B, G)
        s = preds[:, :, self.group_index]
        if self.centre:
            s = s - preds.mean(axis=-1)
        if self.fold_reduction == "mean":
            scores = s.mean(axis=0)
        elif self.fold_reduction == "median":
            scores = np.median(s, axis=0)
        else:
            raise ValueError(f"Unknown fold_reduction {self.fold_reduction!r}")
        return scores, s
