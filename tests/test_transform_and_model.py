"""Tests for the per-example read-time transform (make_transform) and the model's
GPU-side one-hot encoding of uint8 sequence tokens.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from regulonado.dataset.build import _COMPLEMENT, make_transform
from regulonado.model import RegulonadoModel, one_hot_sequence_tokens
from regulonado.model.heads import TransferMLPPerturbHead


def _one_hot_np(tokens: np.ndarray) -> np.ndarray:
    """Reference one-hot encoder matching the old stored (4, L) int8 layout."""
    one_hot = np.zeros((4, tokens.shape[0]), dtype=np.int8)
    for base in range(4):
        one_hot[base] = tokens == base
    return one_hot


class TestTokenReverseComplement:
    def test_rc_token_lookup_equals_old_one_hot_flip(self) -> None:
        tokens = np.array([0, 1, 2, 3, 4, 4, 2, 0], dtype=np.uint8)

        # New: reverse-complement directly on uint8 tokens.
        rc_tokens = _COMPLEMENT[tokens[::-1]]

        # Old: one-hot the raw tokens, then flip both axes (reverses order and swaps
        # A<->T, C<->G row-wise, which is exactly base complementation for a one-hot
        # array ordered A,C,G,T).
        one_hot = _one_hot_np(tokens)
        old_rc_one_hot = np.flip(one_hot, axis=(0, 1)).copy()

        assert np.array_equal(_one_hot_np(rc_tokens), old_rc_one_hot)

    def test_n_and_pad_tokens_complement_to_themselves(self) -> None:
        assert _COMPLEMENT[4] == 4

    def test_base_pairs_complement_correctly(self) -> None:
        # A(0)<->T(3), C(1)<->G(2).
        assert _COMPLEMENT[0] == 3
        assert _COMPLEMENT[3] == 0
        assert _COMPLEMENT[1] == 2
        assert _COMPLEMENT[2] == 1


class TestMakeTransform:
    def test_shift_crop_center_crop_is_deterministic(self) -> None:
        context, n_pred_bins, bin_size, shift_max_bins = 4, 2, 1, 2
        stored_context = context + 2 * shift_max_bins * bin_size
        stored_bins = n_pred_bins + 2 * shift_max_bins

        transform = make_transform(
            scale_factors=np.ones(1),
            clip_soft=1000.0,
            clip_hard=1000.0,
            apply_scale=False,
            apply_squash=False,
            apply_clip=False,
            shift_max_bins=shift_max_bins,
            context_length=context,
            n_pred_bins=n_pred_bins,
            bin_size=bin_size,
            center_crop=True,
        )
        example = {
            "sequence_tokens": np.arange(stored_context, dtype=np.uint8) % 4,
            "signal": np.arange(stored_bins, dtype=np.float32).reshape(1, stored_bins),
            "interval": "chr1:0-10",
        }

        out = transform(example)

        assert out["input_ids"].shape == (context,)
        assert out["labels"].shape == (1, n_pred_bins)
        # center_crop always offsets by shift_max_bins.
        expected_ids = example["sequence_tokens"][
            shift_max_bins * bin_size : shift_max_bins * bin_size + context
        ]
        np.testing.assert_array_equal(out["input_ids"], expected_ids)
        expected_labels = example["signal"][:, shift_max_bins : shift_max_bins + n_pred_bins]
        np.testing.assert_array_equal(out["labels"], expected_labels)
        # Non-transform fields pass through unchanged.
        assert out["interval"] == "chr1:0-10"

    def test_output_dtypes(self) -> None:
        transform = make_transform(scale_factors=np.ones(2), clip_soft=10.0, clip_hard=20.0)
        example = {
            "sequence_tokens": np.array([0, 1, 2, 3], dtype=np.uint8),
            "signal": np.ones((2, 3), dtype=np.float32),
        }

        out = transform(example)

        assert out["input_ids"].dtype == np.uint8
        assert out["labels"].dtype == np.float32


class TestOneHotSequenceTokens:
    def test_bases_map_to_expected_one_hot_columns(self) -> None:
        tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.uint8)

        one_hot = one_hot_sequence_tokens(tokens)

        assert one_hot.shape == (1, 4, 4)
        expected = torch.eye(4)
        torch.testing.assert_close(one_hot[0], expected)

    def test_n_and_pad_tokens_produce_an_all_zero_column(self) -> None:
        tokens = torch.tensor([[0, 4, 5]], dtype=torch.uint8)

        one_hot = one_hot_sequence_tokens(tokens)

        torch.testing.assert_close(one_hot[0, :, 1], torch.zeros(4))
        torch.testing.assert_close(one_hot[0, :, 2], torch.zeros(4))
        # A real base column still has exactly one set bit.
        assert one_hot[0, :, 0].sum() == 1


class _DummyAdapter(nn.Module):
    """Minimal backbone stand-in: passes one-hot/float input straight through a conv."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(4, 4, kernel_size=1, bias=False)
        # Fixed, deterministic weights so different forward calls are directly comparable.
        with torch.no_grad():
            self.conv.weight.copy_(torch.eye(4).unsqueeze(-1))
        self.feature_dim = 4

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.conv(input_ids)

    def iter_named_blocks(self):
        yield "conv", self.conv


class TestRegulonadoModelUint8Input:
    def test_uint8_tokens_and_float_one_hot_give_the_same_output(self) -> None:
        model = RegulonadoModel(
            backbone=_DummyAdapter(),
            head=TransferMLPPerturbHead(in_ch=4, hidden=4, n_tracks=2),
        )
        model.eval()
        tokens = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.uint8)
        float_one_hot = one_hot_sequence_tokens(tokens)

        with torch.no_grad():
            out_uint8 = model(tokens)
            out_float = model(float_one_hot)

        torch.testing.assert_close(out_uint8, out_float)
