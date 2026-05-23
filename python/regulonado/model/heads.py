from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
import torch.nn as nn

HeadType = Literal["bias", "film", "log_film", "residual_film", "transfer_mlp"]
ActivationType = Literal["softplus", "softplus_beta2", "exp", "identity"]

_CONDITION_COLLAPSE_IGNORED_FIELDS = {
    "condition",
    "condition_id",
    "track_index",
    "track_name",
    "name",
    "display_name",
    "scale_factor",
    "scale_factors",
    "clip_hard",
    "clip_soft",
}

_CONDITION_COLLAPSE_PREFERRED_FIELDS = (
    "assay_type_id",
    "assay_type",
    "cell_line_id",
    "cell_line",
    "timepoint_minutes",
    "timepoint",
    "target_id",
    "target",
    "strand",
    "replicate",
    "experiment_series",
)


class _ClampedExp(nn.Module):
    def __init__(self, max_logit: float = 20.0):
        super().__init__()
        self.max_logit = max_logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(x, max=self.max_logit))


def _normalise_track_metadata_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_track_metadata_value(item) for item in value)
    return value


def _track_index_sort_key(record: Mapping[str, object], default: int) -> int:
    value = record.get("track_index")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return default


def build_condition_shared_track_index(
    track_records: Sequence[Mapping[str, object]],
) -> list[int]:
    if not track_records:
        return []

    ordered_records = sorted(
        track_records,
        key=lambda record: _track_index_sort_key(record, len(track_records)),
    )
    available_fields = [
        field
        for field in _CONDITION_COLLAPSE_PREFERRED_FIELDS
        if any(record.get(field) is not None for record in ordered_records)
    ]

    if not available_fields:
        available_fields = sorted(
            {
                key
                for record in ordered_records
                for key, value in record.items()
                if key not in _CONDITION_COLLAPSE_IGNORED_FIELDS and value is not None
            }
        )

    if not available_fields:
        raise ValueError(
            "Cannot derive condition-shared track groups from track_records; "
            "no non-condition metadata fields were found."
        )

    shared_group_to_index: dict[tuple[object, ...], int] = {}
    condition_shared_track_index: list[int] = []
    for record in ordered_records:
        shared_key = tuple(
            _normalise_track_metadata_value(record.get(field)) for field in available_fields
        )
        base_track_index = shared_group_to_index.setdefault(shared_key, len(shared_group_to_index))
        condition_shared_track_index.append(base_track_index)
    return condition_shared_track_index


def _masked_embedding(
    ids: torch.Tensor | None,
    embedding: nn.Embedding | None,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if ids is None or embedding is None:
        return None
    encoded_ids = ids.to(device=device, dtype=torch.long)
    valid = encoded_ids >= 0
    if not valid.any():
        return None
    safe_ids = encoded_ids.clamp(min=0, max=embedding.num_embeddings - 1)
    embedded = embedding(safe_ids)
    return embedded * valid.unsqueeze(-1).to(dtype=embedded.dtype)


class TrackMetadataEncoder(nn.Module):
    def __init__(
        self,
        *,
        use_track_metadata: bool,
        metadata_hidden: int,
        dropout: float,
        num_conditions: int = 0,
        num_cell_lines: int = 0,
        num_assay_types: int = 0,
        num_targets: int = 0,
    ):
        super().__init__()
        self.use_track_metadata = use_track_metadata

        def _maybe_embed(n: int) -> nn.Embedding | None:
            return nn.Embedding(n, metadata_hidden) if (use_track_metadata and n > 0) else None

        self.condition_embedding = _maybe_embed(num_conditions)
        self.cell_line_embedding = _maybe_embed(num_cell_lines)
        self.assay_type_embedding = _maybe_embed(num_assay_types)
        self.target_embedding = _maybe_embed(num_targets)
        self.timepoint_mlp: nn.Sequential | None = (
            nn.Sequential(
                nn.Linear(1, metadata_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(metadata_hidden, metadata_hidden),
            )
            if use_track_metadata
            else None
        )
        self.metadata_mixer: nn.Sequential | None = (
            nn.Sequential(
                nn.LayerNorm(metadata_hidden),
                nn.Linear(metadata_hidden, metadata_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(metadata_hidden, metadata_hidden),
            )
            if use_track_metadata
            else None
        )
        if self.metadata_mixer is not None:
            final_linear = self.metadata_mixer[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.zeros_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        *,
        track_condition_ids: torch.Tensor | None = None,
        track_timepoint_minutes: torch.Tensor | None = None,
        track_cell_line_ids: torch.Tensor | None = None,
        track_assay_type_ids: torch.Tensor | None = None,
        track_target_ids: torch.Tensor | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.use_track_metadata:
            return None

        metadata_state: torch.Tensor | None = None
        embeddings = (
            (track_condition_ids, self.condition_embedding),
            (track_cell_line_ids, self.cell_line_embedding),
            (track_assay_type_ids, self.assay_type_embedding),
            (track_target_ids, self.target_embedding),
        )
        for ids, embedding in embeddings:
            piece = _masked_embedding(ids, embedding, device=device)
            if piece is not None:
                metadata_state = piece if metadata_state is None else metadata_state + piece

        if track_timepoint_minutes is not None and self.timepoint_mlp is not None:
            time_values = torch.nan_to_num(
                track_timepoint_minutes.to(device=device, dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            time_state = self.timepoint_mlp(torch.log1p(time_values).unsqueeze(-1))
            metadata_state = time_state if metadata_state is None else metadata_state + time_state

        if metadata_state is None:
            return None
        encoded_metadata = self.dropout(metadata_state)
        if self.metadata_mixer is not None:
            encoded_metadata = encoded_metadata + self.metadata_mixer(encoded_metadata)
        return encoded_metadata.to(dtype=dtype)


class _PerturbHeadBase(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int,
        hidden: int,
        n_tracks: int,
        metadata_hidden: int,
        dropout: float,
        use_track_metadata: bool,
        num_conditions: int,
        num_cell_lines: int,
        num_assay_types: int,
        num_targets: int,
        condition_shared_track_index: Sequence[int] | None = None,
        activation_type: ActivationType = "softplus",
    ):
        super().__init__()
        self.metadata_hidden = metadata_hidden
        self.n_tracks = n_tracks
        if condition_shared_track_index is not None:
            if len(condition_shared_track_index) != n_tracks:
                raise ValueError(
                    "condition_shared_track_index length must match n_tracks: "
                    f"expected {n_tracks}, got {len(condition_shared_track_index)}"
                )
            shared_index = torch.as_tensor(condition_shared_track_index, dtype=torch.long)
            self.register_buffer("condition_shared_track_index", shared_index)
            n_base_tracks = int(shared_index.max().item()) + 1 if shared_index.numel() else n_tracks
        else:
            self.register_buffer("condition_shared_track_index", None)
            n_base_tracks = n_tracks

        self.n_base_tracks = n_base_tracks
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1),
            nn.GELU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(hidden, n_base_tracks, 1),
        )
        if activation_type == "softplus_beta2":
            self.activation: nn.Module = nn.Softplus(beta=2)
        elif activation_type == "exp":
            self.activation = _ClampedExp()
        elif activation_type == "identity":
            self.activation = nn.Identity()
        else:
            self.activation = nn.Softplus()
        self.metadata_encoder = TrackMetadataEncoder(
            use_track_metadata=use_track_metadata,
            metadata_hidden=metadata_hidden,
            dropout=dropout,
            num_conditions=num_conditions,
            num_cell_lines=num_cell_lines,
            num_assay_types=num_assay_types,
            num_targets=num_targets,
        )

    def _expand_condition_shared_tracks(self, tensor: torch.Tensor) -> torch.Tensor:
        shared_index = self.condition_shared_track_index
        if shared_index is None:
            return tensor
        return tensor.index_select(dim=1, index=shared_index)

    def _shared_output_weights(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        output_layer = self.proj[3]
        weight = output_layer.weight.squeeze(-1)
        bias = output_layer.bias
        shared_index = self.condition_shared_track_index
        if shared_index is not None:
            weight = weight.index_select(0, shared_index)
            bias = bias.index_select(0, shared_index) if bias is not None else None
        return weight, bias

    def _encode_metadata(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        **metadata_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        return self.metadata_encoder(device=device, dtype=dtype, **metadata_ids)


class PerturbHead(_PerturbHeadBase):
    def __init__(
        self,
        *,
        in_ch: int = 1920,
        hidden: int = 512,
        n_tracks: int = 12,
        use_track_metadata: bool = False,
        num_conditions: int = 0,
        num_cell_lines: int = 0,
        num_assay_types: int = 0,
        num_targets: int = 0,
        metadata_hidden: int = 32,
        dropout: float = 0.0,
        condition_shared_track_index: Sequence[int] | None = None,
        activation_type: ActivationType = "softplus",
    ):
        super().__init__(
            in_ch=in_ch,
            hidden=hidden,
            n_tracks=n_tracks,
            metadata_hidden=metadata_hidden,
            dropout=dropout,
            use_track_metadata=use_track_metadata,
            num_conditions=num_conditions,
            num_cell_lines=num_cell_lines,
            num_assay_types=num_assay_types,
            num_targets=num_targets,
            condition_shared_track_index=condition_shared_track_index,
            activation_type=activation_type,
        )
        self.metadata_to_bias = nn.Linear(metadata_hidden, 1) if use_track_metadata else None
        if self.metadata_to_bias is not None:
            nn.init.zeros_(self.metadata_to_bias.weight)
            nn.init.zeros_(self.metadata_to_bias.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        track_condition_ids: torch.Tensor | None = None,
        track_timepoint_minutes: torch.Tensor | None = None,
        track_cell_line_ids: torch.Tensor | None = None,
        track_assay_type_ids: torch.Tensor | None = None,
        track_target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self._expand_condition_shared_tracks(self.proj(x))
        if self.metadata_to_bias is not None:
            metadata = self._encode_metadata(
                device=logits.device,
                dtype=logits.dtype,
                track_condition_ids=track_condition_ids,
                track_timepoint_minutes=track_timepoint_minutes,
                track_cell_line_ids=track_cell_line_ids,
                track_assay_type_ids=track_assay_type_ids,
                track_target_ids=track_target_ids,
            )
            if metadata is not None:
                bias = self.metadata_to_bias(metadata).squeeze(-1)
                if bias.ndim == 1:
                    bias = bias.unsqueeze(0)
                logits = logits + bias.unsqueeze(-1)
        return self.activation(logits)


class FiLMPerturbHead(_PerturbHeadBase):
    def __init__(
        self,
        *,
        in_ch: int = 1920,
        hidden: int = 512,
        n_tracks: int = 12,
        use_track_metadata: bool = False,
        num_conditions: int = 0,
        num_cell_lines: int = 0,
        num_assay_types: int = 0,
        num_targets: int = 0,
        metadata_hidden: int = 32,
        dropout: float = 0.0,
        condition_shared_track_index: Sequence[int] | None = None,
        activation_type: ActivationType = "softplus",
    ):
        super().__init__(
            in_ch=in_ch,
            hidden=hidden,
            n_tracks=n_tracks,
            metadata_hidden=metadata_hidden,
            dropout=dropout,
            use_track_metadata=use_track_metadata,
            num_conditions=num_conditions,
            num_cell_lines=num_cell_lines,
            num_assay_types=num_assay_types,
            num_targets=num_targets,
            condition_shared_track_index=condition_shared_track_index,
            activation_type=activation_type,
        )
        self.metadata_to_scale = nn.Linear(metadata_hidden, 1) if use_track_metadata else None
        self.metadata_to_shift = nn.Linear(metadata_hidden, 1) if use_track_metadata else None
        for layer in (self.metadata_to_scale, self.metadata_to_shift):
            if layer is not None:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        track_condition_ids: torch.Tensor | None = None,
        track_timepoint_minutes: torch.Tensor | None = None,
        track_cell_line_ids: torch.Tensor | None = None,
        track_assay_type_ids: torch.Tensor | None = None,
        track_target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self._expand_condition_shared_tracks(self.proj(x))
        if self.metadata_to_scale is not None and self.metadata_to_shift is not None:
            metadata = self._encode_metadata(
                device=logits.device,
                dtype=logits.dtype,
                track_condition_ids=track_condition_ids,
                track_timepoint_minutes=track_timepoint_minutes,
                track_cell_line_ids=track_cell_line_ids,
                track_assay_type_ids=track_assay_type_ids,
                track_target_ids=track_target_ids,
            )
            if metadata is not None:
                scale = 1.0 + torch.tanh(self.metadata_to_scale(metadata).squeeze(-1))
                shift = self.metadata_to_shift(metadata).squeeze(-1)
                if scale.ndim == 1:
                    scale = scale.unsqueeze(0)
                    shift = shift.unsqueeze(0)
                logits = logits * scale.unsqueeze(-1) + shift.unsqueeze(-1)
        return self.activation(logits)


class LogFiLMPerturbHead(_PerturbHeadBase):
    def __init__(
        self,
        *,
        in_ch: int = 1920,
        hidden: int = 512,
        n_tracks: int = 12,
        use_track_metadata: bool = False,
        num_conditions: int = 0,
        num_cell_lines: int = 0,
        num_assay_types: int = 0,
        num_targets: int = 0,
        metadata_hidden: int = 32,
        dropout: float = 0.0,
        condition_shared_track_index: Sequence[int] | None = None,
        activation_type: ActivationType = "softplus",
    ):
        super().__init__(
            in_ch=in_ch,
            hidden=hidden,
            n_tracks=n_tracks,
            metadata_hidden=metadata_hidden,
            dropout=dropout,
            use_track_metadata=use_track_metadata,
            num_conditions=num_conditions,
            num_cell_lines=num_cell_lines,
            num_assay_types=num_assay_types,
            num_targets=num_targets,
            condition_shared_track_index=condition_shared_track_index,
            activation_type=activation_type,
        )
        self.metadata_to_log_scale = (
            nn.Linear(metadata_hidden, hidden) if use_track_metadata else None
        )
        if self.metadata_to_log_scale is not None:
            nn.init.zeros_(self.metadata_to_log_scale.weight)
            nn.init.zeros_(self.metadata_to_log_scale.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        track_condition_ids: torch.Tensor | None = None,
        track_timepoint_minutes: torch.Tensor | None = None,
        track_cell_line_ids: torch.Tensor | None = None,
        track_assay_type_ids: torch.Tensor | None = None,
        track_target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.proj[:3](x)
        if self.metadata_to_log_scale is None:
            logits = self._expand_condition_shared_tracks(self.proj[3](hidden))
            return self.activation(logits)

        metadata = self._encode_metadata(
            device=x.device,
            dtype=x.dtype,
            track_condition_ids=track_condition_ids,
            track_timepoint_minutes=track_timepoint_minutes,
            track_cell_line_ids=track_cell_line_ids,
            track_assay_type_ids=track_assay_type_ids,
            track_target_ids=track_target_ids,
        )
        if metadata is None:
            logits = self._expand_condition_shared_tracks(self.proj[3](hidden))
            return self.activation(logits)

        log_scale = self.metadata_to_log_scale(metadata)
        if log_scale.ndim == 2:
            log_scale = log_scale.unsqueeze(0)
        elif log_scale.ndim == 3:
            log_scale = log_scale[0].unsqueeze(0)
        track_hidden = hidden.unsqueeze(1) * torch.exp(log_scale.unsqueeze(-1))
        weight, bias = self._shared_output_weights()
        logits = torch.einsum("bthl,th->btl", track_hidden, weight)
        if bias is not None:
            logits = logits + bias.view(1, -1, 1)
        return self.activation(logits)


class ResidualFiLMPerturbHead(_PerturbHeadBase):
    def __init__(
        self,
        *,
        in_ch: int = 1920,
        hidden: int = 768,
        n_tracks: int = 12,
        use_track_metadata: bool = False,
        num_conditions: int = 0,
        num_cell_lines: int = 0,
        num_assay_types: int = 0,
        num_targets: int = 0,
        metadata_hidden: int = 64,
        dropout: float = 0.0,
        condition_shared_track_index: Sequence[int] | None = None,
        refinement_kernel: int = 9,
        activation_type: ActivationType = "softplus",
    ):
        super().__init__(
            in_ch=in_ch,
            hidden=hidden,
            n_tracks=n_tracks,
            metadata_hidden=metadata_hidden,
            dropout=dropout,
            use_track_metadata=use_track_metadata,
            num_conditions=num_conditions,
            num_cell_lines=num_cell_lines,
            num_assay_types=num_assay_types,
            num_targets=num_targets,
            condition_shared_track_index=condition_shared_track_index,
            activation_type=activation_type,
        )
        self.metadata_to_log_scale = (
            nn.Linear(metadata_hidden, hidden) if use_track_metadata else None
        )
        self.metadata_to_shift = nn.Linear(metadata_hidden, hidden) if use_track_metadata else None
        if self.metadata_to_log_scale is not None:
            nn.init.zeros_(self.metadata_to_log_scale.weight)
            nn.init.zeros_(self.metadata_to_log_scale.bias)
        if self.metadata_to_shift is not None:
            nn.init.zeros_(self.metadata_to_shift.weight)
            nn.init.zeros_(self.metadata_to_shift.bias)

        self.refine = nn.Sequential(
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=refinement_kernel,
                padding=refinement_kernel // 2,
                groups=hidden,
            ),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
        )
        self.refine_norm = nn.BatchNorm1d(hidden)

    def forward(
        self,
        x: torch.Tensor,
        *,
        track_condition_ids: torch.Tensor | None = None,
        track_timepoint_minutes: torch.Tensor | None = None,
        track_cell_line_ids: torch.Tensor | None = None,
        track_assay_type_ids: torch.Tensor | None = None,
        track_target_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.proj[:3](x)
        if self.metadata_to_log_scale is None or self.metadata_to_shift is None:
            hidden_refined = self.refine_norm(hidden + self.refine(hidden))
            logits = self._expand_condition_shared_tracks(self.proj[3](hidden_refined))
            return self.activation(logits)

        metadata = self._encode_metadata(
            device=x.device,
            dtype=x.dtype,
            track_condition_ids=track_condition_ids,
            track_timepoint_minutes=track_timepoint_minutes,
            track_cell_line_ids=track_cell_line_ids,
            track_assay_type_ids=track_assay_type_ids,
            track_target_ids=track_target_ids,
        )
        if metadata is None:
            hidden_refined = self.refine_norm(hidden + self.refine(hidden))
            logits = self._expand_condition_shared_tracks(self.proj[3](hidden_refined))
            return self.activation(logits)

        log_scale = self.metadata_to_log_scale(metadata)
        shift = self.metadata_to_shift(metadata)
        if log_scale.ndim == 2:
            log_scale = log_scale.unsqueeze(0)
            shift = shift.unsqueeze(0)
        elif log_scale.ndim == 3:
            log_scale = log_scale[0].unsqueeze(0)
            shift = shift[0].unsqueeze(0)
        conditioned = hidden.unsqueeze(1) * torch.exp(log_scale.unsqueeze(-1))
        conditioned = conditioned + shift.unsqueeze(-1)
        batch_size, n_tracks, hidden_dim, seq_len = conditioned.shape
        conditioned = conditioned.reshape(batch_size * n_tracks, hidden_dim, seq_len)
        refined = self.refine_norm(conditioned + self.refine(conditioned))
        refined = refined.reshape(batch_size, n_tracks, hidden_dim, seq_len)

        weight, bias = self._shared_output_weights()
        logits = torch.einsum("bthl,th->btl", refined, weight)
        if bias is not None:
            logits = logits + bias.view(1, -1, 1)
        return self.activation(logits)


class TransferMLPPerturbHead(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int = 1920,
        hidden: int = 512,
        n_tracks: int = 12,
        mlp_hidden: int | None = None,
        dropout: float = 0.0,
        activation_type: ActivationType = "softplus",
        **_: object,
    ):
        super().__init__()
        mlp_hidden = mlp_hidden or hidden
        self.proj = nn.Sequential(
            nn.Conv1d(in_ch, mlp_hidden, 1),
            nn.GELU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(mlp_hidden, hidden, 1),
            nn.GELU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(hidden, n_tracks, 1),
        )
        if activation_type == "softplus_beta2":
            self.activation: nn.Module = nn.Softplus(beta=2)
        elif activation_type == "exp":
            self.activation = _ClampedExp()
        elif activation_type == "identity":
            self.activation = nn.Identity()
        else:
            self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor, **_: torch.Tensor | None) -> torch.Tensor:
        return self.activation(self.proj(x))


def build_perturb_head(
    *,
    head_type: HeadType,
    activation_type: ActivationType = "softplus",
    **kwargs: object,
) -> nn.Module:
    constructors: dict[HeadType, type[nn.Module]] = {
        "bias": PerturbHead,
        "film": FiLMPerturbHead,
        "log_film": LogFiLMPerturbHead,
        "residual_film": ResidualFiLMPerturbHead,
        "transfer_mlp": TransferMLPPerturbHead,
    }
    try:
        constructor = constructors[head_type]
    except KeyError as exc:
        raise ValueError(f"Unknown head type {head_type!r}") from exc
    return constructor(activation_type=activation_type, **kwargs)
