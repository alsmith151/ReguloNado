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
    """Build a track index that groups tracks by shared condition metadata.

    Derives a set of non-condition metadata fields from track records (e.g.
    assay type, cell line, target) and assigns a base track index to each
    unique combination. This allows prediction heads to share base channels
    across tracks that differ only in their condition ID.

    Parameters
    ----------
    track_records : Sequence[Mapping[str, object]]
        Sequence of track metadata records, each containing fields like
        condition_id, assay_type, cell_line, etc.

    Returns
    -------
    list[int]
        List of base track indices, one per track, where tracks with identical
        non-condition metadata share the same index.

    Raises
    ------
    ValueError
        If no non-condition metadata fields are found in the records.
    """
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
    """Encode track metadata (conditions, cell lines, assays, timepoints) into
    embeddings.

    Embeds categorical track metadata (condition_id, cell_line_id, etc.) and
    processes timepoint as a continuous value via MLP. Combines embeddings
    additively and applies a residual gating layer. Can be disabled entirely
    by setting ``use_track_metadata=False``.

    Parameters
    ----------
    use_track_metadata : bool
        Whether to enable metadata encoding. If False, ``forward`` returns None
        regardless of inputs.
    metadata_hidden : int
        Dimension of embedding and hidden layers.
    dropout : float
        Dropout rate applied after embeddings and in the timepoint MLP.
    num_conditions : int, optional
        Size of condition_id vocabulary. Disable by 0 or False.
    num_cell_lines : int, optional
        Size of cell_line_id vocabulary. Disable by 0 or False.
    num_assay_types : int, optional
        Size of assay_type_id vocabulary. Disable by 0 or False.
    num_targets : int, optional
        Size of target_id vocabulary. Disable by 0 or False.
    """

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
        """Encode track metadata into a dense representation.

        Embeds all provided metadata IDs, processes timepoint, sums embeddings,
        applies dropout, then a residual mixer layer. Returns None if
        use_track_metadata is False or all metadata inputs are None.

        Parameters
        ----------
        track_condition_ids : torch.Tensor | None, optional
            Condition ID indices, shape [n_tracks] with -1 as masked value.
        track_timepoint_minutes : torch.Tensor | None, optional
            Timepoint values in minutes, shape [n_tracks] with NaN as masked.
        track_cell_line_ids : torch.Tensor | None, optional
            Cell line ID indices, shape [n_tracks] with -1 as masked value.
        track_assay_type_ids : torch.Tensor | None, optional
            Assay type ID indices, shape [n_tracks] with -1 as masked value.
        track_target_ids : torch.Tensor | None, optional
            Target ID indices, shape [n_tracks] with -1 as masked value.
        device : torch.device
            Device to place outputs on.
        dtype : torch.dtype
            Data type of output tensor.

        Returns
        -------
        torch.Tensor | None
            Encoded metadata tensor of shape [n_tracks, metadata_hidden] and
            the specified dtype, or None if metadata is disabled or all inputs
            are None.
        """
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
    """Base class for perturbation prediction heads.

    Common initialization for all head variants. Subclasses override ``forward``
    to apply different modulation strategies (bias offset, FiLM, residual, MLP).
    Manages track metadata encoding and optional condition-shared grouping.
    """

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
        """Initialize the perturbation head base.

        Parameters
        ----------
        in_ch : int
            Number of input feature channels from the backbone.
        hidden : int
            Number of hidden channels in the projection MLP.
        n_tracks : int
            Total number of tracks (output channels).
        metadata_hidden : int
            Dimension of metadata embeddings.
        dropout : float
            Dropout rate in the projection and metadata encoder.
        use_track_metadata : bool
            Whether to encode track metadata for modulation.
        num_conditions : int
            Vocabulary size of condition_id. Set to 0 to disable.
        num_cell_lines : int
            Vocabulary size of cell_line_id. Set to 0 to disable.
        num_assay_types : int
            Vocabulary size of assay_type_id. Set to 0 to disable.
        num_targets : int
            Vocabulary size of target_id. Set to 0 to disable.
        condition_shared_track_index : Sequence[int] | None, optional
            Map from track index to a base track index, allowing multiple
            condition-specific tracks to share output channels.
        activation_type : ActivationType, optional
            Output activation function: "softplus" (default), "softplus_beta2",
            "exp", or "identity".

        Raises
        ------
        ValueError
            If condition_shared_track_index length does not match n_tracks.
        """
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
    """Simple perturbation head: projects to logits and optionally adds
    metadata-derived bias.

    When metadata is disabled, this head reduces to a pure projection. When
    enabled, each track receives a scalar bias term conditioned on its metadata
    via a single-output linear layer.

    Parameters
    ----------
    in_ch : int, optional
        Input channels from backbone, by default 1920.
    hidden : int, optional
        Hidden channels in projection, by default 512.
    n_tracks : int, optional
        Number of output tracks, by default 12.
    use_track_metadata : bool, optional
        Enable metadata-derived bias, by default False.
    num_conditions : int, optional
        Condition vocabulary size, by default 0 (disabled).
    num_cell_lines : int, optional
        Cell line vocabulary size, by default 0 (disabled).
    num_assay_types : int, optional
        Assay type vocabulary size, by default 0 (disabled).
    num_targets : int, optional
        Target vocabulary size, by default 0 (disabled).
    metadata_hidden : int, optional
        Metadata embedding dimension, by default 32.
    dropout : float, optional
        Dropout rate, by default 0.0.
    condition_shared_track_index : Sequence[int] | None, optional
        Track grouping map for condition sharing, by default None.
    activation_type : ActivationType, optional
        Output activation function, by default "softplus".
    """

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
        """Project backbone features and optionally apply metadata bias.

        Parameters
        ----------
        x : torch.Tensor
            Backbone features, shape [batch, in_ch, length].
        track_condition_ids : torch.Tensor | None, optional
            Condition indices for each track.
        track_timepoint_minutes : torch.Tensor | None, optional
            Timepoint values in minutes for each track.
        track_cell_line_ids : torch.Tensor | None, optional
            Cell line indices for each track.
        track_assay_type_ids : torch.Tensor | None, optional
            Assay type indices for each track.
        track_target_ids : torch.Tensor | None, optional
            Target indices for each track.

        Returns
        -------
        torch.Tensor
            Predictions with shape [batch, n_tracks, length], after projection
            and optional bias addition.
        """
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
    """FiLM-modulated perturbation head: applies feature-wise affine transform.

    When metadata is disabled, identical to PerturbHead. When enabled, applies
    metadata-derived scale (multiplicative) and shift (additive) to the logits
    before activation: ``output = activation(logits * scale + shift)``.

    Parameters
    ----------
    in_ch : int, optional
        Input channels from backbone, by default 1920.
    hidden : int, optional
        Hidden channels in projection, by default 512.
    n_tracks : int, optional
        Number of output tracks, by default 12.
    use_track_metadata : bool, optional
        Enable FiLM modulation, by default False.
    num_conditions : int, optional
        Condition vocabulary size, by default 0 (disabled).
    num_cell_lines : int, optional
        Cell line vocabulary size, by default 0 (disabled).
    num_assay_types : int, optional
        Assay type vocabulary size, by default 0 (disabled).
    num_targets : int, optional
        Target vocabulary size, by default 0 (disabled).
    metadata_hidden : int, optional
        Metadata embedding dimension, by default 32.
    dropout : float, optional
        Dropout rate, by default 0.0.
    condition_shared_track_index : Sequence[int] | None, optional
        Track grouping map for condition sharing, by default None.
    activation_type : ActivationType, optional
        Output activation function, by default "softplus".
    """

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
        """Apply FiLM transformation: scale and shift logits by metadata.

        Modulation factors are derived from metadata embeddings: scale is
        1 + tanh(...) to keep centered near 1, and shift is unbounded.

        Parameters
        ----------
        x : torch.Tensor
            Backbone features, shape [batch, in_ch, length].
        track_condition_ids : torch.Tensor | None, optional
            Condition indices for each track.
        track_timepoint_minutes : torch.Tensor | None, optional
            Timepoint values in minutes for each track.
        track_cell_line_ids : torch.Tensor | None, optional
            Cell line indices for each track.
        track_assay_type_ids : torch.Tensor | None, optional
            Assay type indices for each track.
        track_target_ids : torch.Tensor | None, optional
            Target indices for each track.

        Returns
        -------
        torch.Tensor
            Predictions with shape [batch, n_tracks, length], after FiLM
            modulation and activation.
        """
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
    """FiLM modulation in log space: scales hidden features before projection.

    More expressive than FiLM on logits. Metadata-derived scaling factors (in
    log space) are applied element-wise to the hidden representation before
    the final linear projection to tracks. This allows per-hidden-channel
    scaling, not just per-track.

    Parameters
    ----------
    in_ch : int, optional
        Input channels from backbone, by default 1920.
    hidden : int, optional
        Hidden channels in projection, by default 512.
    n_tracks : int, optional
        Number of output tracks, by default 12.
    use_track_metadata : bool, optional
        Enable log-space FiLM modulation, by default False.
    num_conditions : int, optional
        Condition vocabulary size, by default 0 (disabled).
    num_cell_lines : int, optional
        Cell line vocabulary size, by default 0 (disabled).
    num_assay_types : int, optional
        Assay type vocabulary size, by default 0 (disabled).
    num_targets : int, optional
        Target vocabulary size, by default 0 (disabled).
    metadata_hidden : int, optional
        Metadata embedding dimension, by default 32.
    dropout : float, optional
        Dropout rate, by default 0.0.
    condition_shared_track_index : Sequence[int] | None, optional
        Track grouping map for condition sharing, by default None.
    activation_type : ActivationType, optional
        Output activation function, by default "softplus".
    """

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
        """Apply per-track log-space scaling to hidden features.

        Extracts hidden features, then scales each track's hidden representation
        by metadata-derived log-scale factors before projecting to outputs.

        Parameters
        ----------
        x : torch.Tensor
            Backbone features, shape [batch, in_ch, length].
        track_condition_ids : torch.Tensor | None, optional
            Condition indices for each track.
        track_timepoint_minutes : torch.Tensor | None, optional
            Timepoint values in minutes for each track.
        track_cell_line_ids : torch.Tensor | None, optional
            Cell line indices for each track.
        track_assay_type_ids : torch.Tensor | None, optional
            Assay type indices for each track.
        track_target_ids : torch.Tensor | None, optional
            Target indices for each track.

        Returns
        -------
        torch.Tensor
            Predictions with shape [batch, n_tracks, length], after per-track
            log-space scaling and output projection.
        """
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
    """Residual FiLM with refinement: log-space scaling plus learned refinement.

    Combines log-space per-track scaling with residual connections and
    depth-wise separable convolutions for local context refinement. This is
    the most expressive head variant, allowing per-track and per-position
    modulation followed by learned perturbation-specific filtering.

    Parameters
    ----------
    in_ch : int, optional
        Input channels from backbone, by default 1920.
    hidden : int, optional
        Hidden channels, by default 768.
    n_tracks : int, optional
        Number of output tracks, by default 12.
    use_track_metadata : bool, optional
        Enable metadata modulation, by default False.
    num_conditions : int, optional
        Condition vocabulary size, by default 0 (disabled).
    num_cell_lines : int, optional
        Cell line vocabulary size, by default 0 (disabled).
    num_assay_types : int, optional
        Assay type vocabulary size, by default 0 (disabled).
    num_targets : int, optional
        Target vocabulary size, by default 0 (disabled).
    metadata_hidden : int, optional
        Metadata embedding dimension, by default 64.
    dropout : float, optional
        Dropout rate, by default 0.0.
    condition_shared_track_index : Sequence[int] | None, optional
        Track grouping map for condition sharing, by default None.
    refinement_kernel : int, optional
        Kernel size for first refinement convolution, by default 9.
    activation_type : ActivationType, optional
        Output activation function, by default "softplus".
    """

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
        """Apply log-space FiLM plus learned refinement to hidden features.

        Extracts hidden features, optionally applies per-track log-scale and
        shift based on metadata, then refines via depth-wise convolutions with
        residual and batch normalization before final projection.

        Parameters
        ----------
        x : torch.Tensor
            Backbone features, shape [batch, in_ch, length].
        track_condition_ids : torch.Tensor | None, optional
            Condition indices for each track.
        track_timepoint_minutes : torch.Tensor | None, optional
            Timepoint values in minutes for each track.
        track_cell_line_ids : torch.Tensor | None, optional
            Cell line indices for each track.
        track_assay_type_ids : torch.Tensor | None, optional
            Assay type indices for each track.
        track_target_ids : torch.Tensor | None, optional
            Target indices for each track.

        Returns
        -------
        torch.Tensor
            Predictions with shape [batch, n_tracks, length], after log-space
            modulation, refinement, and output projection.
        """
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
    """Simple MLP projection head: no metadata or FiLM modulation.

    Pure feed-forward network with two hidden layers and gating. Does not
    support metadata modulation; metadata arguments are silently ignored.
    Suitable for transfer learning scenarios where the backbone is fixed
    and only track predictions vary.

    Parameters
    ----------
    in_ch : int, optional
        Input channels from backbone, by default 1920.
    hidden : int, optional
        First hidden layer size, by default 512.
    n_tracks : int, optional
        Number of output tracks, by default 12.
    mlp_hidden : int | None, optional
        Second MLP hidden layer size; defaults to hidden if None.
    dropout : float, optional
        Dropout rate between layers, by default 0.0.
    activation_type : ActivationType, optional
        Output activation function, by default "softplus".
    **_
        Ignored additional keyword arguments (for interface compatibility).
    """

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
        """Project backbone features through MLP.

        Parameters
        ----------
        x : torch.Tensor
            Backbone features, shape [batch, in_ch, length].
        **_
            Ignored keyword arguments (metadata is not used).

        Returns
        -------
        torch.Tensor
            Predictions with shape [batch, n_tracks, length].
        """
        return self.activation(self.proj(x))


def build_perturb_head(
    *,
    head_type: HeadType,
    activation_type: ActivationType = "softplus",
    **kwargs: object,
) -> nn.Module:
    """Instantiate a perturbation head of the specified type.

    Factory function that selects and constructs the appropriate head class
    based on the type string. All keyword arguments except head_type and
    activation_type are passed to the head constructor.

    Parameters
    ----------
    head_type : HeadType
        Type of head to build: "bias", "film", "log_film", "residual_film",
        or "transfer_mlp".
    activation_type : ActivationType, optional
        Output activation function, by default "softplus".
    **kwargs : object
        Additional keyword arguments forwarded to the head constructor
        (e.g., in_ch, hidden, n_tracks, metadata parameters).

    Returns
    -------
    nn.Module
        Constructed head instance of the requested type.

    Raises
    ------
    ValueError
        If head_type is not recognized.

    Examples
    --------
    >>> head = build_perturb_head(
    ...     head_type="film",
    ...     in_ch=1920,
    ...     n_tracks=12,
    ...     use_track_metadata=True,
    ...     num_conditions=5,
    ... )  # doctest: +SKIP
    """
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
