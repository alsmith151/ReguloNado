from __future__ import annotations

from transformers import PretrainedConfig


class RegulonadoConfig(PretrainedConfig):
    """Configuration for RegulonadoModel.

    Stores everything needed to reconstruct the model architecture and run inference without
    any external metadata files. Saved as ``config.json`` alongside weights by ``save_pretrained``.
    """

    model_type = "regulonado"

    def __init__(
        self,
        # Backbone
        backbone_type: str = "borzoi",
        pretrained_name: str | None = None,
        config_overrides: dict | None = None,
        target_length: int | None = None,
        # Head
        head_type: str = "residual_film",
        head_hidden: int = 512,
        head_dropout: float = 0.0,
        refinement_kernel: int = 9,
        mlp_hidden: int | None = None,
        # Architecture dimensions
        n_tracks: int = 1,
        feature_dim: int = 1920,
        # Track metadata conditioning
        use_track_metadata: bool = False,
        activation_type: str = "softplus",
        num_conditions: int = 0,
        num_cell_lines: int = 0,
        num_assay_types: int = 0,
        num_targets: int = 0,
        metadata_hidden: int = 32,
        condition_shared_track_index: list[int] | None = None,
        # Inference geometry (stored for convenience, not used by forward())
        context_length: int = 524_288,
        n_pred_bins: int = 6_144,
        bin_size: int = 32,
        track_names: list[str] | None = None,
        track_metadata: dict[str, list[int | float | None]] | None = None,
        # Backward-compat: path to the dataset used at training time
        data_path: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.backbone_type = backbone_type
        self.pretrained_name = pretrained_name
        self.config_overrides = config_overrides or {}
        self.target_length = target_length
        self.head_type = head_type
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout
        self.refinement_kernel = refinement_kernel
        self.mlp_hidden = mlp_hidden
        self.n_tracks = n_tracks
        self.feature_dim = feature_dim
        self.use_track_metadata = use_track_metadata
        self.activation_type = activation_type
        self.num_conditions = num_conditions
        self.num_cell_lines = num_cell_lines
        self.num_assay_types = num_assay_types
        self.num_targets = num_targets
        self.metadata_hidden = metadata_hidden
        self.condition_shared_track_index = condition_shared_track_index or []
        self.context_length = context_length
        self.n_pred_bins = n_pred_bins
        self.bin_size = bin_size
        self.track_names = track_names or []
        self.track_metadata = track_metadata or {}
        self.data_path = data_path
