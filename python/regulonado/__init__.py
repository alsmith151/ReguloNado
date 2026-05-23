from regulonado.model import (
    ActivationType,
    BackboneAdapter,
    BackboneSpec,
    BackboneType,
    BorzoiBackboneAdapter,
    EnformerBackboneAdapter,
    FiLMPerturbHead,
    FreezePolicy,
    HeadedSequenceModel,
    HeadType,
    LogFiLMPerturbHead,
    PerturbHead,
    ResidualFiLMPerturbHead,
    TrackMetadataEncoder,
    TransferMLPPerturbHead,
    build_backbone_adapter,
    build_condition_shared_track_index,
    build_perturb_head,
)


def __getattr__(name: str):
    if name in {"metrics", "training", "viz"}:
        import importlib

        module = importlib.import_module(f"regulonado.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'regulonado' has no attribute {name!r}")

__all__ = [
    "ActivationType",
    "BackboneAdapter",
    "BackboneSpec",
    "BackboneType",
    "BorzoiBackboneAdapter",
    "EnformerBackboneAdapter",
    "FiLMPerturbHead",
    "FreezePolicy",
    "HeadType",
    "HeadedSequenceModel",
    "LogFiLMPerturbHead",
    "PerturbHead",
    "ResidualFiLMPerturbHead",
    "TrackMetadataEncoder",
    "TransferMLPPerturbHead",
    "build_backbone_adapter",
    "build_condition_shared_track_index",
    "build_perturb_head",
    "metrics",
    "training",
    "viz",
]
