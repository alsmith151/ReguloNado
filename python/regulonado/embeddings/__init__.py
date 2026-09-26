"""Frozen-backbone embedding cache: :mod:`regulonado.embeddings.cache`.

See :mod:`regulonado.embeddings.cache` for the on-disk format and tiling/pooling rules.
"""

from __future__ import annotations

from regulonado.embeddings.cache import (
    MANIFEST_FILENAME,
    EmbeddingManifest,
    EmbeddingStore,
    embed_regions,
    read_manifest,
    region_table_hash,
    validate_manifest,
)

__all__ = [
    "MANIFEST_FILENAME",
    "EmbeddingManifest",
    "EmbeddingStore",
    "embed_regions",
    "read_manifest",
    "region_table_hash",
    "validate_manifest",
]
