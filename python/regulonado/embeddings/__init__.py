"""Frozen-backbone embedding cache: :mod:`regulonado.embeddings.cache`.

See :mod:`regulonado.embeddings.cache` for the on-disk format and tiling/pooling rules.
"""

from __future__ import annotations

from regulonado.embeddings.cache import (
    CHROM_SUFFIX,
    MANIFEST_FILENAME,
    EmbeddingManifest,
    EmbeddingStore,
    embed_regions,
    embedding_features,
    read_manifest,
    region_table_hash,
    validate_manifest,
    write_chrom_embeddings,
)

__all__ = [
    "CHROM_SUFFIX",
    "MANIFEST_FILENAME",
    "EmbeddingManifest",
    "EmbeddingStore",
    "embed_regions",
    "embedding_features",
    "read_manifest",
    "region_table_hash",
    "validate_manifest",
    "write_chrom_embeddings",
]
