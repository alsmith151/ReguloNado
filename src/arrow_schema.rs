use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, UInt8Array};
use arrow_schema::{DataType, Field, Schema};
use std::collections::HashMap;
use std::sync::Arc;

/// Build the Arrow schema for a window Parquet shard.
///
/// `sequence_tokens` is a `FixedSizeList<uint8, stored_context>` (A0 C1 G2 T3, 4 = N/padding)
/// and `signal` is a `FixedSizeList<FixedSizeList<float32, n_bins>, n_tracks>`, matching the
/// `datasets` `List(..., length=)` features written by the Python build. The `huggingface`
/// schema metadata is passed in verbatim from Python (`Features.to_dict()` JSON) so it cannot
/// drift from what `datasets` itself expects.
pub(crate) fn window_arrow_schema(
    stored_context: usize,
    n_tracks: usize,
    n_bins: usize,
    hf_features_json: String,
) -> Arc<Schema> {
    let sequence_tokens_type = DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::UInt8, true)),
        stored_context as i32,
    );
    let signal_inner_type = DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        n_bins as i32,
    );
    let signal_type = DataType::FixedSizeList(
        Arc::new(Field::new("item", signal_inner_type, true)),
        n_tracks as i32,
    );
    let metadata = HashMap::from([("huggingface".to_string(), hf_features_json)]);
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("sequence_tokens", sequence_tokens_type, true),
            Field::new("signal", signal_type, true),
            Field::new("interval", DataType::Utf8, false),
            Field::new("index", DataType::Int64, false),
            Field::new("local_index", DataType::Int64, false),
        ],
        metadata,
    ))
}

/// Wrap a flat row-major `u8` token buffer as a `FixedSizeList<uint8, stored_context>` array.
pub(crate) fn sequence_tokens_array(values: Vec<u8>, stored_context: usize) -> ArrayRef {
    let values_array = Arc::new(UInt8Array::from(values)) as ArrayRef;
    let field = Arc::new(Field::new("item", DataType::UInt8, true));
    Arc::new(FixedSizeListArray::new(
        field,
        stored_context as i32,
        values_array,
        None,
    ))
}

/// Wrap a flat row-major `f32` signal buffer as a
/// `FixedSizeList<FixedSizeList<float32, n_bins>, n_tracks>` array.
pub(crate) fn signal_array(values: Vec<f32>, n_tracks: usize, n_bins: usize) -> ArrayRef {
    let values_array = Arc::new(Float32Array::from(values)) as ArrayRef;
    let inner_field = Arc::new(Field::new("item", DataType::Float32, true));
    let inner = Arc::new(FixedSizeListArray::new(
        inner_field,
        n_bins as i32,
        values_array,
        None,
    )) as ArrayRef;
    let outer_field = Arc::new(Field::new(
        "item",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            n_bins as i32,
        ),
        true,
    ));
    Arc::new(FixedSizeListArray::new(
        outer_field,
        n_tracks as i32,
        inner,
        None,
    ))
}
