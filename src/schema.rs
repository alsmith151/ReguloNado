use arrow_array::{
    builder::{Float32Builder, Int8Builder, ListBuilder},
    ArrayRef, Float32Array, Int8Array, ListArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Schema};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

/// Build the Arrow schema expected by HuggingFace `datasets`.
///
/// `datasets.Array2D` is represented as nested Arrow lists with extension
/// metadata on the top-level fields. Without this metadata,
/// `datasets.load_from_disk` infers plain `List(List(...))` features and rejects
/// the saved `dataset_info.json`.
pub(crate) fn hf_arrow_schema(context_len: usize, n_tracks: usize, n_bins: usize) -> Arc<Schema> {
    let input_type = DataType::List(Arc::new(Field::new(
        "item",
        DataType::List(Arc::new(Field::new("item", DataType::Int8, true))),
        true,
    )));
    let label_type = DataType::List(Arc::new(Field::new(
        "item",
        DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
        true,
    )));
    let metadata = HashMap::from([(
        "huggingface".to_string(),
        json!({
            "info": {
                "features": {
                    "input_ids": {"shape": [4, context_len], "dtype": "int8", "_type": "Array2D"},
                    "labels": {"shape": [n_tracks, n_bins], "dtype": "float32", "_type": "Array2D"},
                    "interval": {"dtype": "string", "_type": "Value"},
                    "index": {"dtype": "int64", "_type": "Value"},
                    "local_index": {"dtype": "int64", "_type": "Value"}
                }
            }
        })
        .to_string(),
    )]);
    let extension_name = "datasets.features.features.Array2DExtensionType".to_string();
    let input_field = Field::new("input_ids", input_type, true).with_metadata(HashMap::from([
        ("ARROW:extension:name".to_string(), extension_name.clone()),
        (
            "ARROW:extension:metadata".to_string(),
            json!([[4, context_len], "int8"]).to_string(),
        ),
    ]));
    let label_field = Field::new("labels", label_type, true).with_metadata(HashMap::from([
        ("ARROW:extension:name".to_string(), extension_name),
        (
            "ARROW:extension:metadata".to_string(),
            json!([[n_tracks, n_bins], "float32"]).to_string(),
        ),
    ]));
    Arc::new(Schema::new_with_metadata(
        vec![
            input_field,
            label_field,
            Field::new("interval", DataType::Utf8, false),
            Field::new("index", DataType::Int64, false),
            Field::new("local_index", DataType::Int64, false),
        ],
        metadata,
    ))
}

/// Append one row of a 2D int8 tensor using Arrow builders.
///
/// Retained for the sample-major compatibility writer. The direct track-major
/// writer uses zero-copy-ish buffer construction via `make_2d_i8_array`.
pub(crate) fn append_2d_i8(
    builder: &mut ListBuilder<ListBuilder<Int8Builder>>,
    values: &[i8],
    rows: usize,
    cols: usize,
) {
    for row in 0..rows {
        let start = row * cols;
        builder
            .values()
            .values()
            .append_slice(&values[start..start + cols]);
        builder.values().append(true);
    }
    builder.append(true);
}

/// Append one row of a 2D float32 tensor using Arrow builders.
///
/// Retained for the sample-major compatibility writer. The direct track-major
/// writer uses buffer construction via `make_2d_f32_array`, which is faster for
/// large dense label tensors.
pub(crate) fn append_2d_f32(
    builder: &mut ListBuilder<ListBuilder<Float32Builder>>,
    values: &[f32],
    rows: usize,
    cols: usize,
) {
    for row in 0..rows {
        let start = row * cols;
        builder
            .values()
            .values()
            .append_slice(&values[start..start + cols]);
        builder.values().append(true);
    }
    builder.append(true);
}

/// Construct monotonically increasing Arrow list offsets for fixed-width rows.
///
/// Panics if `count * width` exceeds `i32::MAX`, since Arrow's ListArray uses
/// i32 offsets.  Callers that build large 2D arrays should validate the batch
/// size before calling this.
pub(crate) fn offsets(count: usize, width: usize) -> OffsetBuffer<i32> {
    let max_offset: usize = count.checked_mul(width).unwrap_or(usize::MAX);
    assert!(
        max_offset <= i32::MAX as usize,
        "Arrow i32 offset overflow: {count} * {width} = {max_offset} > i32::MAX ({}). \
         Reduce --arrow-batch-size.",
        i32::MAX
    );
    let mut offsets = Vec::with_capacity(count + 1);
    for i in 0..=count {
        offsets.push((i * width) as i32);
    }
    OffsetBuffer::new(ScalarBuffer::from(offsets))
}

/// Wrap a flat row-major buffer as an Arrow Array2D-compatible nested list.
///
/// The logical shape is `(rows, inner_rows, cols)`, but Arrow stores the values
/// as one contiguous primitive array plus two offset buffers.
pub(crate) fn make_2d_i8_array(
    values: Vec<i8>,
    rows: usize,
    inner_rows: usize,
    cols: usize,
) -> ArrayRef {
    let values_array = Arc::new(Int8Array::from(values)) as ArrayRef;
    let inner_field = Arc::new(Field::new("item", DataType::Int8, true));
    let inner = Arc::new(ListArray::new(
        inner_field,
        offsets(rows * inner_rows, cols),
        values_array,
        None,
    )) as ArrayRef;
    let outer_field = Arc::new(Field::new(
        "item",
        DataType::List(Arc::new(Field::new("item", DataType::Int8, true))),
        true,
    ));
    Arc::new(ListArray::new(
        outer_field,
        offsets(rows, inner_rows),
        inner,
        None,
    )) as ArrayRef
}

/// Wrap a flat row-major float32 buffer as an Arrow Array2D-compatible nested list.
pub(crate) fn make_2d_f32_array(
    values: Vec<f32>,
    rows: usize,
    inner_rows: usize,
    cols: usize,
) -> ArrayRef {
    let values_array = Arc::new(Float32Array::from(values)) as ArrayRef;
    let inner_field = Arc::new(Field::new("item", DataType::Float32, true));
    let inner = Arc::new(ListArray::new(
        inner_field,
        offsets(rows * inner_rows, cols),
        values_array,
        None,
    )) as ArrayRef;
    let outer_field = Arc::new(Field::new(
        "item",
        DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
        true,
    ));
    Arc::new(ListArray::new(
        outer_field,
        offsets(rows, inner_rows),
        inner,
        None,
    )) as ArrayRef
}
