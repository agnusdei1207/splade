//! Can tract run the int8 sparse encoder at all?
//!
//! pentesting's runtime is tract, and its dense encoder is already a quantized
//! ONNX, so quantized graphs work in principle. The sparse export is a different
//! shape though: dynamic quantization emits `DynamicQuantizeLinear` and
//! `MatMulInteger`, and the graph ends in a `TopK` that tract has to keep. If any
//! of that is unsupported the whole int8 plan collapses, so this is checked before
//! anything is built on top of it.
//!
//! Skips itself when the artifact is absent, since the export lives outside git.

use std::path::PathBuf;

use tract_onnx::prelude::*;

fn artifact() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("artifacts/quant/mini/document-int8.onnx")
}

#[test]
fn tract_loads_and_runs_the_int8_sparse_encoder() {
    let path = artifact();
    if !path.is_file() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }

    let inference = tract_onnx::onnx()
        .model_for_path(&path)
        .expect("tract must parse the quantized sparse graph");

    let input_names = inference
        .inputs
        .iter()
        .map(|outlet| inference.node(outlet.node).name.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        input_names,
        vec!["input_ids", "attention_mask", "token_type_ids"],
        "input contract must match the exporter"
    );

    let model = inference
        .into_optimized()
        .expect("tract must optimize the quantized graph")
        .into_runnable()
        .expect("tract must make the quantized graph runnable");

    // A short realistic sequence: [CLS] plus a few tokens plus [SEP].
    let ids: Vec<i64> = vec![101, 7592, 2088, 4324, 102];
    let sequence = ids.len();
    let tensor = |values: Vec<i64>| -> TValue {
        tract_ndarray::Array2::from_shape_vec((1, sequence), values)
            .expect("shape")
            .into_tensor()
            .into()
    };
    let outputs = model
        .run(tvec![
            tensor(ids.clone()),
            tensor(vec![1; sequence]),
            tensor(vec![0; sequence]),
        ])
        .expect("quantized inference must run");

    assert_eq!(outputs.len(), 2, "graph returns (values, term_ids)");
    let values = outputs[0]
        .to_plain_array_view::<f32>()
        .expect("values are f32")
        .to_owned();
    let term_ids = outputs[1]
        .to_plain_array_view::<i64>()
        .expect("term ids are i64")
        .to_owned();

    assert_eq!(values.shape(), &[1, 512]);
    assert_eq!(term_ids.shape(), &[1, 512]);

    // SPLADE pooling is relu(log1p(..)) so weights cannot be negative, and topk
    // returns them sorted, which the index relies on when it truncates.
    let row = (0..512).map(|i| values[[0, i]]).collect::<Vec<f32>>();
    assert!(row.iter().all(|weight| *weight >= 0.0));
    assert!(
        row.windows(2).all(|pair| pair[0] >= pair[1]),
        "topk output must stay sorted descending"
    );
    assert!(
        row.iter().any(|weight| *weight > 0.0),
        "a real sentence must activate at least one term"
    );

    let vocab_ceiling = 30_522;
    assert!(
        term_ids
            .iter()
            .all(|term| *term >= 0 && *term < vocab_ceiling),
        "term ids must stay inside the vocabulary"
    );
    assert!(
        term_ids.iter().all(|term| u16::try_from(*term).is_ok()),
        "this vocabulary fits u16, so the index needs no widening"
    );
}
