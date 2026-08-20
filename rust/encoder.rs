use std::collections::BTreeSet;
use std::fs;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail, ensure};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tract_onnx::prelude::*;

use crate::SparseVector;

const MAX_SEQUENCE_LENGTH: usize = 512;
const DOCUMENT_TOP_K: usize = 256;
const QUERY_TOP_K: usize = 32;
const VOCAB_SIZE: usize = 30_522;
const MODEL_KEY: &str = "if-opensearch-mini";
const MODEL_REVISION: &str = "4af867a426867dfdd744097531046f4289a32fdd";

#[derive(Deserialize)]
struct ExportManifest {
    model: String,
    revision: String,
    files: std::collections::HashMap<String, AssetRecord>,
}

#[derive(Deserialize)]
struct AssetRecord {
    bytes: u64,
    sha256: String,
}

pub struct QueryEncoder {
    tokenizer: Tokenizer,
    weights: Vec<f32>,
}

impl QueryEncoder {
    pub fn from_dir(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        Self::from_dir_with_manifest(directory, directory.join("manifest.json"))
    }

    pub fn from_dir_with_manifest(
        directory: impl AsRef<Path>,
        manifest_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let directory = directory.as_ref();
        verify_assets(
            directory,
            manifest_path.as_ref(),
            &["tokenizer.json", "query.safetensors"],
        )?;
        let tokenizer = load_tokenizer(&directory.join("tokenizer.json"), false)?;
        let bytes = fs::read(directory.join("query.safetensors"))
            .context("failed to read static query weights")?;
        let tensors = SafeTensors::deserialize(&bytes)
            .map_err(|error| anyhow!(error.to_string()))
            .context("failed to parse static query weights")?;
        let tensor = tensors
            .tensor("weight")
            .map_err(|error| anyhow!(error.to_string()))
            .context("static query weight tensor is missing")?;
        ensure!(tensor.dtype() == Dtype::F32, "query weights are not f32");
        ensure!(
            tensor.shape() == [VOCAB_SIZE],
            "unexpected query weight shape"
        );
        let weights = tensor
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        ensure!(weights.len() == VOCAB_SIZE, "query weight length mismatch");
        ensure!(
            weights
                .iter()
                .all(|weight| weight.is_finite() && *weight >= 0.0),
            "query weights contain invalid values"
        );
        Ok(Self { tokenizer, weights })
    }

    pub fn encode(&self, text: &str) -> Result<SparseVector> {
        ensure!(!text.trim().is_empty(), "query is empty");
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| anyhow!(error.to_string()))
            .context("query tokenization failed")?;
        let term_ids = encoding
            .get_ids()
            .iter()
            .map(|term_id| usize::try_from(*term_id).expect("u32 fits usize"))
            .collect::<BTreeSet<_>>();
        SparseVector::try_from_pairs(
            term_ids
                .into_iter()
                .map(|term_id| (term_id, self.weights[term_id])),
            QUERY_TOP_K,
        )
    }
}

type Runnable = TypedRunnableModel;

#[derive(Clone, Copy)]
enum InputKind {
    Ids,
    AttentionMask,
    TypeIds,
}

pub struct DocumentEncoder {
    tokenizer: Tokenizer,
    model: Arc<Runnable>,
    input_kinds: Vec<InputKind>,
}

impl DocumentEncoder {
    pub fn from_dir(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        Self::from_dir_with_manifest(directory, directory.join("manifest.json"))
    }

    pub fn from_dir_with_manifest(
        directory: impl AsRef<Path>,
        manifest_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let directory = directory.as_ref();
        verify_assets(
            directory,
            manifest_path.as_ref(),
            &["tokenizer.json", "document.onnx"],
        )?;
        let tokenizer = load_tokenizer(&directory.join("tokenizer.json"), true)?;
        let inference = tract_onnx::onnx()
            .model_for_path(directory.join("document.onnx"))
            .context("failed to parse SPLADE document ONNX graph")?;
        let input_kinds = input_kinds(&inference)?;
        let model = inference
            .into_optimized()
            .context("failed to optimize SPLADE document graph")?
            .into_runnable()
            .context("failed to prepare SPLADE document graph")?;
        Ok(Self {
            tokenizer,
            model,
            input_kinds,
        })
    }

    pub fn encode(&self, text: &str) -> Result<SparseVector> {
        self.encode_batch(&[text])?
            .into_iter()
            .next()
            .context("SPLADE returned no document vector")
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<SparseVector>> {
        ensure!(!texts.is_empty(), "document batch is empty");
        ensure!(
            texts.iter().all(|text| !text.trim().is_empty()),
            "document batch contains an empty document"
        );
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| anyhow!(error.to_string()))
            .context("document tokenization failed")?;
        ensure!(
            encodings.len() == texts.len(),
            "document tokenizer batch length mismatch"
        );
        let batch = encodings.len();
        let sequence = encodings[0].len();
        ensure!(sequence > 0, "document tokenizer returned no tokens");
        ensure!(
            encodings.iter().all(|encoding| encoding.len() == sequence),
            "document tokenizer did not pad the batch"
        );
        let flatten = |select: fn(&tokenizers::Encoding) -> &[u32]| {
            encodings
                .iter()
                .flat_map(select)
                .map(|value| i64::from(*value))
                .collect::<Vec<_>>()
        };
        let ids = flatten(tokenizers::Encoding::get_ids);
        let attention = flatten(tokenizers::Encoding::get_attention_mask);
        let type_ids = flatten(tokenizers::Encoding::get_type_ids);
        let tensor = |values: Vec<i64>| -> Result<TValue> {
            Ok(
                tract_ndarray::Array2::from_shape_vec((batch, sequence), values)
                    .context("failed to shape SPLADE input")?
                    .into_tensor()
                    .into(),
            )
        };
        let mut inputs = TVec::with_capacity(self.input_kinds.len());
        for kind in &self.input_kinds {
            inputs.push(match kind {
                InputKind::Ids => tensor(ids.clone())?,
                InputKind::AttentionMask => tensor(attention.clone())?,
                InputKind::TypeIds => tensor(type_ids.clone())?,
            });
        }
        let outputs = self
            .model
            .run(inputs)
            .context("SPLADE document inference failed")?;
        ensure!(
            outputs.len() == 2,
            "SPLADE graph returned unexpected outputs"
        );
        let values = outputs[0]
            .to_plain_array_view::<f32>()
            .context("SPLADE values output was not f32")?;
        let term_ids = outputs[1]
            .to_plain_array_view::<i64>()
            .context("SPLADE term output was not i64")?;
        ensure!(
            values.shape() == [batch, DOCUMENT_TOP_K]
                && term_ids.shape() == [batch, DOCUMENT_TOP_K],
            "unexpected SPLADE output shapes {:?} and {:?}",
            values.shape(),
            term_ids.shape()
        );
        (0..batch)
            .map(|batch_index| {
                SparseVector::try_from_pairs(
                    (0..DOCUMENT_TOP_K)
                        .map(|position| {
                            let term_id = usize::try_from(term_ids[[batch_index, position]])
                                .context("SPLADE returned a negative term id")?;
                            Ok((term_id, values[[batch_index, position]]))
                        })
                        .collect::<Result<Vec<_>>>()?,
                    DOCUMENT_TOP_K,
                )
            })
            .collect()
    }
}

fn verify_assets(directory: &Path, manifest_path: &Path, names: &[&str]) -> Result<()> {
    let manifest: ExportManifest = serde_json::from_str(
        &fs::read_to_string(manifest_path)
            .with_context(|| format!("failed to read manifest {}", manifest_path.display()))?,
    )
    .context("failed to parse SPLADE export manifest")?;
    ensure!(manifest.model == MODEL_KEY, "unexpected SPLADE model key");
    ensure!(
        manifest.revision == MODEL_REVISION,
        "unexpected SPLADE model revision"
    );
    for name in names {
        let expected = manifest
            .files
            .get(*name)
            .with_context(|| format!("manifest has no record for {name}"))?;
        let path = directory.join(name);
        ensure!(
            fs::metadata(&path)
                .with_context(|| format!("missing SPLADE asset {}", path.display()))?
                .len()
                == expected.bytes,
            "SPLADE asset size differs for {name}"
        );
        let mut reader = BufReader::new(fs::File::open(&path)?);
        let mut digest = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
        ensure!(
            format!("{:x}", digest.finalize()) == expected.sha256,
            "SPLADE asset SHA-256 differs for {name}"
        );
    }
    Ok(())
}

fn load_tokenizer(path: &Path, pad_batches: bool) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path)
        .map_err(|error| anyhow!(error.to_string()))
        .with_context(|| format!("failed to load tokenizer {}", path.display()))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: MAX_SEQUENCE_LENGTH,
            ..TruncationParams::default()
        }))
        .map_err(|error| anyhow!(error.to_string()))
        .context("failed to configure tokenizer truncation")?;
    if pad_batches {
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));
    }
    Ok(tokenizer)
}

fn input_kinds(model: &InferenceModel) -> Result<Vec<InputKind>> {
    let mut kinds = Vec::with_capacity(model.inputs.len());
    for outlet in &model.inputs {
        let name = model.node(outlet.node).name.as_str();
        kinds.push(match name {
            "input_ids" => InputKind::Ids,
            "attention_mask" => InputKind::AttentionMask,
            "token_type_ids" => InputKind::TypeIds,
            other => bail!("unsupported SPLADE input {other:?}"),
        });
    }
    ensure!(kinds.len() == 3, "expected three SPLADE inputs");
    Ok(kinds)
}
