//! HuggingFace safetensors weight loader for Gemma 4.
//!
//! Loads pretrained weights from HuggingFace format safetensors files
//! into the burn `Gemma4Model`. Handles:
//! - Name remapping (HF → burn module paths)
//! - Linear weight transposition (PyTorch [out,in] → Burn [in,out])
//! - RMSNorm parameter renaming (weight → gamma)
//! - Multi-file sharded models
//! - Tied weights (lm_head ← embed_tokens when `tie_word_embeddings`)
//! - Filtering of non-language-model tensors (vision/audio towers)
//!
//! # Gemma 4 Name Mapping Differences from Gemma 2
//!
//! | Aspect | Gemma 2 | Gemma 4 |
//! |--------|---------|---------|
//! | Prefix | `model.` | `model.language_model.` |
//! | Q/K norms | N/A | `q_norm.weight` → `q_norm.gamma` |
//! | Layer scalar | N/A | Skipped (f64 constant, not loadable) |
//! | PLE tensors | N/A | `per_layer_input_gate`, `per_layer_projection`, etc. |
//! | Vision/audio | N/A | Present in safetensors but skipped |
//!
//! # Tensors Skipped During Loading
//!
//! The following tensors are present in the safetensors file but not loaded:
//! - `model.language_model.layers.{i}.layer_scalar` — f64 constant in burn model
//! - `model.vision_tower.*` — vision encoder (not used for text-only LoRA)
//! - `model.audio_tower.*` — audio encoder (not used for text-only LoRA)
//! - `model.embed_vision.*` — vision embedding projection
//! - `model.embed_audio.*` — audio embedding projection

use std::path::{Path, PathBuf};

use burn::module::Param;
use burn::tensor::DType;
use burn::tensor::backend::Backend;
use burn_store::{
    KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    SafetensorsStoreError, TensorSnapshot,
};
use std::rc::Rc;

use crate::model::Gemma4Model;

// ---------------------------------------------------------------------------
// BF16 → F32 Adapter
// ---------------------------------------------------------------------------

/// Adapter that converts BF16 tensors to F32 during loading.
///
/// HuggingFace Gemma 4 weights are stored in BF16, but backends like NdArray
/// only support F32. Chain this adapter after `PyTorchToBurnAdapter` to
/// transparently upcast weights during loading.
///
/// # Example
///
/// ```ignore
/// let adapter = PyTorchToBurnAdapter.chain(Bf16ToF32Adapter);
/// store.with_from_adapter(adapter);
/// ```
#[derive(Debug, Clone)]
pub struct Bf16ToF32Adapter;

impl ModuleAdapter for Bf16ToF32Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype != DType::BF16 {
            return snapshot.clone();
        }

        let original_data_fn = snapshot.clone_data_fn();

        let cast_data_fn = Rc::new(move || {
            let data = original_data_fn()?;
            Ok(data.convert_dtype(DType::F32))
        });

        TensorSnapshot::from_closure(
            cast_data_fn,
            DType::F32,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

// ---------------------------------------------------------------------------
// Error & Report Types
// ---------------------------------------------------------------------------

/// Result of a weight loading operation.
#[derive(Debug, Clone)]
pub struct LoadReport {
    /// Number of tensors successfully loaded.
    pub tensors_loaded: usize,
    /// Tensor names that were skipped (not found or filtered).
    pub tensors_skipped: Vec<String>,
    /// Files that were read.
    pub files_read: Vec<String>,
}

/// Errors that can occur during weight loading.
#[derive(Debug)]
pub enum LoadError {
    /// The specified file or directory was not found.
    FileNotFound(PathBuf),
    /// The file format is invalid.
    InvalidFormat(String),
    /// An I/O error occurred.
    Io(std::io::Error),
    /// An error from the safetensors store.
    Store(SafetensorsStoreError),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::FileNotFound(path) => {
                write!(f, "File not found: {}", path.display())
            }
            LoadError::InvalidFormat(msg) => {
                write!(f, "Invalid format: {msg}")
            }
            LoadError::Io(err) => {
                write!(f, "I/O error: {err}")
            }
            LoadError::Store(err) => {
                write!(f, "Store error: {err}")
            }
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Io(err) => Some(err),
            LoadError::Store(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(err: std::io::Error) -> Self {
        LoadError::Io(err)
    }
}

impl From<SafetensorsStoreError> for LoadError {
    fn from(err: SafetensorsStoreError) -> Self {
        LoadError::Store(err)
    }
}

// ---------------------------------------------------------------------------
// Name Mapping
// ---------------------------------------------------------------------------

/// Build a `KeyRemapper` for HF Gemma 4 → burn name transformations.
///
/// Applies these remappings in order:
/// 1. Strip `model.language_model.` prefix (HF Gemma4ForConditionalGeneration wraps the text model)
/// 2. Rename `embed_tokens.` → `embed.` (HF name → burn module name)
/// 3. Rename norm `.weight` → `.gamma` (for all RMSNorm layers including q_norm, k_norm)
fn build_hf_remapper() -> KeyRemapper {
    KeyRemapper::new()
        // 1. Strip "model.language_model." prefix from HF names
        //    (Gemma 4 uses Gemma4ForConditionalGeneration which wraps the text model)
        .add_pattern(r"^model\.language_model\.", "")
        .expect("valid regex: strip model.language_model prefix")
        // 2. Rename embed_tokens → embed
        //    Note: embed_tokens_per_layer doesn't match because next char is '_', not '.'
        .add_pattern(r"^embed_tokens\.", "embed.")
        .expect("valid regex: embed_tokens to embed")
        // 3. Rename norm .weight → .gamma
        //    Matches all RMSNorm layers in Gemma 4:
        //    - Standard block norms: input_layernorm, post_attention_layernorm,
        //      pre_feedforward_layernorm, post_feedforward_layernorm
        //    - Attention norms: q_norm, k_norm
        //    - PLE norms: post_per_layer_input_norm, per_layer_projection_norm
        //    - Final norm: norm
        .add_pattern(
            r"(input_layernorm|post_attention_layernorm|pre_feedforward_layernorm|post_feedforward_layernorm|q_norm|k_norm|post_per_layer_input_norm|per_layer_projection_norm|norm)\.weight$",
            "$1.gamma",
        )
        .expect("valid regex: norm weight to gamma")
}

/// Map an HF weight name to a burn module path.
///
/// This is the same logic as `build_hf_remapper()` but as a pure function
/// for testing and inspection.
pub fn hf_to_burn_name(hf_name: &str) -> String {
    // RMSNorm layers that need .weight → .gamma
    let norm_suffixes = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "post_per_layer_input_norm.weight",
        "per_layer_projection_norm.weight",
    ];

    // Handle model.language_model.norm.weight
    if hf_name == "model.language_model.norm.weight" {
        return "norm.gamma".to_string();
    }

    // Handle model.language_model.embed_tokens.weight
    if hf_name == "model.language_model.embed_tokens.weight" {
        return "embed.weight".to_string();
    }

    // Skip non-language-model tensors (vision, audio, etc.)
    // They pass through unchanged and won't match any model parameter.
    if !hf_name.starts_with("model.language_model.") {
        return hf_name.to_string();
    }

    // Strip "model.language_model." prefix
    let name = match hf_name.strip_prefix("model.language_model.") {
        Some(n) => n,
        None => hf_name,
    };

    // Rename embed_tokens → embed (but NOT embed_tokens_per_layer)
    let name = if name.starts_with("embed_tokens.") {
        name.replacen("embed_tokens.", "embed.", 1)
    } else {
        name.to_string()
    };

    // Replace norm .weight with .gamma
    for suffix in &norm_suffixes {
        if name.ends_with(suffix) {
            return name.replace(suffix, &suffix.replace(".weight", ".gamma"));
        }
    }

    // Handle top-level norm
    if name == "norm.weight" {
        return "norm.gamma".to_string();
    }

    name
}

/// Check if a burn weight name corresponds to a linear layer that needs transposition.
///
/// PyTorch stores linear weights as [out_features, in_features].
/// Burn expects [in_features, out_features].
/// The `PyTorchToBurnAdapter` handles this during loading.
pub fn needs_transpose(burn_name: &str) -> bool {
    // Attention projections
    burn_name.ends_with("q_proj.weight")
        || burn_name.ends_with("k_proj.weight")
        || burn_name.ends_with("v_proj.weight")
        || burn_name.ends_with("o_proj.weight")
        // MLP projections
        || burn_name.ends_with("gate_proj.weight")
        || burn_name.ends_with("up_proj.weight")
        || burn_name.ends_with("down_proj.weight")
        // PLE linear projections (per-layer)
        || burn_name.ends_with("per_layer_input_gate.weight")
        || burn_name.ends_with("per_layer_projection.weight")
        // PLE model-level projection
        || burn_name.ends_with("per_layer_model_projection.weight")
        // LM head
        || burn_name == "lm_head.weight"
    // Note: embed.weight and embed_tokens_per_layer.weight are NOT transposed (embeddings)
    // Note: all .gamma (RMSNorm) weights are NOT transposed
}

/// Check if an HF tensor name belongs to the language model component.
fn is_language_model_tensor(hf_name: &str) -> bool {
    hf_name.starts_with("model.language_model.")
}

/// Check if an HF tensor name is a layer_scalar (should be skipped).
fn is_layer_scalar(hf_name: &str) -> bool {
    hf_name.ends_with(".layer_scalar")
}

// ---------------------------------------------------------------------------
// File Discovery
// ---------------------------------------------------------------------------

/// Resolve the loading path to a list of safetensors files.
///
/// Supports:
/// - Single `.safetensors` file
/// - Directory containing `model-*.safetensors` files (sharded)
/// - Path to `model.safetensors.index.json` (index file)
fn resolve_safetensors_files(path: &Path) -> Result<Vec<PathBuf>, LoadError> {
    if !path.exists() {
        return Err(LoadError::FileNotFound(path.to_path_buf()));
    }

    // Single file
    if path.is_file() {
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => return Ok(vec![path.to_path_buf()]),
            Some("json") => return parse_index_file(path),
            _ => {
                return Err(LoadError::InvalidFormat(format!(
                    "Expected .safetensors or .json file, got: {}",
                    path.display()
                )));
            }
        }
    }

    // Directory: find model-*.safetensors files
    if path.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("safetensors")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("model-"))
            })
            .collect();

        if files.is_empty() {
            // Try single model.safetensors in directory
            let single = path.join("model.safetensors");
            if single.exists() {
                return Ok(vec![single]);
            }

            return Err(LoadError::InvalidFormat(format!(
                "No model-*.safetensors files found in: {}",
                path.display()
            )));
        }

        // Sort by filename to ensure correct order (model-00001, model-00002, etc.)
        files.sort();
        Ok(files)
    } else {
        Err(LoadError::InvalidFormat(format!(
            "Not a file or directory: {}",
            path.display()
        )))
    }
}

/// Parse a safetensors index JSON file to get the list of shard files.
fn parse_index_file(path: &Path) -> Result<Vec<PathBuf>, LoadError> {
    let content = std::fs::read_to_string(path)?;
    let index: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| LoadError::InvalidFormat(format!("Invalid JSON index: {e}")))?;

    let weight_map = index
        .get("weight_map")
        .ok_or_else(|| LoadError::InvalidFormat("Missing 'weight_map' in index".to_string()))?;

    let file_set: std::collections::BTreeSet<String> = weight_map
        .as_object()
        .ok_or_else(|| LoadError::InvalidFormat("'weight_map' is not an object".to_string()))?
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();

    let base_dir = path.parent().unwrap_or(path);
    let files: Vec<PathBuf> = file_set.into_iter().map(|f| base_dir.join(f)).collect();

    if files.is_empty() {
        return Err(LoadError::InvalidFormat(
            "No files listed in weight_map".to_string(),
        ));
    }

    Ok(files)
}

// ---------------------------------------------------------------------------
// Weight Loading
// ---------------------------------------------------------------------------

/// Load Gemma 4 weights from HuggingFace safetensors files.
///
/// `path` can be:
/// - A single `.safetensors` file
/// - A directory containing `model-*.safetensors` files (sharded)
/// - A path to `model.safetensors.index.json` (index file)
///
/// # Name Mapping
///
/// HF safetensors weight names are remapped to burn module paths:
/// - `model.language_model.embed_tokens.weight` → `embed.weight`
/// - `model.language_model.embed_tokens_per_layer.weight` → `embed_tokens_per_layer.weight`
/// - `model.language_model.layers.{i}.self_attn.q_proj.weight` → `layers.{i}.self_attn.q_proj.weight`
/// - `model.language_model.layers.{i}.self_attn.q_norm.weight` → `layers.{i}.self_attn.q_norm.gamma`
/// - `model.language_model.layers.{i}.input_layernorm.weight` → `layers.{i}.input_layernorm.gamma`
/// - `model.language_model.norm.weight` → `norm.gamma`
/// - `model.language_model.per_layer_model_projection.weight` → `per_layer_model_projection.weight`
///
/// # Transformations
///
/// - Linear weights are transposed (PyTorch [out,in] → Burn [in,out])
/// - RMSNorm `weight` is renamed to `gamma`
/// - The `model.language_model.` prefix is stripped from all weight names
/// - BF16 tensors are upcast to F32
/// - Non-language-model tensors (vision/audio) are skipped
/// - `layer_scalar` tensors are skipped (stored as f64 constant in model)
///
/// # Tied Weights
///
/// In HF Gemma 4, `lm_head.weight` is tied to `embed_tokens.weight` when
/// `tie_word_embeddings: true`. The safetensors file may not contain a
/// separate `lm_head.weight`. The burn model handles this in its forward pass.
///
/// # Example
///
/// ```rust,ignore
/// use lora_gemma4::loader::load_gemma4_weights;
/// use lora_gemma4::{Gemma4Config, Gemma4Model};
/// use burn_ndarray::NdArray;
///
/// type B = NdArray;
/// let device = Default::default();
///
/// let config = Gemma4Config::gemma4_e4b();
/// let mut model = Gemma4Model::<B>::new(&config, &device);
///
/// let report = load_gemma4_weights(&mut model, "model.safetensors".as_ref(), &device)?;
/// println!("Loaded {} tensors from {} files", report.tensors_loaded, report.files_read.len());
/// ```
pub fn load_gemma4_weights<B: Backend>(
    model: &mut Gemma4Model<B>,
    path: &Path,
    device: &B::Device,
) -> Result<LoadReport, LoadError> {
    let files = resolve_safetensors_files(path)?;

    if files.is_empty() {
        return Err(LoadError::FileNotFound(path.to_path_buf()));
    }

    let remapper = build_hf_remapper();
    let mut total_loaded = 0;
    let mut all_skipped = Vec::new();
    let mut files_read = Vec::new();
    let mut had_embed = false;
    let mut had_lm_head = false;
    let mut layer_scalar_count = 0;
    let mut non_lm_count = 0;

    for file_path in &files {
        let file_name = file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        log::info!("Loading weights from: {file_name}");

        let adapter = PyTorchToBurnAdapter.chain(Bf16ToF32Adapter);
        let mut store = SafetensorsStore::from_file(file_path)
            .remap(remapper.clone())
            .with_from_adapter(adapter)
            .allow_partial(true);

        // Count non-LM and layer_scalar tensors before loading
        // (we inspect the original tensor names from skipped/errors)
        let result = model.load_from(&mut store).map_err(LoadError::Store)?;

        total_loaded += result.applied.len();
        all_skipped.extend(result.skipped.clone());

        // Track embed and lm_head weight presence
        had_embed = had_embed || result.applied.iter().any(|p| p == "embed.weight");
        had_lm_head = had_lm_head || result.applied.iter().any(|p| p == "lm_head.weight");

        // Categorize skipped tensors
        for skipped in &result.skipped {
            if is_layer_scalar(skipped) {
                layer_scalar_count += 1;
            } else if !is_language_model_tensor(skipped) {
                non_lm_count += 1;
            }
        }

        // Log missing for debugging (expected for sharded models)
        if !result.missing.is_empty() {
            log::debug!(
                "File {file_name}: {} tensors missing (will be loaded from other shards)",
                result.missing.len()
            );
        }

        // Log errors
        for err in &result.errors {
            log::warn!("Load error: {err}");
        }

        files_read.push(file_name);
    }

    // Log categorized skip statistics
    if layer_scalar_count > 0 {
        log::info!(
            "Skipped {layer_scalar_count} layer_scalar tensors (f64 constants in burn model)"
        );
    }
    if non_lm_count > 0 {
        log::info!("Skipped {non_lm_count} non-language-model tensors (vision/audio towers)");
    }

    // Handle tied weights: copy embed.weight → lm_head.weight (transposed)
    // Gemma 4 uses tie_word_embeddings=true but safetensors omits lm_head.weight.
    // Embedding weight is [vocab, hidden]; Linear weight is [hidden, vocab] in burn format.
    if model.tie_word_embeddings && had_embed && !had_lm_head {
        // Deref Param<Tensor> to &Tensor via Deref trait, clone the Tensor, then transpose.
        // Embed weight is [vocab, hidden]; Linear weight is [hidden, vocab] in burn format.
        let lm_weight = (*model.embed.weight).clone().transpose();
        model.lm_head.weight = Param::from_tensor(lm_weight);
        log::info!(
            "Tied embeddings: copied embed.weight to lm_head.weight (transposed [{vocab}, {hidden}] → [{hidden}, {vocab}])",
            vocab = model.vocab_size,
            hidden = model.hidden_size,
        );
    } else if model.tie_word_embeddings && had_lm_head {
        log::info!("tie_word_embeddings=true: lm_head.weight loaded from safetensors");
    } else if had_embed && !had_lm_head {
        log::warn!(
            "lm_head.weight not found in safetensors and tie_word_embeddings=false. \
             lm_head will use random initialization."
        );
    }

    let _ = device; // device used implicitly by load_from

    let report = LoadReport {
        tensors_loaded: total_loaded,
        tensors_skipped: all_skipped,
        files_read,
    };

    log::info!(
        "Weight loading complete: {} tensors from {} files",
        report.tensors_loaded,
        report.files_read.len()
    );

    Ok(report)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Name Mapping Tests — Top-Level Tensors
    // -----------------------------------------------------------------------

    #[test]
    fn test_hf_to_burn_embedding() {
        assert_eq!(
            hf_to_burn_name("model.language_model.embed_tokens.weight"),
            "embed.weight"
        );
    }

    #[test]
    fn test_hf_to_burn_embed_tokens_per_layer() {
        // PLE embedding — stays as-is (Option<Embedding> in model)
        assert_eq!(
            hf_to_burn_name("model.language_model.embed_tokens_per_layer.weight"),
            "embed_tokens_per_layer.weight"
        );
    }

    #[test]
    fn test_hf_to_burn_final_norm() {
        assert_eq!(
            hf_to_burn_name("model.language_model.norm.weight"),
            "norm.gamma"
        );
    }

    #[test]
    fn test_hf_to_burn_per_layer_model_projection() {
        // Model-level PLE projection (Linear)
        assert_eq!(
            hf_to_burn_name("model.language_model.per_layer_model_projection.weight"),
            "per_layer_model_projection.weight"
        );
    }

    #[test]
    fn test_hf_to_burn_per_layer_projection_norm() {
        // Model-level PLE norm (RmsNorm)
        assert_eq!(
            hf_to_burn_name("model.language_model.per_layer_projection_norm.weight"),
            "per_layer_projection_norm.gamma"
        );
    }

    // -----------------------------------------------------------------------
    // Name Mapping Tests — Attention
    // -----------------------------------------------------------------------

    #[test]
    fn test_hf_to_burn_attention_weights() {
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.self_attn.q_proj.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.5.self_attn.k_proj.weight"),
            "layers.5.self_attn.k_proj.weight"
        );
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.25.self_attn.v_proj.weight"),
            "layers.25.self_attn.v_proj.weight"
        );
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.12.self_attn.o_proj.weight"),
            "layers.12.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn test_hf_to_burn_q_norm() {
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.self_attn.q_norm.weight"),
            "layers.0.self_attn.q_norm.gamma"
        );
    }

    #[test]
    fn test_hf_to_burn_k_norm() {
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.self_attn.k_norm.weight"),
            "layers.0.self_attn.k_norm.gamma"
        );
    }

    // -----------------------------------------------------------------------
    // Name Mapping Tests — MLP
    // -----------------------------------------------------------------------

    #[test]
    fn test_hf_to_burn_mlp_weights() {
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.mlp.gate_proj.weight"),
            "layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.mlp.up_proj.weight"),
            "layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.mlp.down_proj.weight"),
            "layers.0.mlp.down_proj.weight"
        );
    }

    // -----------------------------------------------------------------------
    // Name Mapping Tests — Block Norms
    // -----------------------------------------------------------------------

    #[test]
    fn test_hf_to_burn_input_layernorm() {
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.input_layernorm.weight"),
            "layers.0.input_layernorm.gamma"
        );
    }

    #[test]
    fn test_hf_to_burn_post_attention_layernorm() {
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.post_attention_layernorm.weight"),
            "layers.0.post_attention_layernorm.gamma"
        );
    }

    #[test]
    fn test_hf_to_burn_pre_feedforward_layernorm() {
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.pre_feedforward_layernorm.weight"),
            "layers.0.pre_feedforward_layernorm.gamma"
        );
    }

    #[test]
    fn test_hf_to_burn_post_feedforward_layernorm() {
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.post_feedforward_layernorm.weight"),
            "layers.0.post_feedforward_layernorm.gamma"
        );
    }

    // -----------------------------------------------------------------------
    // Name Mapping Tests — PLE (Per-Layer Embeddings)
    // -----------------------------------------------------------------------

    #[test]
    fn test_hf_to_burn_ple_layer_weights() {
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.per_layer_input_gate.weight"),
            "layers.0.per_layer_input_gate.weight"
        );
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.per_layer_projection.weight"),
            "layers.0.per_layer_projection.weight"
        );
    }

    #[test]
    fn test_hf_to_burn_ple_layer_norm() {
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.post_per_layer_input_norm.weight"),
            "layers.0.post_per_layer_input_norm.gamma"
        );
    }

    // -----------------------------------------------------------------------
    // Name Mapping Tests — Skipped Tensors
    // -----------------------------------------------------------------------

    #[test]
    fn test_hf_to_burn_layer_scalar_not_transformed() {
        // layer_scalar is a f64 constant — not a loadable tensor.
        // The remapper strips the prefix but doesn't change the name.
        let result = hf_to_burn_name("model.language_model.layers.0.layer_scalar");
        // After stripping prefix: "layers.0.layer_scalar" — no .weight suffix, no norm match
        assert_eq!(result, "layers.0.layer_scalar");
    }

    #[test]
    fn test_hf_to_burn_vision_tensor_unchanged() {
        // Vision tower tensors don't start with model.language_model.
        // They pass through unchanged and won't match any model parameter.
        let name = "model.vision_tower.encoder.layers.0.input_layernorm.weight";
        assert_eq!(hf_to_burn_name(name), name);
    }

    #[test]
    fn test_hf_to_burn_audio_tensor_unchanged() {
        let name = "model.audio_tower.layers.0.self_attn.q_proj.linear.weight";
        assert_eq!(hf_to_burn_name(name), name);
    }

    #[test]
    fn test_hf_to_burn_embed_vision_projection_unchanged() {
        let name = "model.embed_vision.embedding_projection.weight";
        assert_eq!(hf_to_burn_name(name), name);
    }

    #[test]
    fn test_hf_to_burn_embed_audio_projection_unchanged() {
        let name = "model.embed_audio.embedding_projection.weight";
        assert_eq!(hf_to_burn_name(name), name);
    }

    // -----------------------------------------------------------------------
    // Name Mapping Tests — Comprehensive Per-Layer
    // -----------------------------------------------------------------------

    #[test]
    fn test_hf_to_burn_all_norms_in_layer() {
        let layer_patterns = [
            (
                "model.language_model.layers.0.input_layernorm.weight",
                "layers.0.input_layernorm.gamma",
            ),
            (
                "model.language_model.layers.0.post_attention_layernorm.weight",
                "layers.0.post_attention_layernorm.gamma",
            ),
            (
                "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                "layers.0.pre_feedforward_layernorm.gamma",
            ),
            (
                "model.language_model.layers.0.post_feedforward_layernorm.weight",
                "layers.0.post_feedforward_layernorm.gamma",
            ),
            (
                "model.language_model.layers.0.self_attn.q_norm.weight",
                "layers.0.self_attn.q_norm.gamma",
            ),
            (
                "model.language_model.layers.0.self_attn.k_norm.weight",
                "layers.0.self_attn.k_norm.gamma",
            ),
            (
                "model.language_model.layers.0.post_per_layer_input_norm.weight",
                "layers.0.post_per_layer_input_norm.gamma",
            ),
        ];

        for (hf, expected_burn) in layer_patterns {
            assert_eq!(hf_to_burn_name(hf), expected_burn, "Failed for: {hf}");
        }
    }

    #[test]
    fn test_hf_to_burn_multiple_layers() {
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.0.input_layernorm.weight"),
            "layers.0.input_layernorm.gamma"
        );
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.13.input_layernorm.weight"),
            "layers.13.input_layernorm.gamma"
        );
        assert_eq!(
            hf_to_burn_name("model.language_model.layers.41.input_layernorm.weight"),
            "layers.41.input_layernorm.gamma"
        );
    }

    // -----------------------------------------------------------------------
    // Name Mapping Tests — Full Layer 0 (all 17 tensors)
    // -----------------------------------------------------------------------

    #[test]
    fn test_hf_to_burn_full_layer() {
        let expected = [
            (
                "model.language_model.layers.0.input_layernorm.weight",
                "layers.0.input_layernorm.gamma",
            ),
            (
                "model.language_model.layers.0.layer_scalar",
                "layers.0.layer_scalar",
            ),
            (
                "model.language_model.layers.0.mlp.down_proj.weight",
                "layers.0.mlp.down_proj.weight",
            ),
            (
                "model.language_model.layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.gate_proj.weight",
            ),
            (
                "model.language_model.layers.0.mlp.up_proj.weight",
                "layers.0.mlp.up_proj.weight",
            ),
            (
                "model.language_model.layers.0.per_layer_input_gate.weight",
                "layers.0.per_layer_input_gate.weight",
            ),
            (
                "model.language_model.layers.0.per_layer_projection.weight",
                "layers.0.per_layer_projection.weight",
            ),
            (
                "model.language_model.layers.0.post_attention_layernorm.weight",
                "layers.0.post_attention_layernorm.gamma",
            ),
            (
                "model.language_model.layers.0.post_feedforward_layernorm.weight",
                "layers.0.post_feedforward_layernorm.gamma",
            ),
            (
                "model.language_model.layers.0.post_per_layer_input_norm.weight",
                "layers.0.post_per_layer_input_norm.gamma",
            ),
            (
                "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                "layers.0.pre_feedforward_layernorm.gamma",
            ),
            (
                "model.language_model.layers.0.self_attn.k_norm.weight",
                "layers.0.self_attn.k_norm.gamma",
            ),
            (
                "model.language_model.layers.0.self_attn.k_proj.weight",
                "layers.0.self_attn.k_proj.weight",
            ),
            (
                "model.language_model.layers.0.self_attn.o_proj.weight",
                "layers.0.self_attn.o_proj.weight",
            ),
            (
                "model.language_model.layers.0.self_attn.q_norm.weight",
                "layers.0.self_attn.q_norm.gamma",
            ),
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.q_proj.weight",
            ),
            (
                "model.language_model.layers.0.self_attn.v_proj.weight",
                "layers.0.self_attn.v_proj.weight",
            ),
        ];

        for (hf, burn) in &expected {
            assert_eq!(hf_to_burn_name(hf), *burn, "Failed for: {hf}");
        }
    }

    // -----------------------------------------------------------------------
    // Transpose Detection Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_needs_transpose_linear_projections() {
        // Attention projections
        assert!(needs_transpose("layers.0.self_attn.q_proj.weight"));
        assert!(needs_transpose("layers.0.self_attn.k_proj.weight"));
        assert!(needs_transpose("layers.0.self_attn.v_proj.weight"));
        assert!(needs_transpose("layers.0.self_attn.o_proj.weight"));
        // MLP projections
        assert!(needs_transpose("layers.0.mlp.gate_proj.weight"));
        assert!(needs_transpose("layers.0.mlp.up_proj.weight"));
        assert!(needs_transpose("layers.0.mlp.down_proj.weight"));
        // PLE per-layer projections
        assert!(needs_transpose("layers.0.per_layer_input_gate.weight"));
        assert!(needs_transpose("layers.0.per_layer_projection.weight"));
        // PLE model-level projection
        assert!(needs_transpose("per_layer_model_projection.weight"));
        // LM head
        assert!(needs_transpose("lm_head.weight"));
    }

    #[test]
    fn test_no_transpose_embedding_and_norms() {
        // Embeddings
        assert!(!needs_transpose("embed.weight"));
        assert!(!needs_transpose("embed_tokens_per_layer.weight"));
        // Block norms
        assert!(!needs_transpose("layers.0.input_layernorm.gamma"));
        assert!(!needs_transpose("layers.0.post_attention_layernorm.gamma"));
        assert!(!needs_transpose("layers.0.pre_feedforward_layernorm.gamma"));
        assert!(!needs_transpose(
            "layers.0.post_feedforward_layernorm.gamma"
        ));
        // Attention norms
        assert!(!needs_transpose("layers.0.self_attn.q_norm.gamma"));
        assert!(!needs_transpose("layers.0.self_attn.k_norm.gamma"));
        // PLE norms
        assert!(!needs_transpose("layers.0.post_per_layer_input_norm.gamma"));
        assert!(!needs_transpose("per_layer_projection_norm.gamma"));
        // Final norm
        assert!(!needs_transpose("norm.gamma"));
    }

    // -----------------------------------------------------------------------
    // KeyRemapper Consistency Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_remapper_matches_function() {
        let test_cases = [
            // Top-level
            ("model.language_model.embed_tokens.weight", "embed.weight"),
            (
                "model.language_model.embed_tokens_per_layer.weight",
                "embed_tokens_per_layer.weight",
            ),
            ("model.language_model.norm.weight", "norm.gamma"),
            (
                "model.language_model.per_layer_model_projection.weight",
                "per_layer_model_projection.weight",
            ),
            (
                "model.language_model.per_layer_projection_norm.weight",
                "per_layer_projection_norm.gamma",
            ),
            // Attention
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.q_proj.weight",
            ),
            (
                "model.language_model.layers.0.self_attn.q_norm.weight",
                "layers.0.self_attn.q_norm.gamma",
            ),
            (
                "model.language_model.layers.0.self_attn.k_norm.weight",
                "layers.0.self_attn.k_norm.gamma",
            ),
            // MLP
            (
                "model.language_model.layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.gate_proj.weight",
            ),
            // Block norms
            (
                "model.language_model.layers.0.input_layernorm.weight",
                "layers.0.input_layernorm.gamma",
            ),
            (
                "model.language_model.layers.0.post_attention_layernorm.weight",
                "layers.0.post_attention_layernorm.gamma",
            ),
            (
                "model.language_model.layers.0.pre_feedforward_layernorm.weight",
                "layers.0.pre_feedforward_layernorm.gamma",
            ),
            (
                "model.language_model.layers.0.post_feedforward_layernorm.weight",
                "layers.0.post_feedforward_layernorm.gamma",
            ),
            // PLE per-layer
            (
                "model.language_model.layers.0.per_layer_input_gate.weight",
                "layers.0.per_layer_input_gate.weight",
            ),
            (
                "model.language_model.layers.0.per_layer_projection.weight",
                "layers.0.per_layer_projection.weight",
            ),
            (
                "model.language_model.layers.0.post_per_layer_input_norm.weight",
                "layers.0.post_per_layer_input_norm.gamma",
            ),
        ];

        let remapper = build_hf_remapper();

        for (hf_name, expected) in &test_cases {
            let function_result = hf_to_burn_name(hf_name);
            assert_eq!(
                &function_result, expected,
                "hf_to_burn_name mismatch for: {hf_name}"
            );

            // Test remapper produces same result
            let path_parts: Vec<String> = hf_name.split('.').map(|s| s.to_string()).collect();
            let snapshot = burn_store::TensorSnapshot::from_closure(
                std::rc::Rc::new(|| Err(burn_store::TensorSnapshotError::IoError("test".into()))),
                burn::tensor::DType::F32,
                burn::tensor::Shape::from([1usize]),
                path_parts,
                vec![],
                burn::module::ParamId::new(),
            );

            let (remapped, _) = remapper.remap(vec![snapshot]);
            assert_eq!(
                remapped[0].full_path(),
                *expected,
                "KeyRemapper mismatch for: {hf_name}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Helper Function Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_language_model_tensor() {
        assert!(is_language_model_tensor(
            "model.language_model.embed_tokens.weight"
        ));
        assert!(is_language_model_tensor(
            "model.language_model.layers.0.self_attn.q_proj.weight"
        ));
        assert!(is_language_model_tensor("model.language_model.norm.weight"));
        assert!(!is_language_model_tensor(
            "model.vision_tower.encoder.layers.0.input_layernorm.weight"
        ));
        assert!(!is_language_model_tensor(
            "model.audio_tower.layers.0.self_attn.q_proj.linear.weight"
        ));
        assert!(!is_language_model_tensor(
            "model.embed_vision.embedding_projection.weight"
        ));
    }

    #[test]
    fn test_is_layer_scalar() {
        assert!(is_layer_scalar(
            "model.language_model.layers.0.layer_scalar"
        ));
        assert!(is_layer_scalar(
            "model.language_model.layers.41.layer_scalar"
        ));
        assert!(!is_layer_scalar(
            "model.language_model.layers.0.input_layernorm.weight"
        ));
        assert!(!is_layer_scalar("model.language_model.norm.weight"));
    }

    // -----------------------------------------------------------------------
    // File Resolution Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_nonexistent_path() {
        let result = resolve_safetensors_files(Path::new("/nonexistent/path/model.safetensors"));
        assert!(matches!(result, Err(LoadError::FileNotFound(_))));
    }

    #[test]
    fn test_resolve_invalid_extension() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.txt");
        std::fs::write(&file_path, "not a model").unwrap();

        let result = resolve_safetensors_files(&file_path);
        assert!(matches!(result, Err(LoadError::InvalidFormat(_))));
    }

    #[test]
    fn test_resolve_empty_directory() {
        let dir = tempfile::tempdir().unwrap();

        let result = resolve_safetensors_files(dir.path());
        assert!(matches!(result, Err(LoadError::InvalidFormat(_))));
    }

    #[test]
    fn test_resolve_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.safetensors");
        std::fs::write(&file_path, "dummy").unwrap();

        let result = resolve_safetensors_files(&file_path);
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], file_path);
    }

    #[test]
    fn test_resolve_directory_with_shards() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-00001-of-00003.safetensors"), "a").unwrap();
        std::fs::write(dir.path().join("model-00003-of-00003.safetensors"), "c").unwrap();
        std::fs::write(dir.path().join("model-00002-of-00003.safetensors"), "b").unwrap();
        // Non-matching file
        std::fs::write(dir.path().join("tokenizer.json"), "{}").unwrap();

        let result = resolve_safetensors_files(dir.path());
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.len(), 3);
        // Should be sorted
        assert!(files[0].to_string_lossy().contains("model-00001"));
        assert!(files[1].to_string_lossy().contains("model-00002"));
        assert!(files[2].to_string_lossy().contains("model-00003"));
    }

    #[test]
    fn test_resolve_directory_with_single_model() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), "dummy").unwrap();

        let result = resolve_safetensors_files(dir.path());
        assert!(result.is_ok());
        let files = result.unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_load_nonexistent_file() {
        type B = burn_ndarray::NdArray;

        let device = Default::default();
        // Use a tiny config to avoid slow 42-layer model initialization.
        // The load function exits early on FileNotFound before using the model.
        let config = crate::Gemma4Config {
            vocab_size: 64,
            hidden_size: 32,
            num_hidden_layers: 1,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 16,
            global_head_dim: 32,
            rms_norm_eps: 1e-6,
            final_logit_softcapping: 30.0,
            sliding_window: 16,
            num_kv_shared_layers: 0,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: 0,
            max_position_embeddings: 32,
            tie_word_embeddings: true,
        };
        let mut model: Gemma4Model<B> = Gemma4Model::new(&config, &device);

        let result = load_gemma4_weights(
            &mut model,
            Path::new("/nonexistent/model.safetensors"),
            &device,
        );
        assert!(matches!(result, Err(LoadError::FileNotFound(_))));
    }
}
