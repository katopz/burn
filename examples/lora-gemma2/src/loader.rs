//! HuggingFace safetensors weight loader for Gemma 2.
//!
//! Loads pretrained weights from HuggingFace format safetensors files
//! into the burn `Gemma2Model`. Handles:
//! - Name remapping (HF → burn module paths)
//! - Linear weight transposition (PyTorch [out,in] → Burn [in,out])
//! - RMSNorm parameter renaming (weight → gamma)
//! - Multi-file sharded models
//! - Tied weights (lm_head ← embed_tokens)

use std::path::{Path, PathBuf};

use burn::tensor::DType;
use burn::tensor::backend::Backend;
use burn_store::{
    KeyRemapper, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
    SafetensorsStoreError, TensorSnapshot,
};
use std::rc::Rc;

use crate::model::Gemma2Model;

// ---------------------------------------------------------------------------
// BF16 → F32 Adapter
// ---------------------------------------------------------------------------

/// Adapter that converts BF16 tensors to F32 during loading.
///
/// HuggingFace Gemma 2 weights are stored in BF16, but backends like NdArray
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
// BF16 → F16 Adapter (for Metal backend)
// ---------------------------------------------------------------------------

/// Adapter that converts BF16 tensors to F16 during loading.
///
/// HuggingFace Gemma 2 weights are stored in BF16, but Metal/WGPU
/// doesn't support BF16. This adapter converts BF16→F16, halving memory
/// compared to F32 (16GB vs 32GB for Gemma 4 E4B).
///
/// Use with `Metal<f16>` backend. Chain after `PyTorchToBurnAdapter`.
#[derive(Debug, Clone)]
pub struct Bf16ToF16Adapter;

impl ModuleAdapter for Bf16ToF16Adapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        if snapshot.dtype != DType::BF16 {
            return snapshot.clone();
        }

        let original_data_fn = snapshot.clone_data_fn();

        let cast_data_fn = Rc::new(move || {
            let data = original_data_fn()?;
            Ok(data.convert_dtype(DType::F16))
        });

        TensorSnapshot::from_closure(
            cast_data_fn,
            DType::F16,
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

/// Build a `KeyRemapper` for HF Gemma 2 → burn name transformations.
///
/// Applies these remappings in order:
/// 1. Strip `model.` prefix (HF CausalLM wraps base model)
/// 2. Rename `embed_tokens` → `embed` (HF name → burn module name)
/// 3. Rename norm `.weight` → `.gamma` (for RMSNorm layers)
fn build_hf_remapper() -> KeyRemapper {
    KeyRemapper::new()
        // 1. Strip "model." prefix from HF names
        .add_pattern(r"^model\.", "")
        .expect("valid regex: strip model prefix")
        // 2. Rename embed_tokens → embed
        .add_pattern(r"^embed_tokens", "embed")
        .expect("valid regex: embed_tokens to embed")
        // 3. Rename norm .weight → .gamma
        //    Matches: input_layernorm, post_attention_layernorm,
        //    pre_feedforward_layernorm, post_feedforward_layernorm, norm
        .add_pattern(
            r"(input_layernorm|post_attention_layernorm|pre_feedforward_layernorm|post_feedforward_layernorm|norm)\.weight$",
            "$1.gamma",
        )
        .expect("valid regex: norm weight to gamma")
}

/// Map an HF weight name to a burn module path.
///
/// This is the same logic as `build_hf_remapper()` but as a pure function
/// for testing and inspection.
pub fn hf_to_burn_name(hf_name: &str) -> String {
    // RMSNorm layers that need .weight -> .gamma
    let norm_suffixes = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
    ];

    // Handle model.norm.weight
    if hf_name == "model.norm.weight" {
        return "norm.gamma".to_string();
    }

    // Handle model.embed_tokens.weight
    if hf_name == "model.embed_tokens.weight" {
        return "embed.weight".to_string();
    }

    // Handle lm_head.weight (no model. prefix, already correct for Linear)
    if hf_name == "lm_head.weight" {
        return "lm_head.weight".to_string();
    }

    // Strip "model." prefix
    let name = match hf_name.strip_prefix("model.") {
        Some(n) => n,
        None => hf_name,
    };

    // Rename embed_tokens → embed
    let name = match name.strip_prefix("embed_tokens") {
        Some(rest) => format!("embed{rest}"),
        None => name.to_string(),
    };

    // Replace norm .weight with .gamma
    for suffix in &norm_suffixes {
        if name.ends_with(suffix) {
            return name.replace(suffix, &suffix.replace(".weight", ".gamma"));
        }
    }

    name
}

/// Check if a burn weight name corresponds to a linear layer that needs transposition.
///
/// PyTorch stores linear weights as [out_features, in_features].
/// Burn expects [in_features, out_features].
/// The `PyTorchToBurnAdapter` handles this during loading.
pub fn needs_transpose(burn_name: &str) -> bool {
    // All linear projections in Gemma 2 (no bias)
    burn_name.ends_with("q_proj.weight")
        || burn_name.ends_with("k_proj.weight")
        || burn_name.ends_with("v_proj.weight")
        || burn_name.ends_with("o_proj.weight")
        || burn_name.ends_with("gate_proj.weight")
        || burn_name.ends_with("up_proj.weight")
        || burn_name.ends_with("down_proj.weight")
        || burn_name == "lm_head.weight"
    // Note: embed.weight is NOT transposed (it's an embedding table)
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

/// Load Gemma 2 weights from HuggingFace safetensors files.
///
/// `path` can be:
/// - A single `.safetensors` file
/// - A directory containing `model-*.safetensors` files (sharded)
/// - A path to `model.safetensors.index.json` (index file)
///
/// # Name Mapping
///
/// HF safetensors weight names are remapped to burn module paths:
/// - `model.embed_tokens.weight` → `embed.weight`
/// - `model.layers.{i}.self_attn.q_proj.weight` → `layers.{i}.self_attn.q_proj.weight`
/// - `model.layers.{i}.input_layernorm.weight` → `layers.{i}.input_layernorm.gamma`
/// - `model.norm.weight` → `norm.gamma`
/// - `lm_head.weight` → `lm_head.weight` (no change needed)
///
/// # Transformations
///
/// - Linear weights are transposed (PyTorch [out,in] → Burn [in,out])
/// - RMSNorm `weight` is renamed to `gamma`
/// - The `model.` prefix is stripped from all weight names
///
/// # Tied Weights
///
/// In HF Gemma 2, `lm_head.weight` is tied to `model.embed_tokens.weight`.
/// Both should be present in the safetensors files. If only one exists,
/// a warning is logged.
///
/// # Example
///
/// ```rust,ignore
/// use lora_gemma2::loader::load_gemma2_weights;
/// use lora_gemma2::{Gemma2Config, Gemma2Model};
/// use burn_ndarray::NdArray;
///
/// type B = NdArray;
/// let device = Default::default();
///
/// let config = Gemma2Config::gemma2_2b();
/// let mut model = Gemma2Model::<B>::new(&config, &device);
///
/// let report = load_gemma2_weights(&mut model, "model.safetensors".as_ref(), &device)?;
/// println!("Loaded {} tensors from {} files", report.tensors_loaded, report.files_read.len());
/// ```
pub fn load_gemma2_weights<B: Backend>(
    model: &mut Gemma2Model<B>,
    path: &Path,
    device: &B::Device,
) -> Result<LoadReport, LoadError> {
    load_gemma2_weights_dtype(model, path, device, DType::F32)
}

/// Fix RMSNorm gamma values for Gemma 2 weight compatibility.
///
/// Gemma 2's HuggingFace implementation stores RMSNorm weights as offsets
/// from 1.0 (i.e., `weight = gamma - 1.0`) and applies them as `(1.0 + weight)`.
/// Burn's `RmsNorm` expects absolute gamma values and applies `x * gamma`.
///
/// This function adds 1.0 to all loaded gamma values to convert from
/// HuggingFace's offset format to burn's absolute format.
fn fix_rmsnorm_gemma2_offset<B: Backend>(model: &mut Gemma2Model<B>) {
    let fix = |norm: &mut burn::nn::RmsNorm<B>| {
        let gamma = norm.gamma.val().clone().add_scalar(1.0);
        norm.gamma = burn::module::Param::from_tensor(gamma);
    };

    for layer in &mut model.layers {
        fix(&mut layer.input_layernorm);
        fix(&mut layer.post_attention_layernorm);
        fix(&mut layer.pre_feedforward_layernorm);
        fix(&mut layer.post_feedforward_layernorm);
    }

    fix(&mut model.norm);
}

/// Load Gemma 2 weights with explicit target dtype.
///
/// Use `DType::F32` for NdArray (CPU) and `DType::F16` for Metal/WGPU.
/// BF16 safetensors are converted to the target dtype during loading.
pub fn load_gemma2_weights_dtype<B: Backend>(
    model: &mut Gemma2Model<B>,
    path: &Path,
    device: &B::Device,
    target_dtype: DType,
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

    for file_path in &files {
        let file_name = file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        log::info!("Loading weights from: {file_name}");

        let adapter = match target_dtype {
            DType::F16 => PyTorchToBurnAdapter.chain(Bf16ToF16Adapter),
            _ => PyTorchToBurnAdapter.chain(Bf16ToF32Adapter),
        };
        let mut store = SafetensorsStore::from_file(file_path)
            .remap(remapper.clone())
            .with_from_adapter(adapter)
            .allow_partial(true);

        let result = model.load_from(&mut store).map_err(LoadError::Store)?;

        total_loaded += result.applied.len();
        all_skipped.extend(result.skipped);

        // Track tied weight presence
        had_embed = had_embed || result.applied.iter().any(|p| p == "embed.weight");
        had_lm_head = had_lm_head || result.applied.iter().any(|p| p == "lm_head.weight");

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

    // Handle tied weights: copy embed.weight → lm_head.weight (transposed)
    // Gemma 2 uses tie_word_embeddings=true but safetensors omits lm_head.weight.
    // Embedding weight is [vocab, hidden]; Linear weight is [hidden, vocab] in burn format.
    if had_embed && !had_lm_head {
        let lm_weight = (*model.embed.weight).clone().transpose();
        model.lm_head.weight = burn::module::Param::from_tensor(lm_weight);
        log::info!(
            "Tied embeddings: copied embed.weight to lm_head.weight (transposed [{vocab}, {hidden}] → [{hidden}, {vocab}])",
            vocab = model.vocab_size,
            hidden = model.hidden_size,
        );
    } else if !had_embed && had_lm_head {
        log::warn!(
            "embed.weight was not found in safetensors files. \
             The embedding layer will use random initialization."
        );
    }

    // Fix RMSNorm gamma values: Gemma 2 stores (gamma - 1.0) in safetensors,
    // but burn's RmsNorm expects absolute gamma values.
    let gamma_count = model.layers.len() * 4 + 1; // 4 norms per layer + final norm
    fix_rmsnorm_gemma2_offset(model);
    log::info!(
        "Fixed {gamma_count} RMSNorm gamma values: added 1.0 to convert from Gemma 2 offset format"
    );

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
    // Name Mapping Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hf_to_burn_embedding() {
        assert_eq!(hf_to_burn_name("model.embed_tokens.weight"), "embed.weight");
    }

    #[test]
    fn test_hf_to_burn_final_norm() {
        assert_eq!(hf_to_burn_name("model.norm.weight"), "norm.gamma");
    }

    #[test]
    fn test_hf_to_burn_lm_head() {
        assert_eq!(hf_to_burn_name("lm_head.weight"), "lm_head.weight");
    }

    #[test]
    fn test_hf_to_burn_attention_weights() {
        assert_eq!(
            hf_to_burn_name("model.layers.0.self_attn.q_proj.weight"),
            "layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            hf_to_burn_name("model.layers.5.self_attn.k_proj.weight"),
            "layers.5.self_attn.k_proj.weight"
        );
        assert_eq!(
            hf_to_burn_name("model.layers.25.self_attn.v_proj.weight"),
            "layers.25.self_attn.v_proj.weight"
        );
        assert_eq!(
            hf_to_burn_name("model.layers.12.self_attn.o_proj.weight"),
            "layers.12.self_attn.o_proj.weight"
        );
    }

    #[test]
    fn test_hf_to_burn_mlp_weights() {
        assert_eq!(
            hf_to_burn_name("model.layers.0.mlp.gate_proj.weight"),
            "layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            hf_to_burn_name("model.layers.0.mlp.up_proj.weight"),
            "layers.0.mlp.up_proj.weight"
        );
        assert_eq!(
            hf_to_burn_name("model.layers.0.mlp.down_proj.weight"),
            "layers.0.mlp.down_proj.weight"
        );
    }

    #[test]
    fn test_hf_to_burn_input_layernorm() {
        assert_eq!(
            hf_to_burn_name("model.layers.0.input_layernorm.weight"),
            "layers.0.input_layernorm.gamma"
        );
    }

    #[test]
    fn test_hf_to_burn_post_attention_layernorm() {
        assert_eq!(
            hf_to_burn_name("model.layers.0.post_attention_layernorm.weight"),
            "layers.0.post_attention_layernorm.gamma"
        );
    }

    #[test]
    fn test_hf_to_burn_pre_feedforward_layernorm() {
        assert_eq!(
            hf_to_burn_name("model.layers.0.pre_feedforward_layernorm.weight"),
            "layers.0.pre_feedforward_layernorm.gamma"
        );
    }

    #[test]
    fn test_hf_to_burn_post_feedforward_layernorm() {
        assert_eq!(
            hf_to_burn_name("model.layers.0.post_feedforward_layernorm.weight"),
            "layers.0.post_feedforward_layernorm.gamma"
        );
    }

    #[test]
    fn test_hf_to_burn_all_norms_in_layer() {
        let layer_patterns = [
            (
                "model.layers.0.input_layernorm.weight",
                "layers.0.input_layernorm.gamma",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "layers.0.post_attention_layernorm.gamma",
            ),
            (
                "model.layers.0.pre_feedforward_layernorm.weight",
                "layers.0.pre_feedforward_layernorm.gamma",
            ),
            (
                "model.layers.0.post_feedforward_layernorm.weight",
                "layers.0.post_feedforward_layernorm.gamma",
            ),
        ];

        for (hf, expected_burn) in layer_patterns {
            assert_eq!(hf_to_burn_name(hf), expected_burn, "Failed for: {hf}");
        }
    }

    #[test]
    fn test_hf_to_burn_multiple_layers() {
        assert_eq!(
            hf_to_burn_name("model.layers.0.input_layernorm.weight"),
            "layers.0.input_layernorm.gamma"
        );
        assert_eq!(
            hf_to_burn_name("model.layers.13.input_layernorm.weight"),
            "layers.13.input_layernorm.gamma"
        );
        assert_eq!(
            hf_to_burn_name("model.layers.25.input_layernorm.weight"),
            "layers.25.input_layernorm.gamma"
        );
    }

    // -----------------------------------------------------------------------
    // Transpose Detection Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_needs_transpose_linear_projections() {
        assert!(needs_transpose("layers.0.self_attn.q_proj.weight"));
        assert!(needs_transpose("layers.0.self_attn.k_proj.weight"));
        assert!(needs_transpose("layers.0.self_attn.v_proj.weight"));
        assert!(needs_transpose("layers.0.self_attn.o_proj.weight"));
        assert!(needs_transpose("layers.0.mlp.gate_proj.weight"));
        assert!(needs_transpose("layers.0.mlp.up_proj.weight"));
        assert!(needs_transpose("layers.0.mlp.down_proj.weight"));
        assert!(needs_transpose("lm_head.weight"));
    }

    #[test]
    fn test_no_transpose_embedding_and_norms() {
        assert!(!needs_transpose("embed.weight"));
        assert!(!needs_transpose("layers.0.input_layernorm.gamma"));
        assert!(!needs_transpose("layers.0.post_attention_layernorm.gamma"));
        assert!(!needs_transpose("layers.0.pre_feedforward_layernorm.gamma"));
        assert!(!needs_transpose(
            "layers.0.post_feedforward_layernorm.gamma"
        ));
        assert!(!needs_transpose("norm.gamma"));
    }

    // -----------------------------------------------------------------------
    // KeyRemapper Consistency Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_remapper_matches_function() {
        let test_cases = [
            ("model.embed_tokens.weight", "embed.weight"),
            ("model.norm.weight", "norm.gamma"),
            ("lm_head.weight", "lm_head.weight"),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "layers.0.self_attn.q_proj.weight",
            ),
            (
                "model.layers.0.input_layernorm.weight",
                "layers.0.input_layernorm.gamma",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "layers.0.post_attention_layernorm.gamma",
            ),
            (
                "model.layers.0.pre_feedforward_layernorm.weight",
                "layers.0.pre_feedforward_layernorm.gamma",
            ),
            (
                "model.layers.0.post_feedforward_layernorm.weight",
                "layers.0.post_feedforward_layernorm.gamma",
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "layers.0.mlp.gate_proj.weight",
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
    // File Resolution Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_nonexistent_path() {
        let result = resolve_safetensors_files(Path::new("/nonexistent/path"));
        assert!(matches!(result, Err(LoadError::FileNotFound(_))));
    }

    #[test]
    fn test_resolve_invalid_extension() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("model.txt");
        std::fs::write(&file_path, "test").unwrap();

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
        std::fs::write(&file_path, "").unwrap();

        let result = resolve_safetensors_files(&file_path).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], file_path);
    }

    #[test]
    fn test_resolve_directory_with_shards() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-00001-of-00003.safetensors"), "").unwrap();
        std::fs::write(dir.path().join("model-00002-of-00003.safetensors"), "").unwrap();
        std::fs::write(dir.path().join("model-00003-of-00003.safetensors"), "").unwrap();

        let result = resolve_safetensors_files(dir.path()).unwrap();
        assert_eq!(result.len(), 3);
        // Should be sorted
        assert!(result[0].to_string_lossy().contains("model-00001"));
        assert!(result[1].to_string_lossy().contains("model-00002"));
        assert!(result[2].to_string_lossy().contains("model-00003"));
    }

    #[test]
    fn test_resolve_directory_with_single_model() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.safetensors"), "").unwrap();

        let result = resolve_safetensors_files(dir.path()).unwrap();
        assert_eq!(result.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Full Weight Loading Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_nonexistent_file() {
        type B = burn_ndarray::NdArray;
        let device = Default::default();
        let config = crate::types::Gemma2Config::gemma2_2b();
        let mut model = Gemma2Model::<B>::new(&config, &device);

        let result = load_gemma2_weights(
            &mut model,
            Path::new("/nonexistent/model.safetensors"),
            &device,
        );
        assert!(matches!(result, Err(LoadError::FileNotFound(_))));
    }

    // -----------------------------------------------------------------------
    // RMSNorm Offset Fix Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fix_rmsnorm_gemma2_offset_adds_one() {
        type B = burn_ndarray::NdArray;
        let device = Default::default();
        let config = crate::types::Gemma2Config::gemma2_2b();
        let mut model = Gemma2Model::<B>::new(&config, &device);

        // Simulate HF-loaded gamma values near 0.0 (offsets from 1.0)
        // by setting all norm gammas to a known small value.
        let hf_gamma_val = 0.15; // typical HF offset value
        let set_gamma = |norm: &mut burn::nn::RmsNorm<B>| {
            let [d] = norm.gamma.shape().dims();
            let tensor = burn::tensor::Tensor::<B, 1>::full([d], hf_gamma_val, &device);
            norm.gamma = burn::module::Param::from_tensor(tensor);
        };

        for layer in &mut model.layers {
            set_gamma(&mut layer.input_layernorm);
            set_gamma(&mut layer.post_attention_layernorm);
            set_gamma(&mut layer.pre_feedforward_layernorm);
            set_gamma(&mut layer.post_feedforward_layernorm);
        }
        set_gamma(&mut model.norm);

        // Apply the fix
        fix_rmsnorm_gemma2_offset(&mut model);

        // Verify all gamma values are now hf_gamma_val + 1.0
        let expected = hf_gamma_val + 1.0;
        let check_gamma = |norm: &burn::nn::RmsNorm<B>, name: &str| {
            let data: Vec<f32> = norm.gamma.val().to_data().to_vec().unwrap();
            for (i, &v) in data.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-5,
                    "{name}[{i}] = {v}, expected {expected}"
                );
            }
        };

        for (i, layer) in model.layers.iter().enumerate() {
            check_gamma(
                &layer.input_layernorm,
                &format!("layers.{i}.input_layernorm.gamma"),
            );
            check_gamma(
                &layer.post_attention_layernorm,
                &format!("layers.{i}.post_attention_layernorm.gamma"),
            );
            check_gamma(
                &layer.pre_feedforward_layernorm,
                &format!("layers.{i}.pre_feedforward_layernorm.gamma"),
            );
            check_gamma(
                &layer.post_feedforward_layernorm,
                &format!("layers.{i}.post_feedforward_layernorm.gamma"),
            );
        }
        check_gamma(&model.norm, "norm.gamma");
    }

    #[test]
    fn test_fix_rmsnorm_gamma_values_after_init() {
        type B = burn_ndarray::NdArray;
        let device = Default::default();
        let config = crate::types::Gemma2Config::gemma2_2b();
        let mut model = Gemma2Model::<B>::new(&config, &device);

        // After init, gamma = ones (1.0). After fix, gamma should be 2.0.
        fix_rmsnorm_gemma2_offset(&mut model);

        // Verify the fix was applied (1.0 + 1.0 = 2.0 for freshly initialized model)
        let data: Vec<f32> = model.norm.gamma.val().to_data().to_vec().unwrap();
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-5,
                "norm.gamma[{i}] = {v}, expected 2.0 (1.0 init + 1.0 fix)"
            );
        }

        // Note: is_require_grad() returns false on NdArray (non-autodiff) backend.
        // Gradient tracking is handled by the Autodiff wrapper in actual training.
        // Param::from_tensor() does call require_grad(), but it's a no-op on non-AD backends.
    }
}
