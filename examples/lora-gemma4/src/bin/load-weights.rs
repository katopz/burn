//! Binary to load real Gemma 4 E4B weights and verify forward pass.
//!
//! Usage:
//!   cargo run -p lora-gemma4 --bin load-weights --features ndarray --release
//!   RUST_LOG=info cargo run -p lora-gemma4 --bin load-weights --features ndarray --release -- --path /path/to/model.safetensors
//!
//! Default path: ~/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it/snapshots/<hash>/model.safetensors
//!
//! Memory: ~30GB RAM needed for F32 (15GB BF16 safetensors → F32 upcast).
//! Uses NdArray (CPU) backend with Bf16ToF32 adapter.
//!
//! Note: Use --release to reduce memory overhead and improve loading speed.

use std::path::PathBuf;

use burn::module::Module;
use burn::tensor::{Int, Tensor};
use clap::Parser;
use lora_gemma4::loader::load_gemma4_weights;
use lora_gemma4::{Gemma4Config, Gemma4Model};

/// Gemma 4 weight loading and forward pass verification.
#[derive(Parser, Debug)]
#[command(
    name = "load-weights",
    about = "Load Gemma 4 E4B weights and run verification"
)]
struct Args {
    /// Path to model.safetensors file.
    /// Defaults to HuggingFace cache for unsloth/gemma-4-E4B-it.
    #[arg(long, short = 'p')]
    path: Option<String>,

    /// Number of tokens for forward pass test.
    #[arg(long, short = 'n', default_value = "16")]
    seq_len: usize,

    /// Skip forward pass (only load weights and report stats).
    #[arg(long, short = 's')]
    skip_forward: bool,
}

/// Resolve the default safetensors path from HuggingFace cache.
fn default_safetensors_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let hub_dir = PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join("models--unsloth--gemma-4-E4B-it");

    if !hub_dir.exists() {
        return None;
    }

    // Find the snapshots directory
    let snapshots_dir = hub_dir.join("snapshots");
    if !snapshots_dir.exists() {
        return None;
    }

    // Find the first snapshot hash directory containing model.safetensors
    for entry in std::fs::read_dir(&snapshots_dir).ok()? {
        let entry = entry.ok()?;
        let model_path = entry.path().join("model.safetensors");
        if model_path.exists() {
            return Some(model_path);
        }
    }

    None
}

/// Run weight loading and verification with a specific backend.
fn run<B: burn::tensor::backend::Backend>(args: Args, device: B::Device) {
    // Resolve safetensors path
    let safetensors_path = match &args.path {
        Some(p) => PathBuf::from(p),
        None => match default_safetensors_path() {
            Some(p) => p,
            None => {
                log::error!(
                    "No safetensors path provided and default cache not found.\n\
                     Download with: huggingface-cli download unsloth/gemma-4-E4B-it\n\
                     Or specify: --path /path/to/model.safetensors"
                );
                std::process::exit(1);
            }
        },
    };

    if !safetensors_path.exists() {
        log::error!("File not found: {}", safetensors_path.display());
        std::process::exit(1);
    }

    let file_size_gb = safetensors_path
        .metadata()
        .map(|m| m.len() as f64 / 1e9)
        .unwrap_or(0.0);
    log::info!(
        "Safetensors file: {} ({file_size_gb:.1} GB)",
        safetensors_path.display()
    );
    log::info!(
        "Estimated memory: ~{:.0} GB (BF16→F32 upcast)",
        file_size_gb * 2.0
    );

    // Build model with full E4B config
    let config = Gemma4Config::gemma4_e4b();
    log::info!(
        "Config: {} layers, {} hidden, {} vocab, {} KV-shared, PLE={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.vocab_size,
        config.num_kv_shared_layers,
        config.has_ple(),
    );
    log::info!(
        "Attention: {} heads, {} KV heads, head_dim={}, global_head_dim={}",
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.global_head_dim,
    );

    log::info!("Initializing model (random weights)...");
    let mut model: Gemma4Model<B> = Gemma4Model::new(&config, &device);

    // Count model parameters using burn's built-in method
    let num_params = model.num_params();
    let num_params_b = num_params as f64 / 1e9;
    log::info!("Model parameters: {num_params:.0} ({num_params_b:.2}B)");

    // Load weights
    log::info!("Loading weights from safetensors...");
    let load_start = std::time::Instant::now();

    let report = match load_gemma4_weights(&mut model, &safetensors_path, &device) {
        Ok(r) => r,
        Err(e) => {
            log::error!("Failed to load weights: {e}");
            std::process::exit(1);
        }
    };

    let load_elapsed = load_start.elapsed();
    log::info!(
        "Weight loading complete in {:.1}s",
        load_elapsed.as_secs_f64()
    );

    // Report stats
    log::info!(
        "Loaded {} tensors from {} files",
        report.tensors_loaded,
        report.files_read.len(),
    );

    // Categorize skipped tensors
    let layer_scalar_count = report
        .tensors_skipped
        .iter()
        .filter(|s| s.contains("layer_scalar"))
        .count();
    let non_lm_count = report
        .tensors_skipped
        .iter()
        .filter(|s| !s.starts_with("model.language_model.") && !s.contains("layer_scalar"))
        .count();
    let other_skipped = report.tensors_skipped.len() - layer_scalar_count - non_lm_count;

    log::info!(
        "Skipped: {} layer_scalar, {} non-LM (vision/audio), {} other",
        layer_scalar_count,
        non_lm_count,
        other_skipped,
    );

    // Expected counts
    let expected_loadable = 677; // 42 layers × 16 per-layer + 5 top-level
    if report.tensors_loaded != expected_loadable {
        log::warn!(
            "Expected ~{expected_loadable} loadable tensors, got {}. \
             Check name mapping if this differs significantly.",
            report.tensors_loaded,
        );
    }

    // Check tied weights
    if model.tie_word_embeddings {
        let embed_shape = model.embed.weight.val().shape();
        let lm_shape = model.lm_head.weight.val().shape();
        log::info!(
            "Tied embeddings: embed.weight={:?}, lm_head.weight={:?}",
            embed_shape.dims::<2>(),
            lm_shape.dims::<2>(),
        );
    }

    if args.skip_forward {
        log::info!("Skipping forward pass (--skip-forward). Weight loading verification complete.");
        return;
    }

    // Forward pass verification
    let seq_len = args.seq_len;
    log::info!("Running forward pass with seq_len={seq_len}...");

    let input_ids = Tensor::<B, 2, Int>::zeros([1, seq_len], &device);
    let fwd_start = std::time::Instant::now();

    let logits = model.forward(input_ids);
    let fwd_elapsed = fwd_start.elapsed();

    let [batch, seq, vocab] = logits.dims();
    log::info!(
        "Forward pass complete in {:.1}s — logits shape: [{batch}, {seq}, {vocab}]",
        fwd_elapsed.as_secs_f64(),
    );

    // Verify logits statistics
    let logits_data = logits.to_data();
    let logits_vec: Vec<f32> = logits_data.to_vec().unwrap_or_default();

    if logits_vec.is_empty() {
        log::error!("Logits are empty!");
        std::process::exit(1);
    }

    let has_nan = logits_vec.iter().any(|v: &f32| v.is_nan());
    let has_inf = logits_vec.iter().any(|v: &f32| v.is_infinite());
    let min_val = logits_vec.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_val = logits_vec.iter().sum::<f32>() / logits_vec.len() as f32;

    log::info!("Logits statistics:");
    log::info!("  min:   {min_val:.4}");
    log::info!("  max:   {max_val:.4}");
    log::info!("  mean:  {mean_val:.4}");
    log::info!("  NaN:   {has_nan}");
    log::info!("  Inf:   {has_inf}");

    // Check if logits are within reasonable range after softcapping (±30)
    let within_range = min_val >= -30.0 && max_val <= 30.0;
    if has_nan {
        log::error!(
            "FAIL: Logits contain NaN values — model weights may be corrupted or name mapping is wrong."
        );
        std::process::exit(1);
    }
    if has_inf {
        log::error!("FAIL: Logits contain Inf values — numerical overflow during forward pass.");
        std::process::exit(1);
    }
    if !within_range {
        log::warn!(
            "WARN: Logits outside softcap range [{min_val:.1}, {max_val:.1}]. \
             Expected within [-30, 30] after softcapping."
        );
    }

    // Sample a few token predictions (argmax of last position)
    let last_token_logits = logits
        .clone()
        .slice([0..1, seq_len - 1..seq_len, 0..vocab])
        .reshape([vocab]);
    let predicted_token: i64 = last_token_logits
        .argmax(0)
        .into_data()
        .iter::<i64>()
        .next()
        .unwrap_or(0);
    log::info!("Last position predicted token ID: {predicted_token}");

    log::info!("✓ Weight loading and forward pass verification PASSED");
}

// ---------------------------------------------------------------------------
// Main: Select Backend
// ---------------------------------------------------------------------------

#[cfg(feature = "ndarray")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    let args = Args::parse();
    let device = Default::default();
    run::<B>(args, device);
}

#[cfg(feature = "wgpu")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn::backend::wgpu::Wgpu;

    type B = Wgpu<f32, i64>;

    let args = Args::parse();
    let device = Default::default();
    run::<B>(args, device);
}

#[cfg(feature = "metal")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn::backend::metal::Metal;

    type B = Metal<f32, i64>;

    let args = Args::parse();
    let device = Default::default();
    run::<B>(args, device);
}

#[cfg(not(any(feature = "ndarray", feature = "wgpu", feature = "metal")))]
fn main() {
    compile_error!(
        "Enable a backend feature:\n  \
         --features ndarray    (CPU, for testing)\n  \
         --features wgpu       (WGPU/Metal)\n  \
         --features metal      (Metal native)"
    );
}
