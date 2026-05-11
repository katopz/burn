#![cfg_attr(
    not(any(feature = "ndarray", feature = "wgpu", feature = "metal")),
    allow(dead_code)
)]
//! Gemma 2 inference binary — demonstrates weight loading from HuggingFace safetensors.
//!
//! Usage:
//!   cargo run -p lora-gemma2 --features ndarray --bin gemma2-inference -- /path/to/model.safetensors
//!   cargo run -p lora-gemma2 --features ndarray --bin gemma2-inference -- /path/to/model-directory/
//!
//! This binary validates that weight loading works correctly by:
//! 1. Loading weights from safetensors files
//! 2. Running a forward pass with dummy input
//! 3. Printing output tensor statistics

use std::path::PathBuf;

use burn::prelude::ElementConversion;
use burn::tensor::{Element, Int, Tensor};
use lora_gemma2::loader::load_gemma2_weights_dtype;
use lora_gemma2::{Gemma2Config, Gemma2Model, LoadReport};

/// Run inference with a specific backend.
fn run_inference<B: burn::tensor::backend::Backend>(model_path: PathBuf, device: B::Device) {
    let config = Gemma2Config::gemma2_2b();
    let mut model = Gemma2Model::<B>::new(&config, &device);

    log::info!("Gemma 2 2B model initialized");
    log::info!("  hidden_size: {}", config.hidden_size);
    log::info!("  num_layers: {}", config.num_hidden_layers);
    log::info!("  num_heads: {}", config.num_attention_heads);
    log::info!("  vocab_size: {}", config.vocab_size);

    // Load weights
    log::info!("Loading weights from: {}", model_path.display());

    let report = match load_gemma2_weights_dtype(
        &mut model,
        &model_path,
        &device,
        <B::FloatElem as Element>::dtype(),
    ) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("Error loading weights: {e}");
            std::process::exit(1);
        }
    };

    print_load_report(&report);

    // Run a forward pass with dummy input to validate
    log::info!("Running forward pass with dummy input...");
    let seq_len = 8;
    let input_ids = Tensor::<B, 2, Int>::zeros([1, seq_len], &device);

    let logits = model.forward(input_ids);

    let [batch, seq, vocab] = logits.dims();
    log::info!("Forward pass complete");
    log::info!("  Output shape: [{batch}, {seq}, {vocab}]");

    // Print some statistics
    let logits_f = logits.clone();
    let min = logits_f.clone().min().into_scalar();
    let max = logits_f.clone().max().into_scalar();
    let mean = logits_f.mean().into_scalar();
    log::info!("  Logit stats: min={min:.4}, max={max:.4}, mean={mean:.4}");

    // Verify output is not all zeros (indicates weights loaded correctly)
    let min_val: f64 = min.elem();
    let max_val: f64 = max.elem();
    if min_val == 0.0 && max_val == 0.0 {
        log::warn!("Output is all zeros — weights may not have loaded correctly");
    } else {
        log::info!("Output contains non-zero values — weights loaded successfully");
    }

    log::info!("Done!");
}

/// Parse CLI args and return model path.
fn parse_args() -> PathBuf {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1) {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("Usage: gemma2-inference <model-path>");
            eprintln!();
            eprintln!("Arguments:");
            eprintln!("  <model-path>  Path to safetensors file, directory, or index JSON");
            eprintln!();
            eprintln!("Examples:");
            eprintln!("  gemma2-inference model.safetensors");
            eprintln!("  gemma2-inference /path/to/gemma-2-2b/");
            eprintln!("  gemma2-inference model.safetensors.index.json");
            std::process::exit(1);
        }
    }
}

fn print_load_report(report: &LoadReport) {
    log::info!("Load report:");
    log::info!("  Tensors loaded: {}", report.tensors_loaded);
    log::info!("  Files read: {}", report.files_read.len());
    for file in &report.files_read {
        log::info!("    - {file}");
    }
    if !report.tensors_skipped.is_empty() {
        log::info!("  Tensors skipped: {}", report.tensors_skipped.len());
    }
}

// ---------------------------------------------------------------------------
// Main: Select Backend
// ---------------------------------------------------------------------------

#[cfg(feature = "ndarray")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    let model_path = parse_args();
    let device = Default::default();
    run_inference::<B>(model_path, device);
}

#[cfg(feature = "wgpu")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn::backend::wgpu::Wgpu;

    type B = Wgpu<f32, i64>;

    let model_path = parse_args();
    let device = Default::default();
    run_inference::<B>(model_path, device);
}

#[cfg(feature = "metal")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn::backend::Metal;
    use burn::tensor::f16;

    // Use f16 on Metal to halve memory.
    // Metal does not support BF16, so BF16 weights are converted to F16 during loading.
    type B = Metal<f16, i64>;

    let model_path = parse_args();
    let device = Default::default();
    run_inference::<B>(model_path, device);
}

#[cfg(not(any(feature = "ndarray", feature = "wgpu", feature = "metal")))]
fn main() {
    panic!(
        "Enable a backend feature:\n  \
         --features ndarray    (CPU, for testing)\n  \
         --features wgpu       (WGPU/Metal)\n  \
         --features metal      (Metal native)"
    );
}
