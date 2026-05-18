//! NaN Investigation: Compare f16 vs f32 Metal Backend
//!
//! Tests forward pass, CE loss, and backward pass to isolate the exact NaN source.
//!
//! ```sh
//! cargo run -p lora-gemma2 --features metal --bin test-nan-f16-vs-f32 -- \
//!   --weights /path/to/gemma-2-2b/ \
//!   --dataset /path/to/train.jsonl
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use burn::module::AutodiffModule;
use burn::nn::lora::{LoraBias, LoraConfig};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::{Element, FloatDType, Int, Tensor};

use lora_gemma2::batcher::SFTBatcher;
use lora_gemma2::dataset::JsonlDataset;
use lora_gemma2::loader::load_gemma2_weights_dtype;
use lora_gemma2::model::Gemma2Model;
use lora_gemma2::model_lora::{
    Gemma2ModelLora, apply_lora_to_gemma2, count_lora_params, count_total_params,
};
use lora_gemma2::tokenizer::GemmaTokenizer;
use lora_gemma2::types::{Gemma2Config, LoraTarget};

// ---------------------------------------------------------------------------
// CLI Args
// ---------------------------------------------------------------------------

struct TestArgs {
    weights: String,
    dataset: String,
    tokenizer: Option<String>,
    lora_rank: usize,
    max_seq_length: usize,
    iterations: usize,
}

impl TestArgs {
    fn parse() -> Self {
        let mut args = Self {
            weights: String::new(),
            dataset: String::new(),
            tokenizer: None,
            lora_rank: 4,
            max_seq_length: 512,
            iterations: 5,
        };

        let cli: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < cli.len() {
            match cli[i].as_str() {
                "--weights" => {
                    args.weights = cli.get(i + 1).expect("--weights required").clone();
                    i += 2;
                }
                "--dataset" => {
                    args.dataset = cli.get(i + 1).expect("--dataset required").clone();
                    i += 2;
                }
                "--tokenizer" => {
                    args.tokenizer = Some(cli.get(i + 1).expect("--tokenizer required").clone());
                    i += 2;
                }
                "--lora-rank" => {
                    args.lora_rank = cli
                        .get(i + 1)
                        .expect("--lora-rank required")
                        .parse()
                        .expect("invalid lora-rank");
                    i += 2;
                }
                "--max-seq-length" => {
                    args.max_seq_length = cli
                        .get(i + 1)
                        .expect("--max-seq-length required")
                        .parse()
                        .expect("invalid max-seq-length");
                    i += 2;
                }
                "--iterations" => {
                    args.iterations = cli
                        .get(i + 1)
                        .expect("--iterations required")
                        .parse()
                        .expect("invalid iterations");
                    i += 2;
                }
                "--help" | "-h" => {
                    eprintln!("Usage: test-nan-f16-vs-f32 [OPTIONS]");
                    eprintln!();
                    eprintln!("Options:");
                    eprintln!("  --weights PATH        Path to gemma-2-2b safetensors (required)");
                    eprintln!("  --dataset PATH        Path to train.jsonl (required)");
                    eprintln!("  --tokenizer PATH      Path to tokenizer.json (optional)");
                    eprintln!("  --lora-rank N         LoRA rank (default: 4)");
                    eprintln!("  --max-seq-length N    Max sequence length (default: 512)");
                    eprintln!("  --iterations N        Training iterations to test (default: 5)");
                    std::process::exit(0);
                }
                other => {
                    eprintln!("Unknown argument: {other}");
                    std::process::exit(1);
                }
            }
        }

        if args.weights.is_empty() {
            eprintln!("Error: --weights is required");
            std::process::exit(1);
        }
        if args.dataset.is_empty() {
            eprintln!("Error: --dataset is required");
            std::process::exit(1);
        }

        args
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check a float tensor for NaN. Returns true if NaN found.
fn has_nan_f32<B: burn::tensor::backend::Backend, const D: usize>(tensor: &Tensor<B, D>) -> bool
where
    B::FloatElem: ToElement,
{
    tensor.clone().contains_nan().into_scalar().to_bool()
}

/// Get min/max of a float tensor as f32.
fn min_max<B: burn::tensor::backend::Backend, const D: usize>(tensor: &Tensor<B, D>) -> (f32, f32)
where
    B::FloatElem: ToElement,
{
    let min_val = tensor
        .clone()
        .min()
        .into_data()
        .iter::<B::FloatElem>()
        .next()
        .unwrap()
        .to_f32();
    let max_val = tensor
        .clone()
        .max()
        .into_data()
        .iter::<B::FloatElem>()
        .next()
        .unwrap()
        .to_f32();
    (min_val, max_val)
}

/// CE loss computation matching the TrainStep standard path.
/// Casts logits to f32 before log_softmax to avoid f16 overflow with 256K vocab.
fn compute_ce_loss<B: burn::tensor::backend::Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    let [batch_size, seq_len, _vocab_size] = logits.dims();
    let logits_f32 = logits.cast(FloatDType::F32);
    let target_indices = targets.reshape([batch_size, seq_len, 1]);
    let token_losses = log_softmax(logits_f32, 2)
        .gather(2, target_indices)
        .reshape([batch_size, seq_len])
        .neg();
    // Simple mean (no padding mask for this test)
    token_losses.mean()
}

fn get_process_rss_gb() -> Option<f64> {
    let pid = std::process::id();
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    let rss_kb: f64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;
    Some(rss_kb / 1024.0 / 1024.0)
}

// ---------------------------------------------------------------------------
// Test Runner
// ---------------------------------------------------------------------------

fn run_test<B: AutodiffBackend>(args: TestArgs, device: B::Device)
where
    B::FloatElem: ToElement,
{
    let elem_name = std::any::type_name::<B::FloatElem>()
        .split("::")
        .last()
        .unwrap_or("unknown");

    log::info!("========================================================");
    log::info!("  NaN Investigation Test — {elem_name} Metal Backend");
    log::info!("========================================================\n");

    // -------------------------------------------------------------------
    // 1. Load tokenizer
    // -------------------------------------------------------------------
    log::info!("[1/7] Loading tokenizer");
    let tokenizer = match &args.tokenizer {
        Some(path) => GemmaTokenizer::from_file(path).expect("Failed to load tokenizer"),
        None => GemmaTokenizer::from_pretrained("google/gemma-2-2b")
            .expect("Failed to load tokenizer from HF"),
    };
    log::info!("  vocab_size={}", tokenizer.vocab_size());

    // -------------------------------------------------------------------
    // 2. Build model + load weights
    // -------------------------------------------------------------------
    log::info!("[2/7] Building model and loading weights");
    let config = Gemma2Config::gemma2_2b();
    log::info!(
        "  Config: hidden={}, layers={}, heads={}, vocab={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.vocab_size
    );

    let mut inner_model = Gemma2Model::<B::InnerBackend>::new(&config, &device);

    let weights_path = PathBuf::from(&args.weights);
    log::info!("  Loading weights from '{}'", args.weights);
    match load_gemma2_weights_dtype(
        &mut inner_model,
        weights_path.as_path(),
        &device,
        <B::FloatElem as Element>::dtype(),
    ) {
        Ok(report) => {
            log::info!("  Loaded {} tensors", report.tensors_loaded);
        }
        Err(e) => {
            log::error!("Failed to load weights: {e}");
            std::process::exit(1);
        }
    }

    // -------------------------------------------------------------------
    // 3. Apply LoRA
    // -------------------------------------------------------------------
    log::info!("[3/7] Applying LoRA (rank={})", args.lora_rank);
    let model = Gemma2Model::<B>::from_inner(inner_model);
    let lora_config = LoraConfig::new(args.lora_rank)
        .with_alpha(32.0)
        .with_dropout(0.0)
        .with_bias(LoraBias::None);

    let targets = LoraTarget::all_targets();
    let lora_model: Gemma2ModelLora<B> =
        apply_lora_to_gemma2(model, &lora_config, targets, &device);

    let lora_params = count_lora_params(&lora_model);
    let total_params = count_total_params(&lora_model);
    log::info!(
        "  LoRA params: {lora_params} ({:.2}% of {total_params})",
        lora_params as f64 / total_params as f64 * 100.0
    );

    // -------------------------------------------------------------------
    // 4. Integrity check — dummy forward pass
    // -------------------------------------------------------------------
    log::info!("[4/7] Running integrity check (dummy forward pass)");
    let dummy_input = Tensor::<B, 2, Int>::zeros([1, 8], &device);
    let logits = lora_model.forward(dummy_input);

    let dummy_nan = has_nan_f32(&logits);
    let (min_val, max_val) = min_max(&logits);

    let logits_data = logits.clone().into_data();
    let logits_slice: Vec<f32> = logits_data
        .iter::<B::FloatElem>()
        .take(10)
        .map(|v| v.to_f32())
        .collect();

    log::info!("  Dummy forward logits (first 10): {logits_slice:?}");
    log::info!("  Logits range: [{min_val:.4}, {max_val:.4}]");

    if dummy_nan {
        log::error!("  !! NaN in dummy forward pass");
    } else {
        log::info!("  OK Forward pass clean (no NaN)");
    }

    // -------------------------------------------------------------------
    // 5. Forward + CE loss on real data
    // -------------------------------------------------------------------
    log::info!("[5/7] Testing forward + CE loss on real data");
    let dataset = JsonlDataset::from_file(&args.dataset).expect("Failed to load dataset");
    log::info!("  Dataset: {} samples", dataset.len());

    let tokenizer_arc = Arc::new(tokenizer);
    let batcher = SFTBatcher::new(tokenizer_arc, args.max_seq_length);

    let dataloader = burn::data::dataloader::DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .shuffle(42)
        .build(dataset);

    let mut forward_nan_count = 0usize;
    let mut loss_nan_count = 0usize;
    let mut total_count = 0usize;

    for (i, batch) in dataloader.iter().enumerate() {
        let batch: lora_gemma2::SFTTrainingBatch<B> = batch;
        if i >= args.iterations {
            break;
        }
        total_count += 1;

        // Step A: Forward pass (on autodiff backend — tracks backward graph)
        let logits = lora_model.forward(batch.tokens_inputs.clone());
        let fwd_nan = has_nan_f32(&logits);
        if fwd_nan {
            forward_nan_count += 1;
        }

        // Step B: CE loss (autodiff backend — tracked for backward graph)
        let loss = compute_ce_loss(logits.clone(), batch.targets);

        let loss_nan = has_nan_f32(&loss);
        if loss_nan {
            loss_nan_count += 1;
        }

        let loss_val: f32 = loss
            .clone()
            .into_data()
            .iter::<B::FloatElem>()
            .next()
            .unwrap()
            .to_f32();

        let (logit_min, logit_max) = min_max(&logits);

        let status = match (fwd_nan, loss_nan) {
            (true, _) => "!! FWD",
            (false, true) => "!! LOSS",
            (false, false) => "OK    ",
        };

        log::info!(
            "  Batch {:>3}: {}  loss={:>8.4}  logits=[{:>10.4}, {:>10.4}]",
            i,
            status,
            loss_val,
            logit_min,
            logit_max
        );
    }

    // -------------------------------------------------------------------
    // 6. Full training step (forward + loss + backward)
    // -------------------------------------------------------------------
    log::info!("[6/7] Testing full training step (forward + loss + backward)");

    // Fresh dataset
    let dataset2 = JsonlDataset::from_file(&args.dataset).expect("Failed to load dataset");
    let tokenizer2 = match &args.tokenizer {
        Some(path) => GemmaTokenizer::from_file(path).expect("Failed to load tokenizer"),
        None => {
            GemmaTokenizer::from_pretrained("google/gemma-2-2b").expect("Failed to load tokenizer")
        }
    };
    let batcher2 = SFTBatcher::new(Arc::new(tokenizer2), args.max_seq_length);
    let dataloader2 = burn::data::dataloader::DataLoaderBuilder::new(batcher2)
        .batch_size(1)
        .shuffle(123)
        .build(dataset2);

    let mut train_nan = 0usize;
    let mut train_total = 0usize;

    for (i, batch) in dataloader2.iter().enumerate() {
        let batch: lora_gemma2::SFTTrainingBatch<B> = batch;
        if i >= args.iterations {
            break;
        }
        train_total += 1;

        // Full training step: forward → loss → backward (all on autodiff backend)
        let logits = lora_model.forward(batch.tokens_inputs.clone());
        let loss = compute_ce_loss(logits.clone(), batch.targets);

        let loss_val: f32 = loss
            .clone()
            .into_data()
            .iter::<B::FloatElem>()
            .next()
            .unwrap()
            .to_f32();

        let loss_nan = has_nan_f32(&loss);

        // Backward pass
        let _grads = loss.backward();

        let status = if loss_nan { "!! NaN" } else { "OK    " };
        log::info!("  Train step {:>3}: {}  loss={:>8.4}", i, status, loss_val);

        if loss_nan {
            train_nan += 1;
        }
    }

    // -------------------------------------------------------------------
    // 7. Summary
    // -------------------------------------------------------------------
    log::info!("========================================================");
    log::info!("  Results Summary — {elem_name} Metal Backend");
    log::info!("--------------------------------------------------------");
    log::info!("  Dummy forward NaN:   {}", dummy_nan);
    log::info!(
        "  Forward-only NaN:    {}/{} batches",
        forward_nan_count,
        total_count
    );
    log::info!(
        "  CE loss NaN:         {}/{} batches",
        loss_nan_count,
        total_count
    );
    log::info!(
        "  Full train step NaN: {}/{} batches",
        train_nan,
        train_total
    );
    log::info!("========================================================\n");

    let total_issues = forward_nan_count + loss_nan_count + train_nan;
    if total_issues == 0 && !dummy_nan {
        log::info!("OK {elem_name} backend is CLEAN — no NaN detected in any phase");
        log::info!("   Forward, loss, and backward all produce valid values.");
    } else {
        log::error!("!! {elem_name} backend produces NaN:");
        if dummy_nan {
            log::error!("   - Dummy forward pass: NaN");
        }
        if forward_nan_count > 0 {
            log::error!(
                "   - Forward on real data: {}/{} batches with NaN",
                forward_nan_count,
                total_count
            );
        }
        if loss_nan_count > 0 {
            log::error!(
                "   - CE loss: {}/{} batches with NaN",
                loss_nan_count,
                total_count
            );
        }
        if train_nan > 0 {
            log::error!(
                "   - Full train step: {}/{} batches with NaN",
                train_nan,
                train_total
            );
        }
    }

    if let Some(rss_gb) = get_process_rss_gb() {
        log::info!("  Peak RSS: {rss_gb:.1} GB");
    }
}

// ---------------------------------------------------------------------------
// Backend Selection
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn::backend::{Autodiff, Metal};

    // ─── CHANGE THIS LINE TO TEST f16 vs f32 ──────────────────────
    type FloatType = burn::tensor::f16;
    // type FloatType = f32;
    // ─────────────────────────────────────────────────────────────

    type Backend = Metal<FloatType, i64>;
    type AD = Autodiff<Backend>;

    log::info!(
        "Using Metal backend with {} precision",
        std::any::type_name::<FloatType>()
    );

    let args = TestArgs::parse();
    let device = Default::default();
    run_test::<AD>(args, device);
}

#[cfg(feature = "ndarray")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn::backend::Autodiff;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;
    type AD = Autodiff<Backend>;

    log::info!("Using NdArray CPU backend with f32 precision");

    let args = TestArgs::parse();
    let device = Default::default();
    run_test::<AD>(args, device);
}

#[cfg(not(any(feature = "metal", feature = "ndarray")))]
fn main() {
    panic!(
        "Enable a backend feature:\n  \
         --features metal    (Metal native, macOS)\n  \
         --features ndarray  (CPU, for testing)"
    );
}
