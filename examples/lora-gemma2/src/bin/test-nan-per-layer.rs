//! Per-Layer NaN Diagnostic for Gemma 2 2B
//!
//! Runs the model forward pass with NaN/value-range checks after every sub-layer
//! to pinpoint exactly where f16 overflow originates.
//!
//! ```sh
//! # f16 (the failing case)
//! cargo run -p lora-gemma2 --features metal --bin test-nan-per-layer -- \
//!   --weights /path/to/gemma-2-2b/ \
//!   --dataset /path/to/train.jsonl \
//!   --dtype f16
//!
//! # f32 (the baseline — should be clean)
//! cargo run -p lora-gemma2 --features metal --bin test-nan-per-layer -- \
//!   --weights /path/to/gemma-2-2b/ \
//!   --dataset /path/to/train.jsonl \
//!   --dtype f32
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use burn::tensor::DType;
use burn::tensor::cast::ToElement;

use lora_gemma2::dataset::JsonlDataset;
use lora_gemma2::loader::load_gemma2_weights_dtype;
use lora_gemma2::model::Gemma2Model;
use lora_gemma2::tokenizer::GemmaTokenizer;
use lora_gemma2::types::Gemma2Config;

// ---------------------------------------------------------------------------
// CLI Args
// ---------------------------------------------------------------------------

struct DiagArgs {
    weights: String,
    dataset: String,
    tokenizer: Option<String>,
    dtype: String,
    max_seq_length: usize,
    batches: usize,
}

impl DiagArgs {
    fn parse() -> Self {
        let mut args = Self {
            weights: String::new(),
            dataset: String::new(),
            tokenizer: None,
            dtype: "f16".to_string(),
            max_seq_length: 256,
            batches: 3,
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
                "--dtype" => {
                    args.dtype = cli.get(i + 1).expect("--dtype required").clone();
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
                "--batches" => {
                    args.batches = cli
                        .get(i + 1)
                        .expect("--batches required")
                        .parse()
                        .expect("invalid batches");
                    i += 2;
                }
                "--help" | "-h" => {
                    eprintln!("Usage: test-nan-per-layer [OPTIONS]");
                    eprintln!();
                    eprintln!("Options:");
                    eprintln!("  --weights PATH         Path to gemma-2-2b safetensors (required)");
                    eprintln!("  --dataset PATH         Path to train.jsonl (required)");
                    eprintln!("  --tokenizer PATH       Path to tokenizer.json (optional)");
                    eprintln!("  --dtype {{f16,f32}}      Data type (default: f16)");
                    eprintln!("  --max-seq-length N     Max sequence length (default: 256)");
                    eprintln!("  --batches N            Number of batches to test (default: 3)");
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
        if !matches!(args.dtype.as_str(), "f16" | "f32") {
            eprintln!("Error: --dtype must be 'f16' or 'f32'");
            std::process::exit(1);
        }

        args
    }
}

// ---------------------------------------------------------------------------
// f16 runner
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
fn run_f16(args: DiagArgs) {
    use burn::backend::Metal;

    type B = Metal<burn::tensor::f16, i64>;
    let device = Default::default();
    run_diagnostic::<B>(args, device, DType::F16);
}

// ---------------------------------------------------------------------------
// f32 runner
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
fn run_f32(args: DiagArgs) {
    use burn::backend::Metal;

    type B = Metal<f32, i64>;
    let device = Default::default();
    run_diagnostic::<B>(args, device, DType::F32);
}

// ---------------------------------------------------------------------------
// Core diagnostic
// ---------------------------------------------------------------------------

fn run_diagnostic<B: burn::tensor::backend::Backend>(
    args: DiagArgs,
    device: B::Device,
    dtype: DType,
) where
    B::FloatElem: ToElement,
{
    let elem_name = format!("{dtype:?}");
    log::info!("========================================================");
    log::info!("  Per-Layer NaN Diagnostic — {elem_name}");
    log::info!("========================================================\n");

    // 1. Load tokenizer
    log::info!("[1/4] Loading tokenizer");
    let tokenizer = match &args.tokenizer {
        Some(path) => GemmaTokenizer::from_file(path).expect("Failed to load tokenizer"),
        None => GemmaTokenizer::from_pretrained("google/gemma-2-2b")
            .expect("Failed to load tokenizer from HF"),
    };

    // 2. Build model + load weights
    log::info!("[2/4] Building model and loading weights");
    let config = Gemma2Config::gemma2_2b();
    log::info!(
        "  hidden={}, layers={}, heads={}, vocab={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.vocab_size
    );

    let mut model = Gemma2Model::<B>::new(&config, &device);
    let weights_path = PathBuf::from(&args.weights);
    log::info!("  Loading weights from '{}'", args.weights);
    match load_gemma2_weights_dtype(&mut model, weights_path.as_path(), &device, dtype) {
        Ok(report) => {
            log::info!("  Loaded {} tensors", report.tensors_loaded);
        }
        Err(e) => {
            log::error!("Failed to load weights: {e}");
            std::process::exit(1);
        }
    }

    // 3. Prepare dataset
    log::info!("[3/4] Loading dataset");
    let dataset = JsonlDataset::from_file(&args.dataset).expect("Failed to load dataset");
    log::info!("  Dataset: {} samples", dataset.len());

    let tokenizer_arc = Arc::new(tokenizer);
    let batcher = lora_gemma2::SFTBatcher::new(tokenizer_arc, args.max_seq_length);

    let dataloader = burn::data::dataloader::DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .shuffle(42)
        .build(dataset);

    // 4. Run diagnostic on batches
    log::info!(
        "[4/4] Running per-layer diagnostic on {batches} batches\n",
        batches = args.batches
    );

    let mut first_nan_layer: Option<String> = None;
    let mut batches_with_nan = 0usize;
    let mut total_checks = 0usize;
    let mut total_nan_checks = 0usize;

    for (batch_idx, batch) in dataloader.iter().enumerate() {
        let batch: lora_gemma2::SFTTrainingBatch<B> = batch;
        if batch_idx >= args.batches {
            break;
        }

        log::info!("═══════════════════════════════════════════════════════");
        log::info!("  Batch {batch_idx}");
        log::info!("═══════════════════════════════════════════════════════");

        let (logits, checks) = model.forward_diagnostic(batch.tokens_inputs);

        let logits_nan = logits.clone().contains_nan().into_scalar().to_bool();
        let logits_min: f32 = logits.clone().min().into_scalar().to_f32();
        let logits_max: f32 = logits.clone().max().into_scalar().to_f32();

        let mut batch_first_nan: Option<String> = None;
        let mut batch_nan_count = 0usize;

        for check in &checks {
            let marker = if check.has_nan {
                batch_nan_count += 1;
                if batch_first_nan.is_none() {
                    batch_first_nan = Some(check.name.clone());
                }
                if first_nan_layer.is_none() {
                    first_nan_layer = Some(format!("batch_{batch_idx}/{}", check.name));
                }
                "◀◀◀"
            } else {
                "   "
            };

            log::info!("{marker}{check}");
        }

        total_checks += checks.len();
        total_nan_checks += batch_nan_count;

        if batch_nan_count > 0 {
            batches_with_nan += 1;
        }

        log::info!("───────────────────────────────────────────────────────");
        log::info!(
            "  Final logits: nan={}, range=[{:.4}, {:.4}]",
            logits_nan,
            logits_min,
            logits_max
        );

        if let Some(ref first) = batch_first_nan {
            log::warn!(
                "  First NaN at: {first} ({batch_nan_checks} total NaN checkpoints)",
                batch_nan_checks = batch_nan_count
            );
        } else {
            log::info!("  ✓ All checkpoints clean — no NaN detected");
        }
        log::info!("");
    }

    // Summary
    log::info!("========================================================");
    log::info!("  Summary — {elem_name}");
    log::info!("--------------------------------------------------------");
    log::info!(
        "  Batches tested:     {batches_tested}",
        batches_tested = args.batches
    );
    log::info!("  Batches with NaN:   {batches_with_nan}");
    log::info!("  Total checkpoints:  {total_checks}");
    log::info!("  NaN checkpoints:    {total_nan_checks}");

    if let Some(ref first) = first_nan_layer {
        log::warn!("  First NaN ever at:  {first}");
    } else {
        log::info!("  ✓ No NaN detected in any layer across all batches");
    }

    log::info!("========================================================\n");

    if batches_with_nan > 0 {
        log::error!(
            "!! {elem_name} produces NaN — see per-layer details above to identify the failing operation"
        );
        std::process::exit(1);
    } else {
        log::info!("✓ {elem_name} is CLEAN — all forward pass layers produce valid values");
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = DiagArgs::parse();

    match args.dtype.as_str() {
        #[cfg(feature = "metal")]
        "f16" => run_f16(args),
        #[cfg(feature = "metal")]
        "f32" => run_f32(args),
        #[cfg(not(feature = "metal"))]
        _ => {
            eprintln!("Error: --features metal is required");
            std::process::exit(1);
        }
        #[cfg(feature = "metal")]
        _ => unreachable!(),
    }
}
