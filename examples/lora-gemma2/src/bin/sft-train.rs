//! SFT (Supervised Fine-Tuning) training binary for LoRA-adapted Gemma 2.
//!
//! Fine-tunes a Gemma 2 model using LoRA on a JSONL dataset with
//! next-token prediction (causal language modeling).
//!
//! # Usage
//!
//! ```sh
//! # CPU backend (testing)
//! cargo run -p lora-gemma2 --features ndarray --bin sft-train -- \
//!   --dataset input/train.jsonl \
//!   --output-dir /tmp/sft-output
//!
//! # WGPU/Metal backend (macOS)
//! cargo run -p lora-gemma2 --features wgpu --bin sft-train -- \
//!   --dataset input/train.jsonl \
//!   --output-dir /tmp/sft-output \
//!   --weights /path/to/gemma-2-2b/
//! ```
//!
//! # Arguments
//!
//! | Flag | Default | Description |
//! |------|---------|-------------|
//! | `--dataset` | (required) | Path to JSONL training data |
//! | `--output-dir` | `/tmp/sft-output` | Directory for adapters and checkpoints |
//! | `--model-name` | `google/gemma-2-2b` | HF model name for tokenizer |
//! | `--weights` | (none) | Path to safetensors model weights |
//! | `--lora-rank` | `16` | LoRA rank (low-rank dimension) |
//! | `--lora-alpha` | `32.0` | LoRA alpha (scaling factor) |
//! | `--epochs` | `3` | Number of training epochs |
//! | `--batch-size` | `4` | Training batch size |
//! | `--lr` | `2e-4` | Learning rate |
//! | `--max-seq-length` | `2048` | Maximum sequence length |
//! | `--val-split` | `0.1` | Validation split ratio |
//! | `--seed` | `42` | Random seed |

use std::path::PathBuf;
use std::sync::Arc;

use burn::data::dataloader::DataLoaderBuilder;
use burn::nn::lora::{LoraBias, LoraConfig};
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::LossMetric;
use burn::train::{Learner, SupervisedTraining};

use lora_gemma2::batcher::SFTBatcher;
use lora_gemma2::dataset::JsonlDataset;
use lora_gemma2::loader::load_gemma2_weights;
use lora_gemma2::model::Gemma2Model;
use lora_gemma2::model_lora::{
    Gemma2ForSFT, apply_lora_to_gemma2, count_lora_params, count_total_params,
};
use lora_gemma2::tokenizer::GemmaTokenizer;
use lora_gemma2::types::{Gemma2Config, LoraTarget};

// ---------------------------------------------------------------------------
// CLI Arguments
// ---------------------------------------------------------------------------

/// SFT training configuration parsed from CLI arguments.
struct SftArgs {
    /// Path to JSONL training data (required).
    dataset: String,
    /// Directory for output (adapters, checkpoints).
    output_dir: String,
    /// HF model name for tokenizer download.
    model_name: String,
    /// Path to safetensors model weights (optional, random init if omitted).
    weights: Option<String>,
    /// LoRA rank.
    lora_rank: usize,
    /// LoRA alpha.
    lora_alpha: f64,
    /// Number of training epochs.
    epochs: usize,
    /// Training batch size.
    batch_size: usize,
    /// Learning rate.
    learning_rate: f64,
    /// Maximum sequence length for tokenization.
    max_seq_length: usize,
    /// Validation split ratio (0.0 to 1.0).
    val_split: f64,
    /// Random seed.
    seed: u64,
    /// Path to local tokenizer.json file (overrides --model-name for tokenizer).
    tokenizer: Option<String>,
}

impl SftArgs {
    /// Parse CLI arguments with defaults.
    fn parse() -> Self {
        let mut args = Self {
            dataset: String::new(),
            output_dir: "/tmp/sft-output".into(),
            model_name: "google/gemma-2-2b".into(),
            weights: None,
            lora_rank: 16,
            lora_alpha: 32.0,
            epochs: 3,
            batch_size: 4,
            learning_rate: 2e-4,
            max_seq_length: 2048,
            val_split: 0.1,
            seed: 42,
            tokenizer: None,
        };

        let cli: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < cli.len() {
            match cli[i].as_str() {
                "--dataset" => {
                    args.dataset = cli.get(i + 1).expect("--dataset requires a value").clone();
                    i += 2;
                }
                "--output-dir" => {
                    args.output_dir = cli
                        .get(i + 1)
                        .expect("--output-dir requires a value")
                        .clone();
                    i += 2;
                }
                "--model-name" => {
                    args.model_name = cli
                        .get(i + 1)
                        .expect("--model-name requires a value")
                        .clone();
                    i += 2;
                }
                "--weights" => {
                    args.weights =
                        Some(cli.get(i + 1).expect("--weights requires a value").clone());
                    i += 2;
                }
                "--lora-rank" => {
                    args.lora_rank = cli
                        .get(i + 1)
                        .expect("--lora-rank requires a value")
                        .parse()
                        .expect("invalid lora-rank");
                    i += 2;
                }
                "--lora-alpha" => {
                    args.lora_alpha = cli
                        .get(i + 1)
                        .expect("--lora-alpha requires a value")
                        .parse()
                        .expect("invalid lora-alpha");
                    i += 2;
                }
                "--epochs" => {
                    args.epochs = cli
                        .get(i + 1)
                        .expect("--epochs requires a value")
                        .parse()
                        .expect("invalid epochs");
                    i += 2;
                }
                "--batch-size" => {
                    args.batch_size = cli
                        .get(i + 1)
                        .expect("--batch-size requires a value")
                        .parse()
                        .expect("invalid batch-size");
                    i += 2;
                }
                "--lr" => {
                    args.learning_rate = cli
                        .get(i + 1)
                        .expect("--lr requires a value")
                        .parse()
                        .expect("invalid lr");
                    i += 2;
                }
                "--max-seq-length" => {
                    args.max_seq_length = cli
                        .get(i + 1)
                        .expect("--max-seq-length requires a value")
                        .parse()
                        .expect("invalid max-seq-length");
                    i += 2;
                }
                "--val-split" => {
                    args.val_split = cli
                        .get(i + 1)
                        .expect("--val-split requires a value")
                        .parse()
                        .expect("invalid val-split");
                    i += 2;
                }
                "--seed" => {
                    args.seed = cli
                        .get(i + 1)
                        .expect("--seed requires a value")
                        .parse()
                        .expect("invalid seed");
                    i += 2;
                }
                "--tokenizer" => {
                    args.tokenizer = Some(
                        cli.get(i + 1)
                            .expect("--tokenizer requires a value")
                            .clone(),
                    );
                    i += 2;
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => {
                    eprintln!("Unknown argument: {other}");
                    print_usage();
                    std::process::exit(1);
                }
            }
        }

        if args.dataset.is_empty() {
            eprintln!("Error: --dataset is required");
            print_usage();
            std::process::exit(1);
        }

        args
    }
}

fn print_usage() {
    eprintln!("Usage: sft-train [OPTIONS]");
    eprintln!();
    eprintln!("Required:");
    eprintln!("  --dataset <PATH>         JSONL training data path");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --output-dir <DIR>       Output directory (default: /tmp/sft-output)");
    eprintln!(
        "  --model-name <NAME>      HF model name for tokenizer (default: google/gemma-2-2b)"
    );
    eprintln!("  --tokenizer <PATH>       Local tokenizer.json path (overrides --model-name)");
    eprintln!("  --weights <PATH>         Safetensors model weights path");
    eprintln!("  --lora-rank <N>          LoRA rank (default: 16)");
    eprintln!("  --lora-alpha <F>         LoRA alpha (default: 32.0)");
    eprintln!("  --epochs <N>             Training epochs (default: 3)");
    eprintln!("  --batch-size <N>         Batch size (default: 4)");
    eprintln!("  --lr <F>                 Learning rate (default: 2e-4)");
    eprintln!("  --max-seq-length <N>     Max sequence length (default: 2048)");
    eprintln!("  --val-split <F>          Validation split ratio (default: 0.1)");
    eprintln!("  --seed <N>               Random seed (default: 42)");
}

// ---------------------------------------------------------------------------
// Training Pipeline
// ---------------------------------------------------------------------------

/// Run the SFT training pipeline.
///
/// Generic over any autodiff backend (NdArray, Wgpu, Metal, etc.)
fn run<B: AutodiffBackend>(args: SftArgs, device: B::Device) {
    log::info!("=== LoRA SFT Training for Gemma 2 ===\n");

    // -----------------------------------------------------------------------
    // 1. Load Tokenizer
    // -----------------------------------------------------------------------
    let tokenizer = match &args.tokenizer {
        Some(path) => {
            log::info!("[1/8] Loading tokenizer from file '{path}'");
            match GemmaTokenizer::from_file(path) {
                Ok(t) => {
                    log::info!("  Tokenizer loaded (vocab_size={})", t.vocab_size());
                    t
                }
                Err(e) => {
                    log::error!("Failed to load tokenizer: {e}");
                    std::process::exit(1);
                }
            }
        }
        None => {
            log::info!("[1/8] Loading tokenizer from '{}'", args.model_name);
            match GemmaTokenizer::from_pretrained(&args.model_name) {
                Ok(t) => {
                    log::info!("  Tokenizer loaded (vocab_size={})", t.vocab_size());
                    t
                }
                Err(e) => {
                    log::error!("Failed to load tokenizer: {e}");
                    log::error!("Set HF_TOKEN or use --tokenizer /path/to/tokenizer.json");
                    std::process::exit(1);
                }
            }
        }
    };
    let pad_token_id = tokenizer.pad_token_id();
    log::info!("  pad_token_id={pad_token_id}");

    // -----------------------------------------------------------------------
    // 2. Load Dataset
    // -----------------------------------------------------------------------
    log::info!("[2/8] Loading dataset from '{}'", args.dataset);
    let dataset = match JsonlDataset::from_file(&args.dataset) {
        Ok(d) => {
            log::info!("  Loaded {} samples", d.len());
            d
        }
        Err(e) => {
            log::error!("Failed to load dataset: {e}");
            std::process::exit(1);
        }
    };

    if dataset.is_empty() {
        log::error!("Dataset is empty");
        std::process::exit(1);
    }

    // -----------------------------------------------------------------------
    // 3. Split Dataset
    // -----------------------------------------------------------------------
    let (train_dataset, val_dataset) = dataset.split(1.0_f32 - args.val_split as f32);
    log::info!(
        "  Split: {} train, {} val ({:.0}% val split)",
        train_dataset.len(),
        val_dataset.len(),
        args.val_split * 100.0
    );

    if val_dataset.is_empty() {
        log::warn!(
            "Validation set is empty. Consider using a larger dataset or smaller val-split."
        );
    }

    // -----------------------------------------------------------------------
    // 4. Create Dataloaders
    // -----------------------------------------------------------------------
    log::info!(
        "[3/8] Creating dataloaders (batch_size={})",
        args.batch_size
    );
    let tokenizer_arc = Arc::new(tokenizer);

    let batcher_train = SFTBatcher::new(tokenizer_arc.clone(), args.max_seq_length);
    let batcher_val = SFTBatcher::new(tokenizer_arc.clone(), args.max_seq_length);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(args.batch_size)
        .shuffle(args.seed)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_val)
        .batch_size(args.batch_size)
        .build(val_dataset);

    // -----------------------------------------------------------------------
    // 5. Build Model
    // -----------------------------------------------------------------------
    log::info!("[4/8] Building Gemma 2 model");
    let config = Gemma2Config::gemma2_2b();
    log::info!(
        "  Config: hidden={}, layers={}, heads={}, vocab={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.vocab_size
    );

    let mut model = Gemma2Model::<B>::new(&config, &device);

    // -----------------------------------------------------------------------
    // 6. Load Pretrained Weights (if provided)
    // -----------------------------------------------------------------------
    match &args.weights {
        Some(weights_path) => {
            log::info!("[5/8] Loading weights from '{weights_path}'");
            match load_gemma2_weights(&mut model, PathBuf::from(weights_path).as_path(), &device) {
                Ok(report) => {
                    log::info!(
                        "  Loaded {} tensors from {} files",
                        report.tensors_loaded,
                        report.files_read.len()
                    );
                }
                Err(e) => {
                    log::error!("Failed to load weights: {e}");
                    log::error!(
                        "Training will continue with random initialization (not recommended for production)."
                    );
                }
            }
        }
        None => {
            log::warn!("[5/8] No --weights provided, using random initialization");
            log::warn!("  This is only useful for testing the training pipeline.");
        }
    }

    // -----------------------------------------------------------------------
    // 7. Apply LoRA
    // -----------------------------------------------------------------------
    log::info!(
        "[6/8] Applying LoRA (rank={}, alpha={})",
        args.lora_rank,
        args.lora_alpha
    );
    let lora_config = LoraConfig::new(args.lora_rank)
        .with_alpha(args.lora_alpha)
        .with_dropout(0.0)
        .with_bias(LoraBias::None);

    let targets = LoraTarget::all_targets();
    let lora_model = apply_lora_to_gemma2(model, &lora_config, targets, &device);

    let lora_params = count_lora_params(&lora_model);
    let total_params = count_total_params(&lora_model);
    let pct = lora_params as f64 / total_params as f64 * 100.0;
    log::info!("  LoRA params: {lora_params} ({pct:.2}% of {total_params})");
    log::info!("  Scaling: {:.4}", lora_config.scaling());

    // Wrap for SFT training
    let sft_model = Gemma2ForSFT::new(lora_model, pad_token_id);

    // -----------------------------------------------------------------------
    // 8. Setup Training
    // -----------------------------------------------------------------------
    log::info!("[7/8] Setting up training");
    log::info!("  Epochs: {}", args.epochs);
    log::info!("  Batch size: {}", args.batch_size);
    log::info!("  Learning rate: {}", args.learning_rate);
    log::info!("  Optimizer: Adam");

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)
        .unwrap_or_else(|e| panic!("Failed to create output dir '{}': {e}", args.output_dir));

    // Seed the backend
    B::seed(&device, args.seed);

    let optimizer = AdamConfig::new().init();

    let training = SupervisedTraining::new(&args.output_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .num_epochs(args.epochs)
        .with_file_checkpointer(CompactRecorder::new())
        .summary();

    // -----------------------------------------------------------------------
    // 9. Run Training
    // -----------------------------------------------------------------------
    log::info!("[8/8] Starting training...\n");

    let learner = Learner::new(sft_model, optimizer, args.learning_rate);
    let result = training.launch(learner);

    log::info!("\n=== Training Complete ===\n");

    // -----------------------------------------------------------------------
    // 10. Save LoRA Adapters
    // -----------------------------------------------------------------------
    let adapter_dir = format!("{}/adapters", args.output_dir);
    log::info!("Saving LoRA adapters to: {adapter_dir}");

    match result.model.model.save_adapters(&adapter_dir) {
        Ok(()) => {
            // Count adapter files
            let adapter_files: Vec<_> = std::fs::read_dir(&adapter_dir)
                .unwrap_or_else(|e| panic!("Failed to read adapter dir: {e}"))
                .flatten()
                .collect();
            log::info!("  Saved adapters for {} layers", adapter_files.len());
        }
        Err(e) => {
            log::error!("Failed to save adapters: {e}");
        }
    }

    // -----------------------------------------------------------------------
    // 11. Summary
    // -----------------------------------------------------------------------
    log::info!("\n=== Summary ===");
    log::info!("  Output directory: {}", args.output_dir);
    log::info!("  Adapter directory: {adapter_dir}");
    log::info!("  LoRA rank: {}", args.lora_rank);
    log::info!("  LoRA alpha: {}", args.lora_alpha);
    log::info!("  Trainable params: {lora_params} ({pct:.2}%)");
    log::info!("  Total params: {total_params}");
    log::info!("  Epochs: {}", args.epochs);

    log::info!("\nDone!");
}

// ---------------------------------------------------------------------------
// Main: Select Backend
// ---------------------------------------------------------------------------

#[cfg(feature = "ndarray")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn::backend::Autodiff;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;
    type AD = Autodiff<Backend>;

    let args = SftArgs::parse();
    let device = Default::default();
    run::<AD>(args, device);
}

#[cfg(feature = "wgpu")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn::backend::{Autodiff, wgpu::Wgpu};

    type Backend = Wgpu<f32, i64>;
    type AD = Autodiff<Backend>;

    let args = SftArgs::parse();
    let device = Default::default();
    run::<AD>(args, device);
}

#[cfg(feature = "metal")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn::backend::{Autodiff, metal::Metal};

    type Backend = Metal<f32, i64>;
    type AD = Autodiff<Backend>;

    let args = SftArgs::parse();
    let device = Default::default();
    run::<AD>(args, device);
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
