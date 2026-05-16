#![cfg_attr(
    not(any(feature = "ndarray", feature = "wgpu", feature = "metal")),
    allow(dead_code)
)]
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
//! | `--quantize` | `none` | Quantize frozen weights: `none`, `q4s`, `q8s` |
//! | `--grad-accum` | `8` | Gradient accumulation steps (effective batch = batch × accum) |
//! | `--max-grad-norm` | `1.0` | Max gradient norm for clipping (0 = disabled) |
//! | `--weight-decay` | `0.01` | Adam weight decay (L2 regularization) |
//! | `--max-duration` | (none) | Max training duration in seconds (unlimited if not set) |
//! | `--max-ram` | (none) | Max RAM usage in GB, stops training if exceeded (macOS only) |
//! | `--warmup-steps` | `0` | Linear LR warmup steps before cosine decay (0 = no warmup) |
//! | `--use-fused-ce` | `false` | Use fused CE kernel (saves ~4GB memory, slower than standard CE) |
//! | `--no-mixed-precision` | `false` | Disable f32 optimizer states (keeps moments in model dtype) |

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use burn::data::dataloader::DataLoaderBuilder;
use burn::module::{AutodiffModule, Module, Quantizer};
use burn::nn::lora::{LoraBias, LoraConfig};
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::lr_scheduler::composed::{ComposedLrSchedulerConfig, SchedulerReduction};
use burn::optim::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::optim::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::record::CompactRecorder;
use burn::tensor::Element;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::quantization::{
    BlockSize, Calibration, QuantLevel, QuantMode, QuantScheme, QuantValue,
};
use burn::train::metric::LossMetric;
use burn::train::{Learner, SupervisedTraining};

use lora_gemma2::batcher::SFTBatcher;
use lora_gemma2::dataset::JsonlDataset;
use lora_gemma2::loader::load_gemma2_weights_dtype;
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
    /// Quantization scheme for frozen weights: none, q4s, q8s (default: none).
    quantize: String,
    /// Gradient accumulation steps (effective batch = batch_size × grad_accum).
    grad_accum: usize,
    /// Max gradient norm for clipping (0 = disabled).
    max_grad_norm: f32,
    /// Adam weight decay (L2 regularization).
    weight_decay: f32,
    /// Max training duration in seconds (None = unlimited).
    max_duration: Option<u64>,
    /// Max RAM usage in GB (None = unlimited, macOS only).
    max_ram: Option<f64>,
    /// Linear LR warmup steps before cosine decay (0 = no warmup).
    warmup_steps: usize,
    /// Use fused CE kernel with inline softcapping instead of standard CE.
    /// Default: false (standard CE is faster; fused CE saves ~4GB memory but is slower).
    use_fused_ce: bool,
    /// Disable mixed precision optimizer (keep moments in f32 regardless of model dtype).
    no_mixed_precision: bool,
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
            epochs: 1,
            batch_size: 1,
            learning_rate: 2e-4,
            max_seq_length: 512,
            val_split: 0.1,
            seed: 42,
            tokenizer: None,
            quantize: "none".into(),
            grad_accum: 4,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            max_duration: None,
            max_ram: None,
            warmup_steps: 0,
            use_fused_ce: false,
            no_mixed_precision: false,
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
                "--quantize" => {
                    args.quantize = cli.get(i + 1).expect("--quantize requires a value").clone();
                    i += 2;
                }
                "--grad-accum" => {
                    args.grad_accum = cli
                        .get(i + 1)
                        .expect("--grad-accum requires a value")
                        .parse()
                        .expect("invalid grad-accum");
                    i += 2;
                }
                "--max-grad-norm" => {
                    args.max_grad_norm = cli
                        .get(i + 1)
                        .expect("--max-grad-norm requires a value")
                        .parse()
                        .expect("invalid max-grad-norm");
                    i += 2;
                }
                "--weight-decay" => {
                    args.weight_decay = cli
                        .get(i + 1)
                        .expect("--weight-decay requires a value")
                        .parse()
                        .expect("invalid weight-decay");
                    i += 2;
                }
                "--max-duration" => {
                    args.max_duration = Some(
                        cli.get(i + 1)
                            .expect("--max-duration requires a value")
                            .parse()
                            .expect("invalid max-duration (seconds)"),
                    );
                    i += 2;
                }
                "--max-ram" => {
                    args.max_ram = Some(
                        cli.get(i + 1)
                            .expect("--max-ram requires a value")
                            .parse()
                            .expect("invalid max-ram (GB)"),
                    );
                    i += 2;
                }
                "--warmup-steps" => {
                    args.warmup_steps = cli
                        .get(i + 1)
                        .expect("--warmup-steps requires a value")
                        .parse()
                        .expect("invalid warmup-steps");
                    i += 2;
                }
                "--no-mixed-precision" => {
                    args.no_mixed_precision = true;
                }
                "--use-fused-ce" => {
                    args.use_fused_ce = true;
                    i += 1;
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

/// Valid quantization schemes for the `--quantize` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QuantizeScheme {
    /// No quantization (full precision).
    None,
    /// 4-bit symmetric quantization.
    Q4S,
    /// 8-bit symmetric quantization.
    Q8S,
    /// 8-bit affine quantization (per-tensor scale + bias, native i8 storage).
    Q8fAffine,
    /// 4-bit affine quantization with per-tensor scale + bias (packed u32 storage).
    Q4fAffineTensor,
    /// 4-bit affine quantization with per-block scales+biases.
    Q4fAffine,
}

impl QuantizeScheme {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(Self::None),
            "q4s" => Some(Self::Q4S),
            "q8s" => Some(Self::Q8S),
            "q8f_affine" => Some(Self::Q8fAffine),
            "q4f_affine_tensor" => Some(Self::Q4fAffineTensor),
            "q4f_affine" => Some(Self::Q4fAffine),
            _ => None,
        }
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
    eprintln!("  --epochs <N>             Training epochs (default: 1)");
    eprintln!("  --batch-size <N>         Batch size (default: 1)");
    eprintln!("  --lr <F>                 Learning rate (default: 2e-4)");
    eprintln!("  --max-seq-length <N>     Max sequence length (default: 512)");
    eprintln!("  --val-split <F>          Validation split ratio (default: 0.1)");
    eprintln!("  --seed <N>               Random seed (default: 42)");
    eprintln!(
        "  --quantize <SCHEME>      Quantize frozen weights: none, q4s, q8s, q8f_affine, q4f_affine_tensor, q4f_affine (default: none)"
    );
    eprintln!("  --grad-accum <N>         Gradient accumulation steps (default: 4)");
    eprintln!("  --max-grad-norm <F>      Max gradient norm for clipping, 0=off (default: 1.0)");
    eprintln!("  --weight-decay <F>       Adam weight decay / L2 reg (default: 0.01)");
    eprintln!("  --max-duration <S>       Max training duration in seconds (default: unlimited)");
    eprintln!(
        "  --max-ram <GB>           Max RAM usage in GB, stops if exceeded (default: unlimited)"
    );
    eprintln!("  --warmup-steps <N>       Linear LR warmup steps (default: 0, no warmup)");
    eprintln!("  --use-fused-ce           Use fused CE kernel (saves ~4GB memory, default: false)");
    eprintln!(
        "  --no-mixed-precision     Disable f32 optimizer states (default: false, mixed precision on)"
    );
}

// ---------------------------------------------------------------------------
// Training Pipeline
// ---------------------------------------------------------------------------

/// Backend trait bound for SFT training.
///
/// For cubecl backends (metal, wgpu, cuda, vulkan, rocm): includes fused CE kernel support.
/// For other backends (ndarray, tch): standard autodiff backend only.
#[cfg(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
))]
trait SftBackend:
    AutodiffBackend
    + lora_gemma2::fused_ops::FusedCEBackend
    + lora_gemma2::fused_ops::FusedLoraMLPBackend
{
}
#[cfg(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
))]
impl<T> SftBackend for T where
    T: AutodiffBackend
        + lora_gemma2::fused_ops::FusedCEBackend
        + lora_gemma2::fused_ops::FusedLoraMLPBackend
{
}

#[cfg(not(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
)))]
trait SftBackend: AutodiffBackend {}
#[cfg(not(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
)))]
impl<T: AutodiffBackend> SftBackend for T {}

/// Get current process RSS (Resident Set Size) in GB.
///
/// Uses `ps` command (macOS/Linux compatible). Returns `None` if unavailable.
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
    Some(rss_kb / 1024.0 / 1024.0) // KB -> GB
}

/// Run the SFT training pipeline.
///
/// Generic over any autodiff backend (NdArray, Wgpu, Metal, etc.)
fn run<B: SftBackend>(args: SftArgs, device: B::Device) {
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
    let train_samples = train_dataset.len();
    log::info!(
        "  Split: {} train, {} val ({:.0}% val split)",
        train_samples,
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

    // Create on inner backend (no autodiff) so loaded weights become leaf tensors
    let mut inner_model = Gemma2Model::<B::InnerBackend>::new(&config, &device);

    // -----------------------------------------------------------------------
    // 6. Load Pretrained Weights (if provided)
    // -----------------------------------------------------------------------
    match &args.weights {
        Some(weights_path) => {
            log::info!("[5/8] Loading weights from '{weights_path}'");
            match load_gemma2_weights_dtype(
                &mut inner_model,
                PathBuf::from(weights_path).as_path(),
                &device,
                <B::FloatElem as Element>::dtype(),
            ) {
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
    // 6b. Quantize Frozen Weights (optional)
    // -----------------------------------------------------------------------
    let quant_scheme = QuantizeScheme::from_str(&args.quantize).unwrap_or_else(|| {
        eprintln!(
            "Error: unknown quantize scheme '{}'. Use: none, q4s, q8s, q8f_affine, q4f_affine_tensor, q4f_affine",
            args.quantize
        );
        std::process::exit(1);
    });

    let inner_model = match quant_scheme {
        QuantizeScheme::None => inner_model,
        QuantizeScheme::Q4S => {
            log::info!("[5b/8] Quantizing frozen weights to Q4S (4-bit symmetric)");
            let scheme = QuantScheme::default()
                .with_value(QuantValue::Q4S)
                .with_level(QuantLevel::Tensor);
            let mut quantizer = Quantizer {
                calibration: Calibration::MinMax,
                scheme,
            };
            let quantized = inner_model.quantize_weights(&mut quantizer);
            log::info!("  Quantization complete");
            quantized
        }
        QuantizeScheme::Q8S => {
            log::info!("[5b/8] Quantizing frozen weights to Q8S (8-bit symmetric)");
            let scheme = QuantScheme::default()
                .with_value(QuantValue::Q8S)
                .with_level(QuantLevel::Tensor);
            let mut quantizer = Quantizer {
                calibration: Calibration::MinMax,
                scheme,
            };
            let quantized = inner_model.quantize_weights(&mut quantizer);
            log::info!("  Quantization complete");
            quantized
        }
        QuantizeScheme::Q8fAffine => {
            log::info!("[5b/8] Quantizing frozen weights to Q8F affine (8-bit affine, per-tensor)");
            let scheme = QuantScheme::default()
                .with_value(QuantValue::Q8F)
                .with_mode(QuantMode::Affine)
                .with_level(QuantLevel::Tensor);
            let mut quantizer = Quantizer {
                calibration: Calibration::MinMax,
                scheme,
            };
            let quantized = inner_model.quantize_weights(&mut quantizer);
            log::info!("  Quantization complete");
            quantized
        }
        QuantizeScheme::Q4fAffineTensor => {
            log::info!(
                "[5b/8] Quantizing frozen weights to Q4F affine tensor (4-bit affine, per-tensor, packed u32)"
            );
            let scheme = QuantScheme::default()
                .with_value(QuantValue::Q4F)
                .with_mode(QuantMode::Affine)
                .with_level(QuantLevel::Tensor);
            let mut quantizer = Quantizer {
                calibration: Calibration::MinMax,
                scheme,
            };
            let quantized = inner_model.quantize_weights(&mut quantizer);
            log::info!("  Quantization complete");
            quantized
        }
        QuantizeScheme::Q4fAffine => {
            log::info!("[5b/8] Quantizing frozen weights to Q4F affine (4-bit affine, blk-64)");
            let scheme = QuantScheme::default()
                .with_value(QuantValue::Q4F)
                .with_mode(QuantMode::Affine)
                .with_level(QuantLevel::Block(BlockSize::new([64])));
            let mut quantizer = Quantizer {
                calibration: Calibration::MinMax,
                scheme,
            };
            let quantized = inner_model.quantize_weights(&mut quantizer);
            log::info!("  Quantization complete");
            quantized
        }
    };

    // Convert to autodiff backend — from_inner creates proper leaf tensors
    // that can be tracked for gradient computation during LoRA training
    let model = Gemma2Model::<B>::from_inner(inner_model);

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
    let sft_model = Gemma2ForSFT::new(lora_model, pad_token_id, args.use_fused_ce);

    // -----------------------------------------------------------------------
    // 8. Setup Training
    // -----------------------------------------------------------------------
    log::info!("[7/8] Setting up training");
    log::info!("  Epochs: {}", args.epochs);
    log::info!("  Batch size: {}", args.batch_size);

    // Compute total iterations for cosine LR schedule (train_samples saved before dataloader move)
    let steps_per_epoch = train_samples.div_ceil(args.batch_size);
    let total_steps = steps_per_epoch * args.epochs;
    let effective_lr = args.learning_rate.min(1.0);

    log::info!(
        "  Learning rate: {:.6} (cosine schedule, {total_steps} steps)",
        effective_lr
    );
    log::info!("  Optimizer: Adam (weight_decay={})", args.weight_decay);
    log::info!(
        "  Gradient accumulation: {} (effective batch={})",
        args.grad_accum,
        args.batch_size * args.grad_accum
    );
    log::info!("  Gradient clipping: max_norm={}", args.max_grad_norm);

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)
        .unwrap_or_else(|e| panic!("Failed to create output dir '{}': {e}", args.output_dir));

    // Seed the backend
    B::seed(&device, args.seed);

    let mut adam_config = AdamConfig::new().with_mixed_precision(!args.no_mixed_precision);

    // Wire weight decay (L2 regularization)
    if args.weight_decay > 0.0 {
        adam_config =
            adam_config.with_weight_decay(Some(WeightDecayConfig::new(args.weight_decay)));
    }

    // Wire gradient clipping (prevents NaN with rank=16)
    if args.max_grad_norm > 0.0 {
        adam_config =
            adam_config.with_grad_clipping(Some(GradientClippingConfig::Norm(args.max_grad_norm)));
    }

    let optimizer = adam_config.init();

    let mut training =
        SupervisedTraining::new(&args.output_dir, dataloader_train, dataloader_valid)
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .num_epochs(args.epochs)
            .with_file_checkpointer(CompactRecorder::new());

    // Wire gradient accumulation (effective batch = batch_size × grad_accum)
    if args.grad_accum > 1 {
        training = training.grads_accumulation(args.grad_accum);
    }

    let training = training.summary();

    // -----------------------------------------------------------------------
    // 8b. Resource Limits (Duration + RAM) + Peak Memory Tracking
    // -----------------------------------------------------------------------
    let max_duration = args.max_duration;
    let max_ram = args.max_ram;
    let peak_rss_kb: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));

    {
        let interrupter = training.interrupter();
        let peak_rss_kb = peak_rss_kb.clone();

        if max_duration.is_some() || max_ram.is_some() {
            log::info!("  Resource limits:");
            if let Some(secs) = max_duration {
                log::info!("    Duration: {secs}s ({:.1}min)", secs as f64 / 60.0);
            }
            if let Some(gb) = max_ram {
                log::info!("    RAM: {gb:.1} GB");
            }
        }

        // Spawn monitoring thread — tracks peak RSS + checks limits every 5s
        // Thread exits when a limit is hit; otherwise runs until process exits.
        std::thread::spawn(move || {
            let start = Instant::now();
            let interval = std::time::Duration::from_secs(5);
            loop {
                std::thread::sleep(interval);

                // Track peak RSS (always, regardless of limits)
                if let Some(rss_gb) = get_process_rss_gb() {
                    let rss_kb = (rss_gb * 1024.0 * 1024.0) as u64;
                    peak_rss_kb.fetch_max(rss_kb, Ordering::Relaxed);
                }

                if let Some(max_secs) = max_duration {
                    let elapsed = start.elapsed().as_secs();
                    if elapsed >= max_secs {
                        let reason = format!(
                            "Duration limit: {elapsed}s >= {max_secs}s ({:.1}min)",
                            max_secs as f64 / 60.0
                        );
                        log::warn!("⚠ {reason}");
                        interrupter.stop(Some(&reason));
                        return;
                    }
                }

                if let Some(max_gb) = max_ram
                    && let Some(rss_gb) = get_process_rss_gb()
                    && rss_gb > max_gb
                {
                    let reason = format!("RAM limit: {rss_gb:.1} GB > {max_gb:.1} GB");
                    log::warn!("⚠ {reason}");
                    interrupter.stop(Some(&reason));
                    return;
                }
            }
        });
    }

    // -----------------------------------------------------------------------
    // 9. Run Training
    // -----------------------------------------------------------------------
    log::info!("[8/8] Starting training...");
    if let Some(rss_gb) = get_process_rss_gb() {
        log::info!("  RSS after model load: {rss_gb:.1} GB");
    }
    log::info!("\n");

    // LR schedule: optional linear warmup + cosine annealing decay
    let warmup_steps = args.warmup_steps;
    let cosine_config = CosineAnnealingLrSchedulerConfig::new(effective_lr, total_steps)
        .with_min_lr(effective_lr * 0.1);

    let lr_scheduler = if warmup_steps > 0 {
        // Warmup: linear 0.01 → 1.0 over warmup_steps, then stays at 1.0
        // Combined with cosine via Prod: warmup_mult × cosine_lr
        // Step 0: 0.01 × lr ≈ 0, Step warmup: 1.0 × lr, then cosine decay
        let warmup_config = LinearLrSchedulerConfig::new(0.01, 1.0, warmup_steps);

        ComposedLrSchedulerConfig::new()
            .linear(warmup_config)
            .cosine(cosine_config)
            .with_reduction(SchedulerReduction::Prod)
            .init()
            .expect("Invalid composed LR scheduler config")
    } else {
        // No warmup — pure cosine decay
        ComposedLrSchedulerConfig::new()
            .cosine(cosine_config)
            .with_reduction(SchedulerReduction::Prod)
            .init()
            .expect("Invalid LR scheduler config")
    };

    if warmup_steps > 0 {
        log::info!(
            "  LR schedule: linear warmup ({warmup_steps} steps) + cosine decay (initial={effective_lr:.6}, min={:.6}, steps={total_steps})",
            effective_lr * 0.1,
        );
    } else {
        log::info!(
            "  LR schedule: cosine (initial={effective_lr:.6}, min={:.6}, steps={total_steps})",
            effective_lr * 0.1,
        );
    }
    log::info!(
        "  Total iterations: {total_steps} ({steps_per_epoch}/epoch × {} epochs)",
        args.epochs
    );

    let learner = Learner::new(sft_model, optimizer, lr_scheduler);
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

    // Peak RSS from monitoring thread
    let peak_kb = peak_rss_kb.load(Ordering::Relaxed);
    if peak_kb > 0 {
        let peak_gb = peak_kb as f64 / 1024.0 / 1024.0;
        log::info!("  Peak RSS: {peak_gb:.1} GB");
    } else {
        // Monitoring thread may not have sampled yet (short run)
        if let Some(rss_gb) = get_process_rss_gb() {
            log::info!("  RSS: {rss_gb:.1} GB");
        }
    }

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

    use burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing;
    use burn::backend::{Autodiff, Metal};
    use burn::tensor::f16;

    // Use f16 on Metal to halve memory and maximize Metal Performance Shaders throughput.
    //
    // NOTE: f16 training has a known instability — loss spikes at ~iter 9 (14-16 range)
    // before slowly recovering. Root cause: burn does NOT do mixed precision like MLX does.
    // MLX internally upcasts to f32 for critical ops (RMSNorm variance, softmax, grad accum)
    // while keeping weights in f16. Burn's metal backend uses pure f16 everywhere.
    //
    // This is a burn upstream limitation, not a fixable issue in this project.
    // See Plan 014 benchmarks for details: metal/f16 = ~3-4s/iter, metal/f32 = ~22s/iter.
    //
    // If training quality matters more than speed, switch to f32:
    //   type Backend = Metal<f32, i64>;  // ~22s/iter, stable loss
    type Backend = Metal<f16, i64>; // ~3-4s/iter, loss spikes but 5x faster
    // BalancedCheckpointing: memory-bound ops recompute during backward instead of
    // saving all intermediates. Reduces peak memory ~30-50% at cost of ~10% more compute.
    // Default (NoCheckpointing) stores every intermediate tensor — ~2x peak memory.
    type AD = Autodiff<Backend, BalancedCheckpointing>;

    let args = SftArgs::parse();
    let device = Default::default();
    run::<AD>(args, device);
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
