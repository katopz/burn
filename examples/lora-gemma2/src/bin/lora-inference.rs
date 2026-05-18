#![cfg_attr(
    not(any(feature = "ndarray", feature = "wgpu", feature = "metal")),
    allow(dead_code)
)]
//! LoRA Inference binary — autoregressive text generation with optional LoRA adapter.
//!
//! Loads a Gemma 2 model from HuggingFace safetensors, optionally applies a LoRA
//! adapter (MPK format from sft-train), and generates text using greedy decoding.
//!
//! **No merge needed** — LoRA is applied on-the-fly during forward pass:
//! `output = W_base @ x + (B @ A) @ x * scaling`
//!
//! This allows hot-swapping adapters without reloading the base model.
//!
//! # Usage
//!
//! ```sh
//! # Base model only (no adapter)
//! cargo run -p lora-gemma2 --features metal --bin lora-inference -- \
//!   --model /path/to/gemma-2-2b-it \
//!   --test 1
//!
//! # With LoRA adapter (on-the-fly, no merge)
//! cargo run -p lora-gemma2 --features metal --bin lora-inference -- \
//!   --model /path/to/gemma-2-2b-it \
//!   --adapter /path/to/output/adapters \
//!   --test 1
//!
//! # Custom prompt
//! cargo run -p lora-gemma2 --features metal --bin lora-inference -- \
//!   --model /path/to/gemma-2-2b-it \
//!   --adapter /path/to/output/adapters \
//!   --prompt "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)" \
//!   --max-tokens 128
//! ```

use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::{Element, Int, Tensor, TensorData};
use lora_gemma2::loader::load_gemma2_weights_dtype;
use lora_gemma2::model_lora::{apply_lora_to_gemma2, count_lora_params, count_total_params};
use lora_gemma2::tokenizer::{ChatMessage, GemmaTokenizer, Role};
use lora_gemma2::types::LoraTarget;
use lora_gemma2::{Gemma2Config, Gemma2Model, Gemma2ModelLora, LoadReport};

// ---------------------------------------------------------------------------
// Inference model wrapper — base or LoRA, no merge needed
// ---------------------------------------------------------------------------

/// Wrapper for either base model or LoRA-adapted model.
///
/// LoRA is applied **on-the-fly** during forward pass — no merge required.
/// This allows swapping adapters without reloading the base model.
enum InferenceModel<B: burn::tensor::backend::Backend> {
    Base(Gemma2Model<B>),
    Lora(Gemma2ModelLora<B>),
}

impl<B: burn::tensor::backend::Backend> InferenceModel<B> {
    /// Forward pass — dispatches to base or LoRA model.
    /// LoRA path: `output = W_base @ x + (B @ A) @ x * scaling`
    fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        match self {
            InferenceModel::Base(m) => m.forward(input_ids),
            InferenceModel::Lora(m) => m.forward(input_ids),
        }
    }

    fn is_lora(&self) -> bool {
        matches!(self, InferenceModel::Lora(_))
    }
}

// ---------------------------------------------------------------------------
// Built-in test prompts (py→rs translation)
// ---------------------------------------------------------------------------

/// Python→Rust test prompts from the training corpus domain.
struct TestPrompt {
    index: usize,
    python: &'static str,
    description: &'static str,
}

const TEST_PROMPTS: &[TestPrompt] = &[
    TestPrompt {
        index: 1,
        python: "def add(a, b): return a + b",
        description: "Simple function with two args",
    },
    TestPrompt {
        index: 2,
        python: "def greet(name): print(f\"Hello, {name}!\")",
        description: "Function with f-string print",
    },
    TestPrompt {
        index: 3,
        python: "for i in range(10): print(i)",
        description: "For loop with range",
    },
    TestPrompt {
        index: 4,
        python: "data = [x * 2 for x in items]",
        description: "List comprehension",
    },
    TestPrompt {
        index: 5,
        python: "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y",
        description: "Simple class with constructor",
    },
];

fn format_prompt_text(python: &str) -> String {
    format!(
        "Translate this Python code to idiomatic Rust:\n\n```python\n{}\n```",
        python
    )
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    model_path: PathBuf,
    adapter_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    prompt: Option<String>,
    test_index: Option<usize>,
    max_tokens: usize,
    lora_rank: usize,
    lora_alpha: f64,
    model_name: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = None;
    let mut adapter_path = None;
    let mut tokenizer_path = None;
    let mut prompt = None;
    let mut test_index = None;
    let mut max_tokens = 256usize;
    let mut lora_rank = 16usize;
    let mut lora_alpha = 32.0f64;
    let mut model_name = "google/gemma-2-2b-it".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(PathBuf::from(&args[i]));
            }
            "--adapter" => {
                i += 1;
                adapter_path = Some(PathBuf::from(&args[i]));
            }
            "--tokenizer" => {
                i += 1;
                tokenizer_path = Some(PathBuf::from(&args[i]));
            }
            "--prompt" => {
                i += 1;
                prompt = Some(args[i].clone());
            }
            "--test" => {
                i += 1;
                test_index = Some(args[i].parse().expect("test index must be a number"));
            }
            "--max-tokens" => {
                i += 1;
                max_tokens = args[i].parse().expect("max-tokens must be a number");
            }
            "--rank" => {
                i += 1;
                lora_rank = args[i].parse().expect("rank must be a number");
            }
            "--alpha" => {
                i += 1;
                lora_alpha = args[i].parse().expect("alpha must be a number");
            }
            "--model-name" => {
                i += 1;
                model_name = args[i].clone();
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                // First positional arg = model path
                if model_path.is_none() && !other.starts_with('-') {
                    model_path = Some(PathBuf::from(other));
                } else {
                    eprintln!("Unknown argument: {other}");
                    print_usage();
                    std::process::exit(1);
                }
            }
        }
        i += 1;
    }

    let model_path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("Error: --model <path> is required");
            print_usage();
            std::process::exit(1);
        }
    };

    // Validate test index
    if let Some(idx) = test_index
        && !(1..=TEST_PROMPTS.len()).contains(&idx)
    {
        eprintln!(
            "Error: --test index must be 1-{}, got {idx}",
            TEST_PROMPTS.len()
        );
        std::process::exit(1);
    }

    Args {
        model_path,
        adapter_path,
        tokenizer_path,
        prompt,
        test_index,
        max_tokens,
        lora_rank,
        lora_alpha,
        model_name,
    }
}

fn print_usage() {
    eprintln!("Usage: lora-inference [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --model <PATH>         Path to base model (safetensors dir or file) [required]");
    eprintln!("  --adapter <PATH>       Path to LoRA adapter directory (MPK format)");
    eprintln!("  --tokenizer <PATH>     Path to tokenizer.json [auto-detect from --model]");
    eprintln!("  --prompt <TEXT>        Input prompt for generation");
    eprintln!("  --test <INDEX>         Run built-in test prompt (1-5)");
    eprintln!("  --max-tokens <N>       Max tokens to generate [default: 256]");
    eprintln!("  --rank <N>             LoRA rank [default: 16]");
    eprintln!("  --alpha <F>            LoRA alpha [default: 32.0]");
    eprintln!(
        "  --model-name <NAME>    Model name for tokenizer download [default: google/gemma-2-2b-it]"
    );
    eprintln!("  --help                 Show this help");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  lora-inference --model /path/to/gemma-2-2b-it --test 1");
    eprintln!("  lora-inference --model /path/to/gemma-2-2b-it --adapter output/adapters --test 1");
    eprintln!("  lora-inference --model /path/to/gemma-2-2b-it --adapter output/adapters \\");
    eprintln!("    --prompt \"def add(a, b): return a + b\" --max-tokens 128");
    eprintln!();
    eprintln!("Note: LoRA is applied on-the-fly (no merge). You can swap adapters");
    eprintln!("without reloading the base model.");
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

/// Run autoregressive generation with greedy decoding.
fn generate<B: burn::tensor::backend::Backend>(
    model: &InferenceModel<B>,
    tokenizer: &GemmaTokenizer,
    prompt: &str,
    max_tokens: usize,
    device: &B::Device,
) -> String {
    // 1. Encode prompt with chat template
    let messages = vec![ChatMessage::new(Role::User, prompt.to_string())];
    let tokens = tokenizer.encode_chat(&messages, true);
    log::info!("Prompt encoded: {} tokens", tokens.len());

    // 2. Create input tensor: [1, seq_len]
    let tokens_i64: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
    let input_ids =
        Tensor::<B, 2, Int>::from_data(TensorData::new(tokens_i64, [1, tokens.len()]), device);

    let mut generated_tokens: Vec<usize> = tokens.clone();
    let eos_id = tokenizer.eos_token_id();
    let start = Instant::now();

    // 3. Autoregressive loop
    let mut current_input = input_ids;
    for step in 0..max_tokens {
        let logits = model.forward(current_input.clone());

        // Greedy: argmax at last position
        // logits: [1, seq, vocab] → last pos → [1, vocab] → argmax → [1]
        let [_batch, seq, _vocab] = logits.dims();
        let last_logits = logits.slice([0..1, (seq - 1)..seq]).squeeze_dim::<2>(1);
        let next_token_tensor = last_logits.argmax(1); // [1]

        // Read token value
        let next_token_id: usize = next_token_tensor
            .clone()
            .into_data()
            .as_slice::<i64>()
            .unwrap()[0] as usize;

        // Check EOS
        if next_token_id == eos_id {
            log::info!("EOS token at step {step}");
            break;
        }

        generated_tokens.push(next_token_id);

        // Append for next iteration: [1] → [1, 1]
        let next_2d = next_token_tensor.unsqueeze_dim::<2>(1);
        current_input = Tensor::cat(vec![current_input, next_2d], 1);

        // Progress log every 50 tokens
        if (step + 1) % 50 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let tok_per_sec = (step + 1) as f64 / elapsed;
            log::info!(
                "  Step {step}: {} tokens generated ({tok_per_sec:.1} tok/s)",
                step + 1
            );
        }
    }

    let elapsed = start.elapsed();
    let total_generated = generated_tokens.len() - tokens.len();
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 {
        total_generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    log::info!(
        "Generation complete: {total_generated} tokens in {:.1}s ({tok_per_sec:.1} tok/s)",
        elapsed.as_secs_f64()
    );

    // Decode only the generated portion (skip prompt)
    let generated_only = &generated_tokens[tokens.len()..];
    tokenizer.decode(generated_only, true)
}

/// Print model and adapter summary.
fn print_summary<B: burn::tensor::backend::Backend>(
    model: &InferenceModel<B>,
    adapter_path: Option<&PathBuf>,
    config: &Gemma2Config,
) {
    let adapter_label = if model.is_lora() {
        "YES (on-the-fly, no merge)"
    } else {
        "NO (base model only)"
    };
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Gemma 2 LoRA Inference                                      │");
    println!("├─────────────────────────────────────────────────────────────┤");
    println!("│ Model:       Gemma 2 2B (vocab={})", config.vocab_size);
    println!("│ Layers:      {}", config.num_hidden_layers);
    println!("│ Hidden:      {}", config.hidden_size);
    println!(
        "│ Heads:       {} (KV: {})",
        config.num_attention_heads, config.num_key_value_heads
    );
    println!("│ Adapter:     {adapter_label}");
    if let Some(path) = adapter_path {
        println!("│ Adapter dir: {}", path.display());
    }
    println!("└─────────────────────────────────────────────────────────────┘");
}

/// Print load report.
fn print_load_report(report: &LoadReport) {
    log::info!("Load report:");
    log::info!("  Tensors loaded: {}", report.tensors_loaded);
    log::info!("  Files read: {}", report.files_read.len());
    for file in &report.files_read {
        log::info!("    - {file}");
    }
    if !report.tensors_skipped.is_empty() {
        log::info!(
            "  Tensors skipped ({}): {:?}",
            report.tensors_skipped.len(),
            report.tensors_skipped
        );
    }
}

/// Run inference with a specific backend.
fn run_inference<B: burn::tensor::backend::Backend>(args: Args, device: B::Device) {
    let config = Gemma2Config::gemma2_2b();

    // -----------------------------------------------------------------------
    // 1. Load tokenizer
    // -----------------------------------------------------------------------
    log::info!("[1/4] Loading tokenizer");
    let tokenizer = match &args.tokenizer_path {
        Some(path) => {
            log::info!("  From file: {}", path.display());
            GemmaTokenizer::from_file(path).expect("Failed to load tokenizer")
        }
        None => {
            log::info!("  From pretrained: {}", args.model_name);
            GemmaTokenizer::from_pretrained(&args.model_name).expect("Failed to load tokenizer")
        }
    };
    log::info!(
        "  Vocab size: {}, BOS: {}, EOS: {}, PAD: {}",
        tokenizer.vocab_size(),
        tokenizer.bos_token_id(),
        tokenizer.eos_token_id(),
        tokenizer.pad_token_id()
    );

    // -----------------------------------------------------------------------
    // 2. Load base model
    // -----------------------------------------------------------------------
    log::info!(
        "[2/4] Loading base model from: {}",
        args.model_path.display()
    );
    let mut model = Gemma2Model::<B>::new(&config, &device);
    let load_start = Instant::now();

    let report = match load_gemma2_weights_dtype(
        &mut model,
        &args.model_path,
        &device,
        <B::FloatElem as Element>::dtype(),
    ) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("Error loading weights: {e}");
            std::process::exit(1);
        }
    };

    let load_time = load_start.elapsed();
    print_load_report(&report);
    log::info!("  Weights loaded in {:.1}s", load_time.as_secs_f64());

    // -----------------------------------------------------------------------
    // 3. Apply LoRA adapter (optional, on-the-fly — no merge)
    // -----------------------------------------------------------------------
    let adapter_path = args.adapter_path.clone();
    let model = if let Some(ref adapter_dir) = args.adapter_path {
        log::info!(
            "[3/4] Applying LoRA adapter (on-the-fly): {}",
            adapter_dir.display()
        );
        let lora_start = Instant::now();

        let lora_config =
            burn::nn::lora::LoraConfig::new(args.lora_rank).with_alpha(args.lora_alpha);
        let targets = LoraTarget::all_targets().to_vec();

        // Wrap base model with LoRA layers (creates A/B matrices, initialized to zero)
        let lora_model = apply_lora_to_gemma2(model, &lora_config, &targets, &device);

        // Load trained adapter weights into A/B matrices
        let lora_model = match lora_model.load_adapters(adapter_dir, &device) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error loading LoRA adapter: {e}");
                eprintln!(
                    "  Expected: {}/layer_0/q_proj.mpk etc.",
                    adapter_dir.display()
                );
                std::process::exit(1);
            }
        };

        let n_lora = count_lora_params(&lora_model);
        let n_total = count_total_params(&lora_model);
        let lora_pct = (n_lora as f64 / n_total as f64) * 100.0;
        log::info!("  LoRA params: {n_lora} / {n_total} ({lora_pct:.2}%)");
        log::info!(
            "  Adapter loaded in {:.1}s",
            lora_start.elapsed().as_secs_f64()
        );
        log::info!(
            "  Using on-the-fly LoRA (no merge) — output = W_base @ x + (B @ A) @ x * scale"
        );

        InferenceModel::Lora(lora_model)
    } else {
        log::info!("[3/4] No adapter — using base model");
        InferenceModel::Base(model)
    };

    // -----------------------------------------------------------------------
    // 4. Generate
    // -----------------------------------------------------------------------
    print_summary(&model, adapter_path.as_ref(), &config);

    if let Some(idx) = args.test_index {
        // Run built-in test prompt
        let test = &TEST_PROMPTS[idx - 1];
        let prompt_text = format_prompt_text(test.python);

        println!();
        println!("━━━ Test Prompt #{}: {} ━━━", test.index, test.description);
        println!("Python input:");
        println!("  {}", test.python);
        println!();
        println!("Generating Rust translation...");
        println!("─────────────────────────────────────────────────────────────");

        let output = generate(&model, &tokenizer, &prompt_text, args.max_tokens, &device);

        println!("{output}");
        println!("─────────────────────────────────────────────────────────────");
    } else if let Some(prompt) = &args.prompt {
        println!();
        println!("━━━ Custom Prompt ━━━");
        println!("Input: {prompt}");
        println!();
        println!("Generating...");
        println!("─────────────────────────────────────────────────────────────");

        let output = generate(&model, &tokenizer, prompt, args.max_tokens, &device);

        println!("{output}");
        println!("─────────────────────────────────────────────────────────────");
    } else {
        // No prompt specified — run all test prompts
        println!();
        println!("━━━ Running all {} test prompts ━━━", TEST_PROMPTS.len());
        println!();

        for test in TEST_PROMPTS {
            let prompt_text = format_prompt_text(test.python);

            println!("─── Test #{}: {} ───", test.index, test.description);
            println!("Python: {}", test.python);

            let output = generate(&model, &tokenizer, &prompt_text, args.max_tokens, &device);

            println!("Output: {output}");
            println!();
        }
    }

    log::info!("Done!");
}

// ---------------------------------------------------------------------------
// Main: Select Backend
// ---------------------------------------------------------------------------

#[cfg(feature = "ndarray")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    let args = parse_args();
    let device = Default::default();
    run_inference::<B>(args, device);
}

#[cfg(feature = "wgpu")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn::backend::wgpu::Wgpu;

    type B = Wgpu<f32, i64>;

    let args = parse_args();
    let device = Default::default();
    run_inference::<B>(args, device);
}

#[cfg(feature = "metal")]
fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    use burn::backend::Metal;
    use burn::tensor::f16;

    // Use f16 on Metal to halve memory.
    // Metal does not support BF16, so BF16 weights are converted to F16 during loading.
    type B = Metal<f16, i64>;

    let args = parse_args();
    let device = Default::default();
    run_inference::<B>(args, device);
}

#[cfg(not(any(feature = "ndarray", feature = "wgpu", feature = "metal")))]
fn main() {
    eprintln!("Error: Enable a backend feature:");
    eprintln!("  --features ndarray    (CPU, for testing)");
    eprintln!("  --features wgpu       (WGPU/Metal)");
    eprintln!("  --features metal      (Metal native, recommended)");
    std::process::exit(1);
}
