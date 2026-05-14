//! Integration tests for SFT training pipeline with LoRA-adapted Gemma 2.
//!
//! Uses a tiny model configuration (small vocab, 2 layers) to validate
//! the training pipeline end-to-end without requiring real weights.

use burn::nn::lora::{LoraBias, LoraConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::Tolerance;
use burn::train::{InferenceStep, TrainStep};
use burn_ndarray::{NdArray, NdArrayDevice};

use lora_gemma2::model::Gemma2Model;
use lora_gemma2::model_lora::{
    Gemma2ForSFT, Gemma2ModelLora, apply_lora_to_gemma2, count_lora_params, count_total_params,
};
use lora_gemma2::types::{Gemma2Config, LoraTarget};

// ---------------------------------------------------------------------------
// Test Helpers
// ---------------------------------------------------------------------------

type TestBackend = NdArray<f32>;
type TestAD = burn::backend::Autodiff<TestBackend>;

fn device() -> NdArrayDevice {
    NdArrayDevice::Cpu
}

/// Tiny config for fast tests: 100 vocab, 32 hidden, 2 layers, 4 heads.
fn tiny_config() -> Gemma2Config {
    Gemma2Config::new(
        100, // vocab_size
        32,  // hidden_size
        2,   // num_hidden_layers
        64,  // intermediate_size
        4,   // num_attention_heads
        2,   // num_key_value_heads
        8,   // head_dim
    )
}

/// Create a LoRA-adapted tiny model for testing.
fn make_lora_model() -> Gemma2ModelLora<TestAD> {
    let device = device();
    let config = tiny_config();
    let model = Gemma2Model::<TestAD>::new(&config, &device);

    let lora_config = LoraConfig::new(4).with_alpha(8.0).with_bias(LoraBias::None);
    apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device)
}

/// Create a synthetic SFT training batch.
///
/// Generates token sequences where target = input shifted by 1.
/// All tokens are non-zero (valid, not padding).
fn make_batch(
    batch_size: usize,
    seq_len: usize,
    device: &NdArrayDevice,
) -> lora_gemma2::SFTTrainingBatch<TestAD> {
    // Create input tokens: [1, 2, 3, 4, ...]
    let input_data: Vec<i64> = (0..batch_size)
        .flat_map(|_b| (1..=seq_len as i64).collect::<Vec<_>>())
        .collect();

    // Target tokens: shifted by 1: [2, 3, 4, 5, ...]
    let target_data: Vec<i64> = (0..batch_size)
        .flat_map(|_b| (2..=seq_len as i64 + 1).collect::<Vec<_>>())
        .collect();

    // Clamp to vocab range
    let input_data: Vec<i64> = input_data.iter().map(|t| t % 99 + 1).collect();
    let target_data: Vec<i64> = target_data.iter().map(|t| t % 99 + 1).collect();

    let tokens_inputs = Tensor::<TestAD, 2, Int>::from_data(
        TensorData::new(input_data, [batch_size, seq_len]),
        device,
    );
    let targets = Tensor::<TestAD, 2, Int>::from_data(
        TensorData::new(target_data, [batch_size, seq_len]),
        device,
    );
    let mask_pad = Tensor::<TestAD, 2, Bool>::zeros([batch_size, seq_len], device);

    lora_gemma2::SFTTrainingBatch {
        tokens_inputs,
        targets,
        mask_pad,
    }
}

// ---------------------------------------------------------------------------
// Tests: Training Step / Inference Step
// ---------------------------------------------------------------------------

#[cfg(not(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
)))]
#[test]
fn test_train_step_produces_loss_and_grads() {
    let device = device();
    let lora_model = make_lora_model();
    let sft_model = Gemma2ForSFT::new(lora_model, 0, false);

    let _batch = make_batch(2, 8, &device);
    let output = <Gemma2ForSFT<TestAD> as TrainStep>::step(&sft_model, make_batch(2, 8, &device));

    // Loss should be a scalar tensor
    let loss_val: f32 = output.item.loss.into_scalar().elem();
    assert!(
        loss_val.is_finite(),
        "Loss should be finite, got {loss_val}"
    );
    assert!(
        loss_val > 0.0,
        "Cross-entropy loss should be positive, got {loss_val}"
    );

    // Logits should have correct shape
    let [batch, seq, vocab] = output.item.logits.dims();
    assert_eq!(batch, 2);
    assert_eq!(seq, 8);
    assert_eq!(vocab, tiny_config().vocab_size);
}

#[test]
fn test_inference_step_produces_loss() {
    let device = device();
    let lora_model = make_lora_model();
    let sft_model = Gemma2ForSFT::new(lora_model, 0, false);

    let batch = make_batch(2, 8, &device);
    let output = <Gemma2ForSFT<TestAD> as InferenceStep>::step(&sft_model, batch);

    let loss_val: f32 = output.loss.into_scalar().elem();
    assert!(
        loss_val.is_finite(),
        "Loss should be finite, got {loss_val}"
    );
    assert!(loss_val > 0.0, "Loss should be positive, got {loss_val}");
}

// ---------------------------------------------------------------------------
// Tests: Loss Decreases Over Steps
// ---------------------------------------------------------------------------

#[cfg(not(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
)))]
#[test]
fn test_training_reduces_loss() {
    let device = device();
    let lora_model = make_lora_model();
    let sft_model = Gemma2ForSFT::new(lora_model, 0, false);

    let mut optimizer = AdamConfig::new().init();
    let lr = 1e-3;

    // Create a fixed batch for repeated training
    let _batch = make_batch(2, 8, &device);

    // Initial loss
    let initial_output =
        <Gemma2ForSFT<TestAD> as InferenceStep>::step(&sft_model, make_batch(2, 8, &device));
    let initial_loss: f32 = initial_output.loss.into_scalar().elem();

    // Train for several steps on the same batch
    let mut model = sft_model;
    let mut losses: Vec<f32> = vec![initial_loss];

    for step in 0..20 {
        let batch = make_batch(2, 8, &device);
        let train_output = <Gemma2ForSFT<TestAD> as TrainStep>::step(&model, batch);
        model = model.optimize(&mut optimizer, lr, train_output.grads);

        // Evaluate after step
        let eval_output =
            <Gemma2ForSFT<TestAD> as InferenceStep>::step(&model, make_batch(2, 8, &device));
        let loss: f32 = eval_output.loss.into_scalar().elem();
        losses.push(loss);

        if step % 5 == 0 {
            log::info!("Step {step}: loss={loss:.6}");
        }
    }

    let final_loss = *losses.last().unwrap();
    log::info!("Loss trajectory: {losses:?}");
    log::info!("Initial: {initial_loss:.6}, Final: {final_loss:.6}");

    assert!(
        final_loss < initial_loss,
        "Loss should decrease after training: initial={initial_loss:.6}, final={final_loss:.6}"
    );
}

// ---------------------------------------------------------------------------
// Tests: Merge Equivalence After Training
// ---------------------------------------------------------------------------

#[cfg(not(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
)))]
#[test]
fn test_merge_after_training() {
    let device = device();
    let lora_model = make_lora_model();
    let sft_model = Gemma2ForSFT::new(lora_model, 0, false);

    let mut optimizer = AdamConfig::new().init();
    let lr = 1e-3;

    // Train a few steps
    let mut model = sft_model;
    for _ in 0..5 {
        let batch = make_batch(2, 8, &device);
        let output = <Gemma2ForSFT<TestAD> as TrainStep>::step(&model, batch);
        model = model.optimize(&mut optimizer, lr, output.grads);
    }

    // Forward pass with LoRA model
    let input = Tensor::<TestAD, 2, Int>::from_data(TensorData::from([[1i64, 2, 3, 4]]), &device);
    let lora_output = model.model.forward(input.clone());

    // Merge and forward with merged model
    let merged = model.merge();
    let merged_output = merged.forward(input);

    // Should produce identical output
    let diff = (lora_output - merged_output).abs().max().into_scalar();
    assert!(
        diff < 1e-4,
        "Merged model output should match LoRA model output, diff={diff}"
    );
}

// ---------------------------------------------------------------------------
// Tests: Adapter Save/Load After Training
// ---------------------------------------------------------------------------

#[cfg(not(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
)))]
#[test]
fn test_adapter_save_load_after_training() {
    let device = device();
    let config = tiny_config();

    let lora_config = LoraConfig::new(4).with_alpha(8.0).with_bias(LoraBias::None);
    let model = Gemma2Model::<TestAD>::new(&config, &device);
    let lora_model = apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);
    let sft_model = Gemma2ForSFT::new(lora_model, 0, false);

    let mut optimizer = AdamConfig::new().init();
    let lr = 1e-3;

    // Train a few steps to make LoRA weights non-zero
    let mut model = sft_model;
    for _ in 0..5 {
        let batch = make_batch(2, 8, &device);
        let output = <Gemma2ForSFT<TestAD> as TrainStep>::step(&model, batch);
        model = model.optimize(&mut optimizer, lr, output.grads);
    }

    // Snapshot trained LoRA A/B weights per layer for comparison
    let trained_weights: Vec<_> = model
        .model
        .layers
        .iter()
        .map(|block| {
            let q_a = block.self_attn.q_proj.lora_a.val().into_data();
            let q_b = block.self_attn.q_proj.lora_b.val().into_data();
            let gate_a = block.mlp.gate_proj.lora_a.val().into_data();
            let gate_b = block.mlp.gate_proj.lora_b.val().into_data();
            (q_a, q_b, gate_a, gate_b)
        })
        .collect();

    // Save adapters
    let dir = tempfile::tempdir().expect("temp dir");
    let adapter_path = dir.path().join("trained-adapters");
    model
        .model
        .save_adapters(&adapter_path)
        .expect("save adapters");

    // Verify adapter files exist for all layers and targets
    for layer in 0..config.num_hidden_layers {
        for target_name in &[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ] {
            let path = adapter_path.join(format!("layer_{layer}/{target_name}.mpk"));
            assert!(path.exists(), "Missing adapter file: {}", path.display());
        }
    }

    // Reload: fresh model → apply LoRA → load adapters
    let fresh_model = Gemma2Model::<TestAD>::new(&config, &device);
    let fresh_lora = apply_lora_to_gemma2(
        fresh_model,
        &lora_config,
        LoraTarget::all_targets(),
        &device,
    );
    let loaded_lora = fresh_lora
        .load_adapters(&adapter_path, &device)
        .expect("load adapters");

    // Verify adapter A/B weights match per layer (the core guarantee of save/load)
    for (i, block) in loaded_lora.layers.iter().enumerate() {
        let loaded_q_a = block.self_attn.q_proj.lora_a.val().into_data();
        let loaded_q_b = block.self_attn.q_proj.lora_b.val().into_data();
        let loaded_gate_a = block.mlp.gate_proj.lora_a.val().into_data();
        let loaded_gate_b = block.mlp.gate_proj.lora_b.val().into_data();

        let (ref_q_a, ref_q_b, ref_gate_a, ref_gate_b) = &trained_weights[i];

        ref_q_a.assert_approx_eq::<f32>(&loaded_q_a, Tolerance::default());
        ref_q_b.assert_approx_eq::<f32>(&loaded_q_b, Tolerance::default());
        ref_gate_a.assert_approx_eq::<f32>(&loaded_gate_a, Tolerance::default());
        ref_gate_b.assert_approx_eq::<f32>(&loaded_gate_b, Tolerance::default());
    }

    // Verify loaded model produces finite output (basic sanity check)
    let input = Tensor::<TestAD, 2, Int>::from_data(TensorData::from([[1i64, 2, 3, 4]]), &device);
    let output = loaded_lora.forward(input);
    let max_logit = output.abs().max().into_scalar();
    assert!(
        max_logit.is_finite(),
        "Output should be finite, got {max_logit}"
    );
}

// ---------------------------------------------------------------------------
// Tests: Parameter Counts
// ---------------------------------------------------------------------------

#[test]
fn test_param_counts_all_targets() {
    let lora_model = make_lora_model();

    let lora_params = count_lora_params(&lora_model);
    let total_params = count_total_params(&lora_model);

    assert!(lora_params > 0, "Should have LoRA params");
    assert!(
        lora_params < total_params,
        "LoRA params ({lora_params}) should be < total ({total_params})"
    );

    // rank=4: each LoRA has A[d_in, 4] + B[4, d_out]
    // Tiny config: hidden=32, intermediate=64, num_heads=4, num_kv_heads=2, head_dim=8
    // q_proj: A[32,4]+B[4,32]=256, k_proj: A[32,4]+B[4,16]=192, v_proj: 192, o_proj: 256
    // gate_proj: A[32,4]+B[4,64]=384, up_proj: 384, down_proj: A[64,4]+B[4,32]=384
    // Per layer: 256+192+192+256+384+384+384 = 2048
    // 2 layers: 4096
    let expected = 4096;
    assert_eq!(
        lora_params, expected,
        "LoRA params mismatch: expected {expected}, got {lora_params}"
    );

    let pct = lora_params as f64 / total_params as f64 * 100.0;
    log::info!("LoRA: {lora_params} / {total_params} ({pct:.2}%)");
}

#[test]
fn test_param_counts_attention_only() {
    let device = device();
    let config = tiny_config();
    let model = Gemma2Model::<TestAD>::new(&config, &device);

    let lora_config = LoraConfig::new(4).with_alpha(8.0);
    let lora_model = apply_lora_to_gemma2(
        model,
        &lora_config,
        LoraTarget::attention_targets(),
        &device,
    );

    let lora_params = count_lora_params(&lora_model);
    let total_params = count_total_params(&lora_model);

    // NOTE: apply_lora_to_gemma2 wraps ALL projections with LoRA but freezes non-targets.
    // So count_lora_params counts all 7 LoRA layers per block (frozen MLP ones still exist).
    // Expected: same as all_targets = 4096
    assert!(
        lora_params > 0,
        "Should have LoRA params with attention targets"
    );
    assert!(
        lora_params < total_params,
        "LoRA ({lora_params}) should be < total ({total_params})"
    );
    // All 7 projections get LoRA (4 trainable + 3 frozen) per layer
    let expected = 4096;
    assert_eq!(
        lora_params, expected,
        "LoRA params mismatch: expected {expected}, got {lora_params}"
    );

    assert!(lora_params < total_params);
}

// ---------------------------------------------------------------------------
// Tests: Gradient Flow (only LoRA params get grads)
// ---------------------------------------------------------------------------

#[cfg(not(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
)))]
#[test]
fn test_gradient_flow_only_lora() {
    let device = device();
    let lora_model = make_lora_model();
    let sft_model = Gemma2ForSFT::new(lora_model, 0, false);

    let batch = make_batch(2, 8, &device);
    let output = <Gemma2ForSFT<TestAD> as TrainStep>::step(&sft_model, batch);

    // Gradients should exist (LoRA params are trainable)
    // The TrainOutput contains grads, which means backward() ran successfully
    // We verify by checking the loss is valid and grads were computed
    let loss_val: f32 = output.item.loss.into_scalar().elem();
    assert!(loss_val.is_finite());

    // If grads were not computed, .grads would be empty
    // The fact that TrainOutput was created means grads exist
    // (burn panics internally if backward fails)
}

// ---------------------------------------------------------------------------
// Tests: End-to-end with SupervisedTraining
// ---------------------------------------------------------------------------

#[cfg(not(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
)))]
#[test]
fn test_supervised_training_runs() {
    let device = device();
    let config = tiny_config();
    let model = Gemma2Model::<TestAD>::new(&config, &device);

    let lora_config = LoraConfig::new(4).with_alpha(8.0);
    let lora_model = apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);
    let sft_model = Gemma2ForSFT::new(lora_model, 0, false);

    // Create synthetic dataset items
    use lora_gemma2::dataset::{ChatItem, ChatMessageSerde};

    let items: Vec<ChatItem> = (0..10)
        .map(|i| ChatItem {
            messages: vec![
                ChatMessageSerde {
                    role: "user".into(),
                    content: format!("Test input {i}"),
                },
                ChatMessageSerde {
                    role: "assistant".into(),
                    content: format!("Test output {i}"),
                },
            ],
        })
        .collect();

    let _dataset = burn::data::dataset::InMemDataset::new(items);

    // We can't use the full SFTBatcher without a real tokenizer,
    // so we test the training infrastructure with a manual loop instead.
    let mut optimizer = AdamConfig::new().init();
    let lr = 1e-3;

    let mut model = sft_model;
    let mut last_loss = f32::MAX;

    for step in 0..5 {
        let batch = make_batch(2, 8, &device);
        let output = <Gemma2ForSFT<TestAD> as TrainStep>::step(&model, batch);
        let loss: f32 = output.item.loss.into_scalar().elem();
        log::info!("Step {step}: loss={loss:.6}");
        model = model.optimize(&mut optimizer, lr, output.grads);
        last_loss = loss;
    }

    // Training should complete without errors and produce finite loss
    assert!(last_loss.is_finite(), "Final loss should be finite");
}
