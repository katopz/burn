//! LoRA-adapted Gemma 4 model for parameter-efficient SFT (Supervised Fine-Tuning).
//!
//! Wraps [`Gemma4Model`](crate::Gemma4Model) with LoRA (Low-Rank Adaptation) layers
//! on attention projections (q/k/v/o) and MLP projections (gate/up/down).
//!
//! # Architecture
//!
//! The LoRA-adapted model mirrors the original Gemma 4 architecture:
//! - [`Gemma4AttentionLora`] — GQA attention with LoRA on q/k/v/o + Q/K norms
//! - [`Gemma4MLPLora`] — Gated MLP with LoRA on gate/up/down projections
//! - [`Gemma4BlockLora`] — Transformer block with LoRA attention + MLP + PLE
//! - [`Gemma4ModelLora`] — Full model with LoRA-adapted layers + KV sharing
//! - [`Gemma4ForSFT`] — Training wrapper with cross-entropy loss
//!
//! # Usage
//!
//! ```ignore
//! use lora_gemma4::{Gemma4Config, Gemma4Model, LoraTarget};
//! use lora_gemma4::model_lora::{apply_lora_to_gemma4, Gemma4ForSFT};
//! use burn::nn::lora::{LoraConfig, LoraBias};
//!
//! let config = Gemma4Config::gemma4_e4b();
//! let model = Gemma4Model::new(&config, &device);
//!
//! let lora_config = LoraConfig::new(16).with_alpha(32.0).with_bias(LoraBias::None);
//! let targets = LoraTarget::all_targets();
//! let lora_model = apply_lora_to_gemma4(model, &lora_config, targets, &device);
//!
//! let sft_model = Gemma4ForSFT::new(lora_model, 0);
//! // ... train with burn's Learner ...
//! let merged = sft_model.model.merge();
//! ```

use std::path::PathBuf;

use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::nn::lora::{LoraAdaptable, LoraConfig, LoraLinear};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Embedding, Linear, RmsNorm, RotaryEncoding};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{
    DType, Element, Int, Tensor, activation::gelu_approximate, activation::softmax,
    backend::Backend,
};
use burn::train::{InferenceStep, SequenceOutput, TrainOutput, TrainStep};

use crate::model::{
    Gemma4Attention, Gemma4Block, Gemma4MLP, Gemma4Model, KvPair, causal_mask, rms_norm_no_scale,
    sliding_window_mask,
};
use crate::types::{LayerType, LoraTarget};

// ---------------------------------------------------------------------------
// SFT Training Batch
// ---------------------------------------------------------------------------

/// SFT training batch for Gemma 4.
///
/// Contains input token IDs and shifted target token IDs for next-token prediction.
#[derive(Clone, Debug)]
pub struct SFTTrainingBatch<B: Backend> {
    /// Input token IDs `[batch, seq]`.
    pub tokens_inputs: Tensor<B, 2, Int>,
    /// Target token IDs `[batch, seq]` (shifted by 1).
    pub targets: Tensor<B, 2, Int>,
}

// ---------------------------------------------------------------------------
// LoRA Attention
// ---------------------------------------------------------------------------

/// Gemma 4 multi-head attention with LoRA on all projections.
///
/// Identical forward logic to [`Gemma4Attention`](crate::Gemma4Attention)
/// but uses [`LoraLinear`] for q/k/v/o projections. Includes Q/K norms,
/// partial rotary encoding, and KV sharing support.
#[derive(Module, Debug)]
pub struct Gemma4AttentionLora<B: Backend> {
    pub q_proj: LoraLinear<B>,
    pub k_proj: LoraLinear<B>,
    pub v_proj: LoraLinear<B>,
    pub o_proj: LoraLinear<B>,
    pub q_norm: RmsNorm<B>,
    pub k_norm: RmsNorm<B>,
    pub rotary: RotaryEncoding<B>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub layer_type: LayerType,
    pub has_own_kv: bool,
    pub partial_rotary_factor: f64,
}

impl<B: Backend> Gemma4AttentionLora<B> {
    /// Apply rotary encoding, handling partial rotation for full attention layers.
    fn apply_rotary(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        if self.partial_rotary_factor >= 1.0 {
            return self.rotary.forward(x);
        }

        let dims = x.dims();
        let [batch, heads, seq, dim] = [dims[0], dims[1], dims[2], dims[3]];
        let rotary_dim = ((dim as f64) * self.partial_rotary_factor) as usize;

        if rotary_dim == 0 || rotary_dim >= dim {
            return self.rotary.forward(x);
        }

        let x_clone = x.clone();
        let x_rot = x.slice([0..batch, 0..heads, 0..seq, 0..rotary_dim]);
        let x_pass = x_clone.slice([0..batch, 0..heads, 0..seq, rotary_dim..dim]);
        let x_rot = self.rotary.forward(x_rot);
        Tensor::cat(vec![x_rot, x_pass], 3)
    }

    /// Forward pass: `[batch, seq, hidden] -> ([batch, seq, hidden], KvPair)`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        shared_kv: KvPair<B>,
    ) -> (Tensor<B, 3>, KvPair<B>) {
        let [batch, seq, _hidden] = x.dims();
        let kv_groups = self.num_heads / self.num_kv_heads;

        // --- Query (LoRA) ---
        let q = self.q_proj.forward(x.clone());
        let q = q.reshape([batch, seq, self.num_heads, self.head_dim]);
        let q = self.q_norm.forward(q);
        let q = q.swap_dims(1, 2);
        let q = self.apply_rotary(q);

        // --- Key, Value ---
        let (keys, values, own_kv) = match shared_kv {
            Some((shared_k, shared_v)) => (shared_k, shared_v, None),
            None => {
                let k = self.k_proj.forward(x.clone());
                let k = k.reshape([batch, seq, self.num_kv_heads, self.head_dim]);
                let k = self.k_norm.forward(k);
                let k = k.swap_dims(1, 2);
                let k = self.apply_rotary(k);

                let v = self.v_proj.forward(x);
                let v = v.reshape([batch, seq, self.num_kv_heads, self.head_dim]);
                let v = rms_norm_no_scale(v, 1e-6);
                let v = v.swap_dims(1, 2);

                (k.clone(), v.clone(), Some((k, v)))
            }
        };

        // --- GQA expansion ---
        let (keys, values) = match kv_groups > 1 {
            true => {
                let k = keys
                    .reshape([batch, self.num_kv_heads, 1, seq, self.head_dim])
                    .repeat_dim(2, kv_groups)
                    .reshape([batch, self.num_heads, seq, self.head_dim]);
                let v = values
                    .reshape([batch, self.num_kv_heads, 1, seq, self.head_dim])
                    .repeat_dim(2, kv_groups)
                    .reshape([batch, self.num_heads, seq, self.head_dim]);
                (k, v)
            }
            false => (keys, values),
        };

        // --- Attention scores in f32 to prevent f16 overflow (head_dim=512 for full attention) ---
        // HF computes entire attention in float32 to avoid f16 precision loss:
        //   torch.matmul(query, key_states.transpose(2, 3)) * scaling
        //   nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        let original_dtype = B::FloatElem::dtype();
        let scores = q
            .cast(DType::F32)
            .matmul(keys.cast(DType::F32).swap_dims(2, 3));

        let scores = match mask {
            Some(m) => scores.add(m.cast(DType::F32).reshape([1, 1, seq, seq])),
            None => scores,
        };

        // Softmax in f32, then cast back for value weighted sum
        let weights = softmax(scores, 3).cast(original_dtype);
        let output = weights.matmul(values);

        // --- Reshape + output projection (LoRA) ---
        let output = output
            .swap_dims(1, 2)
            .reshape([batch, seq, self.num_heads * self.head_dim]);
        let output = self.o_proj.forward(output);

        (output, own_kv)
    }
}

// ---------------------------------------------------------------------------
// LoRA MLP
// ---------------------------------------------------------------------------

/// Gemma 4 MLP with LoRA on gate/up/down projections (GeGLU).
#[derive(Module, Debug)]
pub struct Gemma4MLPLora<B: Backend> {
    pub gate_proj: LoraLinear<B>,
    pub up_proj: LoraLinear<B>,
    pub down_proj: LoraLinear<B>,
}

impl<B: Backend> Gemma4MLPLora<B> {
    /// Forward pass: `[batch, seq, hidden] -> [batch, seq, hidden]`.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gate = gelu_approximate(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate.mul(up))
    }
}

// ---------------------------------------------------------------------------
// LoRA Transformer Block
// ---------------------------------------------------------------------------

/// Gemma 4 transformer block with LoRA-adapted attention and MLP.
///
/// Includes sandwich normalization, PLE (Per-Layer Embeddings), and layer scalar.
#[derive(Module, Debug)]
pub struct Gemma4BlockLora<B: Backend> {
    pub self_attn: Gemma4AttentionLora<B>,
    pub mlp: Gemma4MLPLora<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub pre_feedforward_layernorm: RmsNorm<B>,
    pub post_feedforward_layernorm: RmsNorm<B>,
    pub per_layer_input_gate: Option<Linear<B>>,
    pub per_layer_projection: Option<Linear<B>>,
    pub post_per_layer_input_norm: Option<RmsNorm<B>>,
    pub layer_scalar: f64,
}

impl<B: Backend> Gemma4BlockLora<B> {
    /// Forward pass with KV sharing and optional PLE input.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        shared_kv: KvPair<B>,
        per_layer_input: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, KvPair<B>) {
        // Attention with sandwich norm
        let residual = x.clone();
        let (attn_out, own_kv) =
            self.self_attn
                .forward(self.input_layernorm.forward(x), mask, shared_kv);
        let h = residual + self.post_attention_layernorm.forward(attn_out);

        // MLP with sandwich norm
        let residual = h.clone();
        let mlp_out = self.mlp.forward(self.pre_feedforward_layernorm.forward(h));
        let mut h = residual + self.post_feedforward_layernorm.forward(mlp_out);

        // Per-Layer Embeddings (PLE)
        if let (Some(gate), Some(proj), Some(norm), Some(ple_input)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let residual = h.clone();
            let gate_val = gelu_approximate(gate.forward(h));
            let gated = gate_val.mul(ple_input);
            let projected = proj.forward(gated);
            h = residual + norm.forward(projected);
        }

        // Layer scalar (always 1.0 for E4B)
        if (self.layer_scalar - 1.0).abs() > f64::EPSILON {
            h = h.mul_scalar(self.layer_scalar);
        }

        (h, own_kv)
    }
}

// ---------------------------------------------------------------------------
// LoRA Model
// ---------------------------------------------------------------------------

/// Gemma 4 model with LoRA-adapted transformer blocks.
///
/// Embedding, final norm, LM head, and PLE model-level components are frozen.
/// Only the transformer block projections (attention + MLP) have LoRA.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Gemma4ModelLora<B: Backend> {
    pub embed: Embedding<B>,
    pub layers: Vec<Gemma4BlockLora<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
    /// PLE model-level components (frozen).
    pub embed_tokens_per_layer: Option<Embedding<B>>,
    pub per_layer_model_projection: Option<Linear<B>>,
    pub per_layer_projection_norm: Option<RmsNorm<B>>,
    // Config
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub final_logit_softcapping: f64,
    pub tie_word_embeddings: bool,
    pub num_hidden_layers: usize,
    pub hidden_size_per_layer_input: usize,
    pub sliding_window: usize,
    /// KV sharing map: for each layer, the index of the layer it shares KV with.
    pub kv_source_map: Vec<usize>,
}

impl<B: Backend> Gemma4ModelLora<B> {
    /// Forward pass: token IDs → logits.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let hidden = self.forward_hidden(input_ids);
        self.hidden_to_logits(hidden)
    }

    /// Forward pass returning hidden states (before LM head).
    pub fn forward_hidden(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = input_ids.device();
        let [_batch, seq_len] = input_ids.dims();

        // Embedding + Gemma scaling
        let scale = (self.hidden_size as f64).sqrt();
        let h = self.embed.forward(input_ids.clone()).mul_scalar(scale);

        // Pre-compute masks
        let causal = causal_mask::<B>(seq_len, &device);
        let sliding = sliding_window_mask::<B>(seq_len, self.sliding_window, &device);

        // Pre-compute per-layer inputs (PLE)
        let per_layer_inputs = self.compute_per_layer_inputs(&h, &input_ids);

        // Forward through layers with KV sharing
        let mut h = h;
        let mut kv_cache: Vec<KvPair<B>> = vec![None; self.num_hidden_layers];

        for (i, layer) in self.layers.iter().enumerate() {
            // Select mask based on layer type
            let mask = match layer.self_attn.layer_type {
                LayerType::Sliding => Some(sliding.clone()),
                LayerType::Full => Some(causal.clone()),
            };

            // Get shared KV if this is a shared layer
            let shared_kv = match i < self.kv_source_map.len() && self.kv_source_map[i] != i {
                true => kv_cache[self.kv_source_map[i]].clone(),
                false => None,
            };

            let ple_input = per_layer_inputs.as_ref().map(|inputs| inputs[i].clone());

            let (new_h, own_kv) = layer.forward(h, mask, shared_kv, ple_input);

            if let Some(kv) = own_kv {
                kv_cache[i] = Some(kv);
            }

            h = new_h;
        }

        self.norm.forward(h)
    }

    /// Compute logits from hidden states with logit softcapping.
    pub fn hidden_to_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let logits = self.lm_head.forward(hidden);

        match (self.final_logit_softcapping - 0.0).abs() > f64::EPSILON {
            true => logits
                .div_scalar(self.final_logit_softcapping)
                .tanh()
                .mul_scalar(self.final_logit_softcapping),
            false => logits,
        }
    }

    /// Compute per-layer inputs for PLE.
    fn compute_per_layer_inputs(
        &self,
        hidden: &Tensor<B, 3>,
        input_ids: &Tensor<B, 2, Int>,
    ) -> Option<Vec<Tensor<B, 3>>> {
        let ple_dim = self.hidden_size_per_layer_input;
        if ple_dim == 0 {
            return None;
        }

        let embed_per_layer = self.embed_tokens_per_layer.as_ref()?;
        let proj_per_layer = self.per_layer_model_projection.as_ref()?;
        let norm_per_layer = self.per_layer_projection_norm.as_ref()?;

        let [batch, seq, _hidden] = hidden.dims();
        let num_layers = self.num_hidden_layers;

        let ple_scale = (ple_dim as f64).sqrt();
        let token_ple = embed_per_layer
            .forward(input_ids.clone())
            .mul_scalar(ple_scale);
        let token_ple = token_ple.reshape([batch, seq, num_layers, ple_dim]);

        let proj_scale = 1.0 / (self.hidden_size as f64).sqrt();
        let hidden_ple = proj_per_layer
            .forward(hidden.clone())
            .mul_scalar(proj_scale);
        let hidden_ple = hidden_ple.reshape([batch, seq, num_layers, ple_dim]);
        let hidden_ple = norm_per_layer.forward(hidden_ple);

        let combine_scale = 1.0 / 2.0_f64.sqrt();
        let combined = (token_ple + hidden_ple).mul_scalar(combine_scale);

        let per_layer_inputs: Vec<Tensor<B, 3>> = (0..num_layers)
            .map(|i| {
                combined
                    .clone()
                    .slice([0..batch, 0..seq, i..(i + 1), 0..ple_dim])
                    .reshape([batch, seq, ple_dim])
            })
            .collect();

        Some(per_layer_inputs)
    }

    /// Merge all LoRA weights into base layers for inference.
    ///
    /// Returns a standard [`Gemma4Model`] with no LoRA overhead.
    pub fn merge(self) -> Gemma4Model<B> {
        let layers = self
            .layers
            .into_iter()
            .map(|block| Gemma4Block {
                self_attn: Gemma4Attention {
                    q_proj: block.self_attn.q_proj.merge(),
                    k_proj: block.self_attn.k_proj.merge(),
                    v_proj: block.self_attn.v_proj.merge(),
                    o_proj: block.self_attn.o_proj.merge(),
                    q_norm: block.self_attn.q_norm,
                    k_norm: block.self_attn.k_norm,
                    rotary: block.self_attn.rotary,
                    num_heads: block.self_attn.num_heads,
                    num_kv_heads: block.self_attn.num_kv_heads,
                    head_dim: block.self_attn.head_dim,
                    layer_type: block.self_attn.layer_type,
                    has_own_kv: block.self_attn.has_own_kv,
                    partial_rotary_factor: block.self_attn.partial_rotary_factor,
                },
                mlp: Gemma4MLP {
                    gate_proj: block.mlp.gate_proj.merge(),
                    up_proj: block.mlp.up_proj.merge(),
                    down_proj: block.mlp.down_proj.merge(),
                },
                input_layernorm: block.input_layernorm,
                post_attention_layernorm: block.post_attention_layernorm,
                pre_feedforward_layernorm: block.pre_feedforward_layernorm,
                post_feedforward_layernorm: block.post_feedforward_layernorm,
                per_layer_input_gate: block.per_layer_input_gate,
                per_layer_projection: block.per_layer_projection,
                post_per_layer_input_norm: block.post_per_layer_input_norm,
                layer_scalar: block.layer_scalar,
            })
            .collect();

        Gemma4Model {
            embed: self.embed,
            layers,
            norm: self.norm,
            lm_head: self.lm_head,
            embed_tokens_per_layer: self.embed_tokens_per_layer,
            per_layer_model_projection: self.per_layer_model_projection,
            per_layer_projection_norm: self.per_layer_projection_norm,
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
            final_logit_softcapping: self.final_logit_softcapping,
            tie_word_embeddings: self.tie_word_embeddings,
            num_hidden_layers: self.num_hidden_layers,
            hidden_size_per_layer_input: self.hidden_size_per_layer_input,
            sliding_window: self.sliding_window,
            kv_source_map: self.kv_source_map,
        }
    }

    /// Save all LoRA adapter weights to a directory.
    pub fn save_adapters(&self, path: impl Into<PathBuf>) -> Result<(), RecorderError> {
        let path = path.into();
        std::fs::create_dir_all(&path).map_err(|e| RecorderError::Unknown(format!("{e}")))?;

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        for (i, block) in self.layers.iter().enumerate() {
            let layer_dir = path.join(format!("layer_{i}"));
            std::fs::create_dir_all(&layer_dir)
                .map_err(|e| RecorderError::Unknown(format!("{e}")))?;

            let attn = &block.self_attn;
            attn.q_proj
                .save_adapter(layer_dir.join("q_proj"), &recorder)?;
            attn.k_proj
                .save_adapter(layer_dir.join("k_proj"), &recorder)?;
            attn.v_proj
                .save_adapter(layer_dir.join("v_proj"), &recorder)?;
            attn.o_proj
                .save_adapter(layer_dir.join("o_proj"), &recorder)?;

            let mlp = &block.mlp;
            mlp.gate_proj
                .save_adapter(layer_dir.join("gate_proj"), &recorder)?;
            mlp.up_proj
                .save_adapter(layer_dir.join("up_proj"), &recorder)?;
            mlp.down_proj
                .save_adapter(layer_dir.join("down_proj"), &recorder)?;
        }

        Ok(())
    }

    /// Load LoRA adapter weights from a directory.
    pub fn load_adapters(
        self,
        path: impl Into<PathBuf>,
        device: &B::Device,
    ) -> Result<Self, RecorderError> {
        let path = path.into();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        let mut layers = Vec::with_capacity(self.layers.len());
        for (i, block) in self.layers.into_iter().enumerate() {
            let layer_dir = path.join(format!("layer_{i}"));

            let q_proj = block.self_attn.q_proj.load_adapter_file(
                layer_dir.join("q_proj"),
                &recorder,
                device,
            )?;
            let k_proj = block.self_attn.k_proj.load_adapter_file(
                layer_dir.join("k_proj"),
                &recorder,
                device,
            )?;
            let v_proj = block.self_attn.v_proj.load_adapter_file(
                layer_dir.join("v_proj"),
                &recorder,
                device,
            )?;
            let o_proj = block.self_attn.o_proj.load_adapter_file(
                layer_dir.join("o_proj"),
                &recorder,
                device,
            )?;

            let gate_proj = block.mlp.gate_proj.load_adapter_file(
                layer_dir.join("gate_proj"),
                &recorder,
                device,
            )?;
            let up_proj = block.mlp.up_proj.load_adapter_file(
                layer_dir.join("up_proj"),
                &recorder,
                device,
            )?;
            let down_proj = block.mlp.down_proj.load_adapter_file(
                layer_dir.join("down_proj"),
                &recorder,
                device,
            )?;

            layers.push(Gemma4BlockLora {
                self_attn: Gemma4AttentionLora {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: block.self_attn.q_norm,
                    k_norm: block.self_attn.k_norm,
                    rotary: block.self_attn.rotary,
                    num_heads: block.self_attn.num_heads,
                    num_kv_heads: block.self_attn.num_kv_heads,
                    head_dim: block.self_attn.head_dim,
                    layer_type: block.self_attn.layer_type,
                    has_own_kv: block.self_attn.has_own_kv,
                    partial_rotary_factor: block.self_attn.partial_rotary_factor,
                },
                mlp: Gemma4MLPLora {
                    gate_proj,
                    up_proj,
                    down_proj,
                },
                input_layernorm: block.input_layernorm,
                post_attention_layernorm: block.post_attention_layernorm,
                pre_feedforward_layernorm: block.pre_feedforward_layernorm,
                post_feedforward_layernorm: block.post_feedforward_layernorm,
                per_layer_input_gate: block.per_layer_input_gate,
                per_layer_projection: block.per_layer_projection,
                post_per_layer_input_norm: block.post_per_layer_input_norm,
                layer_scalar: block.layer_scalar,
            });
        }

        Ok(Gemma4ModelLora {
            layers,
            embed: self.embed,
            norm: self.norm,
            lm_head: self.lm_head,
            embed_tokens_per_layer: self.embed_tokens_per_layer,
            per_layer_model_projection: self.per_layer_model_projection,
            per_layer_projection_norm: self.per_layer_projection_norm,
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
            final_logit_softcapping: self.final_logit_softcapping,
            tie_word_embeddings: self.tie_word_embeddings,
            num_hidden_layers: self.num_hidden_layers,
            hidden_size_per_layer_input: self.hidden_size_per_layer_input,
            sliding_window: self.sliding_window,
            kv_source_map: self.kv_source_map,
        })
    }
}

impl<B: Backend> ModuleDisplay for Gemma4ModelLora<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("hidden_size", &self.hidden_size)
            .add("vocab_size", &self.vocab_size)
            .add("num_layers", &self.layers.len())
            .add("final_logit_softcapping", &self.final_logit_softcapping)
            .optional()
    }
}

// ---------------------------------------------------------------------------
// Apply LoRA to Model
// ---------------------------------------------------------------------------

/// Apply LoRA adaptation to a [`Gemma4Model`].
///
/// Converts the model into a [`Gemma4ModelLora`] by wrapping specified
/// projection layers with LoRA. Layers matching `targets` get trainable
/// LoRA params; non-target layers are wrapped with frozen LoRA
/// (output is zero at initialization since B is initialized to zeros).
pub fn apply_lora_to_gemma4<B: Backend>(
    model: Gemma4Model<B>,
    config: &LoraConfig,
    targets: &[LoraTarget],
    device: &B::Device,
) -> Gemma4ModelLora<B> {
    let Gemma4Model {
        embed,
        layers,
        norm,
        lm_head,
        embed_tokens_per_layer,
        per_layer_model_projection,
        per_layer_projection_norm,
        hidden_size,
        vocab_size,
        final_logit_softcapping,
        tie_word_embeddings,
        num_hidden_layers,
        hidden_size_per_layer_input,
        sliding_window,
        kv_source_map,
    } = model;

    let layers = layers
        .into_iter()
        .map(|block| apply_lora_to_block(block, config, targets, device))
        .collect();

    Gemma4ModelLora {
        embed: embed.no_grad(),
        layers,
        norm,
        lm_head: lm_head.no_grad(),
        embed_tokens_per_layer: embed_tokens_per_layer.map(|e| e.no_grad()),
        per_layer_model_projection: per_layer_model_projection.map(|p| p.no_grad()),
        per_layer_projection_norm,
        hidden_size,
        vocab_size,
        final_logit_softcapping,
        tie_word_embeddings,
        num_hidden_layers,
        hidden_size_per_layer_input,
        sliding_window,
        kv_source_map,
    }
}

fn apply_lora_to_block<B: Backend>(
    block: Gemma4Block<B>,
    config: &LoraConfig,
    targets: &[LoraTarget],
    device: &B::Device,
) -> Gemma4BlockLora<B> {
    Gemma4BlockLora {
        self_attn: apply_lora_to_attention(block.self_attn, config, targets, device),
        mlp: apply_lora_to_mlp(block.mlp, config, targets, device),
        input_layernorm: block.input_layernorm,
        post_attention_layernorm: block.post_attention_layernorm,
        pre_feedforward_layernorm: block.pre_feedforward_layernorm,
        post_feedforward_layernorm: block.post_feedforward_layernorm,
        per_layer_input_gate: block.per_layer_input_gate,
        per_layer_projection: block.per_layer_projection,
        post_per_layer_input_norm: block.post_per_layer_input_norm,
        layer_scalar: block.layer_scalar,
    }
}

fn apply_lora_to_attention<B: Backend>(
    attn: Gemma4Attention<B>,
    config: &LoraConfig,
    targets: &[LoraTarget],
    device: &B::Device,
) -> Gemma4AttentionLora<B> {
    Gemma4AttentionLora {
        q_proj: wrap_lora(attn.q_proj, config, LoraTarget::QProj, targets, device),
        k_proj: wrap_lora(attn.k_proj, config, LoraTarget::KProj, targets, device),
        v_proj: wrap_lora(attn.v_proj, config, LoraTarget::VProj, targets, device),
        o_proj: wrap_lora(attn.o_proj, config, LoraTarget::OProj, targets, device),
        q_norm: attn.q_norm,
        k_norm: attn.k_norm,
        rotary: attn.rotary,
        num_heads: attn.num_heads,
        num_kv_heads: attn.num_kv_heads,
        head_dim: attn.head_dim,
        layer_type: attn.layer_type,
        has_own_kv: attn.has_own_kv,
        partial_rotary_factor: attn.partial_rotary_factor,
    }
}

fn apply_lora_to_mlp<B: Backend>(
    mlp: Gemma4MLP<B>,
    config: &LoraConfig,
    targets: &[LoraTarget],
    device: &B::Device,
) -> Gemma4MLPLora<B> {
    Gemma4MLPLora {
        gate_proj: wrap_lora(mlp.gate_proj, config, LoraTarget::GateProj, targets, device),
        up_proj: wrap_lora(mlp.up_proj, config, LoraTarget::UpProj, targets, device),
        down_proj: wrap_lora(mlp.down_proj, config, LoraTarget::DownProj, targets, device),
    }
}

/// Wrap a Linear layer with LoRA.
///
/// If `target` is in `targets`, LoRA params are trainable.
/// Otherwise, everything is frozen (LoRA output is zero at init).
fn wrap_lora<B: Backend>(
    linear: Linear<B>,
    config: &LoraConfig,
    target: LoraTarget,
    targets: &[LoraTarget],
    device: &B::Device,
) -> LoraLinear<B> {
    let lora = linear.with_lora(config, device);
    match targets.contains(&target) {
        true => lora,
        false => lora.no_grad(),
    }
}

// ---------------------------------------------------------------------------
// SFT Training Wrapper
// ---------------------------------------------------------------------------

/// Training wrapper for LoRA-adapted Gemma 4 with SFT (Supervised Fine-Tuning).
///
/// Implements [`TrainStep`] and [`InferenceStep`] for use with burn's
/// [`Learner`](burn::train::Learner). Computes cross-entropy loss
/// for next-token prediction, ignoring pad tokens.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Gemma4ForSFT<B: Backend> {
    /// The LoRA-adapted Gemma 4 model.
    pub model: Gemma4ModelLora<B>,
    /// Pad token ID to ignore in cross-entropy loss.
    pub pad_token_id: usize,
}

impl<B: Backend> Gemma4ForSFT<B> {
    /// Create a new SFT training wrapper.
    pub fn new(model: Gemma4ModelLora<B>, pad_token_id: usize) -> Self {
        Self {
            model,
            pad_token_id,
        }
    }

    /// Forward pass: token IDs -> logits.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.model.forward(input_ids)
    }

    /// Merge LoRA weights and return base model for inference.
    pub fn merge(self) -> Gemma4Model<B> {
        self.model.merge()
    }
}

impl<B: Backend> ModuleDisplay for Gemma4ForSFT<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("pad_token_id", &self.pad_token_id)
            .add("model", &self.model)
            .optional()
    }
}

// ---------------------------------------------------------------------------
// TrainStep / InferenceStep
// ---------------------------------------------------------------------------

impl<B: AutodiffBackend> TrainStep for Gemma4ForSFT<B> {
    type Input = SFTTrainingBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, batch: SFTTrainingBatch<B>) -> TrainOutput<SequenceOutput<B>> {
        let logits = self.model.forward(batch.tokens_inputs);
        let targets = batch.targets;

        let [batch_size, seq_len, vocab_size] = logits.dims();
        let flat_logits = logits.clone().reshape([batch_size * seq_len, vocab_size]);
        let flat_targets = targets.clone().reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token_id]))
            .init(&logits.device())
            .forward(flat_logits, flat_targets);

        TrainOutput::new(
            self,
            loss.backward(),
            SequenceOutput::new(loss, logits, None, targets),
        )
    }
}

impl<B: Backend> InferenceStep for Gemma4ForSFT<B> {
    type Input = SFTTrainingBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, batch: SFTTrainingBatch<B>) -> SequenceOutput<B> {
        let logits = self.model.forward(batch.tokens_inputs);
        let targets = batch.targets;

        let [batch_size, seq_len, vocab_size] = logits.dims();
        let flat_logits = logits.clone().reshape([batch_size * seq_len, vocab_size]);
        let flat_targets = targets.clone().reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token_id]))
            .init(&logits.device())
            .forward(flat_logits, flat_targets);

        SequenceOutput::new(loss, logits, None, targets)
    }
}

// ---------------------------------------------------------------------------
// Parameter Counting
// ---------------------------------------------------------------------------

/// Count total LoRA parameters (A + B matrices) across all layers.
///
/// This represents the number of trainable parameters during fine-tuning.
pub fn count_lora_params<B: Backend>(model: &Gemma4ModelLora<B>) -> usize {
    model
        .layers
        .iter()
        .map(|block| {
            let attn = &block.self_attn;
            let mlp = &block.mlp;

            let q = attn.q_proj.lora_a.shape().num_elements()
                + attn.q_proj.lora_b.shape().num_elements();
            let k = attn.k_proj.lora_a.shape().num_elements()
                + attn.k_proj.lora_b.shape().num_elements();
            let v = attn.v_proj.lora_a.shape().num_elements()
                + attn.v_proj.lora_b.shape().num_elements();
            let o = attn.o_proj.lora_a.shape().num_elements()
                + attn.o_proj.lora_b.shape().num_elements();

            let gate = mlp.gate_proj.lora_a.shape().num_elements()
                + mlp.gate_proj.lora_b.shape().num_elements();
            let up = mlp.up_proj.lora_a.shape().num_elements()
                + mlp.up_proj.lora_b.shape().num_elements();
            let down = mlp.down_proj.lora_a.shape().num_elements()
                + mlp.down_proj.lora_b.shape().num_elements();

            q + k + v + o + gate + up + down
        })
        .sum()
}

/// Count all parameters in the model (base weights + LoRA).
pub fn count_total_params<B: Backend>(model: &Gemma4ModelLora<B>) -> usize {
    let mut total: usize = model.embed.weight.shape().num_elements();

    for block in &model.layers {
        // Attention base + LoRA
        total += block.self_attn.q_proj.base.weight.shape().num_elements();
        total += block.self_attn.q_proj.lora_a.shape().num_elements();
        total += block.self_attn.q_proj.lora_b.shape().num_elements();

        total += block.self_attn.k_proj.base.weight.shape().num_elements();
        total += block.self_attn.k_proj.lora_a.shape().num_elements();
        total += block.self_attn.k_proj.lora_b.shape().num_elements();

        total += block.self_attn.v_proj.base.weight.shape().num_elements();
        total += block.self_attn.v_proj.lora_a.shape().num_elements();
        total += block.self_attn.v_proj.lora_b.shape().num_elements();

        total += block.self_attn.o_proj.base.weight.shape().num_elements();
        total += block.self_attn.o_proj.lora_a.shape().num_elements();
        total += block.self_attn.o_proj.lora_b.shape().num_elements();

        // Attention norms
        total += block.self_attn.q_norm.gamma.shape().num_elements();
        total += block.self_attn.k_norm.gamma.shape().num_elements();

        // MLP base + LoRA
        total += block.mlp.gate_proj.base.weight.shape().num_elements();
        total += block.mlp.gate_proj.lora_a.shape().num_elements();
        total += block.mlp.gate_proj.lora_b.shape().num_elements();

        total += block.mlp.up_proj.base.weight.shape().num_elements();
        total += block.mlp.up_proj.lora_a.shape().num_elements();
        total += block.mlp.up_proj.lora_b.shape().num_elements();

        total += block.mlp.down_proj.base.weight.shape().num_elements();
        total += block.mlp.down_proj.lora_a.shape().num_elements();
        total += block.mlp.down_proj.lora_b.shape().num_elements();

        // Block norms
        total += block.input_layernorm.gamma.shape().num_elements();
        total += block.post_attention_layernorm.gamma.shape().num_elements();
        total += block.pre_feedforward_layernorm.gamma.shape().num_elements();
        total += block
            .post_feedforward_layernorm
            .gamma
            .shape()
            .num_elements();

        // PLE block-level (optional)
        if let Some(gate) = &block.per_layer_input_gate {
            total += gate.weight.shape().num_elements();
        }
        if let Some(proj) = &block.per_layer_projection {
            total += proj.weight.shape().num_elements();
        }
        if let Some(norm) = &block.post_per_layer_input_norm {
            total += norm.gamma.shape().num_elements();
        }
    }

    // Final norm + LM head
    total += model.norm.gamma.shape().num_elements();
    total += model.lm_head.weight.shape().num_elements();

    // PLE model-level (optional)
    if let Some(emb) = &model.embed_tokens_per_layer {
        total += emb.weight.shape().num_elements();
    }
    if let Some(proj) = &model.per_layer_model_projection {
        total += proj.weight.shape().num_elements();
    }
    if let Some(norm) = &model.per_layer_projection_norm {
        total += norm.gamma.shape().num_elements();
    }

    total
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Gemma4Config;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn tiny_config() -> Gemma4Config {
        Gemma4Config::new(
            100, // vocab_size
            32,  // hidden_size
            2,   // num_hidden_layers
            64,  // intermediate_size
            4,   // num_attention_heads
            2,   // num_key_value_heads
            8,   // head_dim
        )
    }

    #[test]
    fn test_apply_lora_all_targets() {
        let device = device();
        let config = tiny_config();
        let model = Gemma4Model::<TestBackend>::new(&config, &device);

        let lora_config = LoraConfig::new(4).with_alpha(8.0);
        let lora_model =
            apply_lora_to_gemma4(model, &lora_config, LoraTarget::all_targets(), &device);

        assert_eq!(lora_model.layers.len(), config.num_hidden_layers);
        assert_eq!(lora_model.layers[0].self_attn.q_proj.rank(), 4);
    }

    #[test]
    fn test_apply_lora_attention_only() {
        let device = device();
        let config = tiny_config();
        let model = Gemma4Model::<TestBackend>::new(&config, &device);

        let lora_config = LoraConfig::new(8).with_alpha(16.0);
        let targets = LoraTarget::attention_targets();
        let lora_model = apply_lora_to_gemma4(model, &lora_config, targets, &device);

        assert_eq!(lora_model.layers.len(), 2);
        assert_eq!(lora_model.layers[0].self_attn.q_proj.rank(), 8);
    }

    #[test]
    fn test_lora_identity_start() {
        // LoRA starts with B=zeros, so output should match base model.
        let device = device();
        let config = tiny_config();
        let model = Gemma4Model::<TestBackend>::new(&config, &device);

        let input = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);
        let base_output = model.forward(input.clone());

        let lora_config = LoraConfig::new(4).with_alpha(8.0);
        let lora_model =
            apply_lora_to_gemma4(model, &lora_config, LoraTarget::all_targets(), &device);
        let lora_output = lora_model.forward(input);

        let diff = (base_output - lora_output).abs().max().into_scalar();
        assert!(
            diff < 1e-5,
            "LoRA should start as identity (B=zeros), diff={diff}"
        );
    }

    #[test]
    fn test_merge_output_equivalence() {
        let device = device();
        let config = tiny_config();
        let model = Gemma4Model::<TestBackend>::new(&config, &device);

        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        let base_output = model.forward(input.clone());

        let lora_config = LoraConfig::new(4).with_alpha(8.0);
        let lora_model =
            apply_lora_to_gemma4(model, &lora_config, LoraTarget::all_targets(), &device);

        let merged = lora_model.merge();
        let merged_output = merged.forward(input);

        let diff = (base_output - merged_output).abs().max().into_scalar();
        assert!(diff < 1e-4, "Merged output should match base, diff={diff}");
    }

    #[test]
    fn test_param_counts() {
        let device = device();
        let config = tiny_config();
        let model = Gemma4Model::<TestBackend>::new(&config, &device);

        let lora_config = LoraConfig::new(4);
        let lora_model =
            apply_lora_to_gemma4(model, &lora_config, LoraTarget::all_targets(), &device);

        let lora_params = count_lora_params(&lora_model);
        let total_params = count_total_params(&lora_model);

        assert!(lora_params > 0, "Should have LoRA params");
        assert!(
            lora_params < total_params,
            "LoRA params ({lora_params}) should be less than total ({total_params})"
        );
    }
}
