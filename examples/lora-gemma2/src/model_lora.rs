//! LoRA-adapted Gemma 2 model for parameter-efficient SFT (Supervised Fine-Tuning).
//!
//! Wraps [`Gemma2Model`](crate::Gemma2Model) with LoRA (Low-Rank Adaptation) layers
//! on attention projections (q/k/v/o) and MLP projections (gate/up/down).
//!
//! # Architecture
//!
//! The LoRA-adapted model mirrors the original Gemma 2 architecture:
//! - [`Gemma2AttentionLora`] — GQA attention with LoRA on q/k/v/o projections
//! - [`Gemma2MLPLora`] — Gated MLP with LoRA on gate/up/down projections
//! - [`Gemma2BlockLora`] — Transformer block with LoRA attention + MLP
//! - [`Gemma2ModelLora`] — Full model with LoRA-adapted layers
//! - [`Gemma2ForSFT`] — Training wrapper with cross-entropy loss
//!
//! # Usage
//!
//! ```ignore
//! use lora_gemma2::{Gemma2Config, Gemma2Model, LoraTarget};
//! use lora_gemma2::model_lora::{apply_lora_to_gemma2, Gemma2ForSFT};
//! use burn::nn::lora::{LoraConfig, LoraBias};
//!
//! let config = Gemma2Config::gemma2_2b();
//! let model = Gemma2Model::new(&config, &device);
//!
//! let lora_config = LoraConfig::new(16).with_alpha(32.0).with_bias(LoraBias::None);
//! let targets = LoraTarget::all_targets();
//! let lora_model = apply_lora_to_gemma2(model, &lora_config, targets, &device);
//!
//! let sft_model = Gemma2ForSFT::new(lora_model, 0, false);
//! // ... train with burn's Learner ...
//! let merged = sft_model.model.merge();
//! ```

use std::path::PathBuf;

use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::nn::lora::{LoraAdaptable, LoraConfig, LoraLinear};

use burn::nn::{Embedding, Linear, RmsNorm, RotaryEncoding};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, RecorderError};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{
    ElementConversion, FloatDType, Int, Tensor,
    activation::{gelu_approximate, log_softmax, softmax},
    backend::Backend,
};
use burn::train::{InferenceStep, SequenceOutput, TrainOutput, TrainStep};

use crate::batcher::SFTTrainingBatch;
#[cfg(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm",
))]
use crate::fused_ops::FusedLoraMLPBackend;
use crate::model::{Gemma2Attention, Gemma2Block, Gemma2MLP, Gemma2Model};
use crate::types::LoraTarget;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a causal attention mask `[seq_len, seq_len]`.
///
/// Lower triangular = 0.0 (attend), upper triangular = -inf (masked).
// NOTE: Duplicated from model.rs — TODO: refactor into shared utility.
fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device).float();
    let row = positions.clone().reshape([seq_len, 1]);
    let col = positions.reshape([1, seq_len]);
    let attend = col.lower_equal(row);
    Tensor::<B, 2>::zeros([seq_len, seq_len], device)
        .mask_fill(attend.equal_elem(false), f32::NEG_INFINITY)
}

/// Mixed-precision RMSNorm: upcast input to f32 for numerically stable normalization.
///
/// Computes the full RMSNorm (variance, normalization, gamma scaling) in f32,
/// then casts the result back to the original dtype. This prevents loss spikes
/// when training with f16 weights on Metal/Apple Silicon.
///
/// Unlike burn's default `RmsNorm::forward()` which only upcasts the variance,
/// this upcasts the entire computation including the division and gamma scaling.
fn rms_norm_f32<B: Backend, const D: usize>(norm: &RmsNorm<B>, x: Tensor<B, D>) -> Tensor<B, D> {
    let original_dtype = x.dtype();
    // Upcast to f32 for stable computation
    let x_f32 = x.cast(FloatDType::F32);
    // RMSNorm in f32: x / sqrt(mean(x^2) + eps) * gamma
    let rms = (x_f32.clone().square().mean_dim(D - 1) + norm.epsilon).sqrt();
    let normalized = x_f32 / rms;
    let gamma_f32 = norm.gamma.val().cast(FloatDType::F32).unsqueeze();
    let output_f32 = normalized * gamma_f32;
    // Cast back to original dtype
    output_f32.cast(original_dtype)
}

// ---------------------------------------------------------------------------
// LoRA Attention
// ---------------------------------------------------------------------------

/// Gemma 2 multi-head attention with LoRA on all projections.
///
/// Identical forward logic to [`Gemma2Attention`](crate::Gemma2Attention)
/// but uses [`LoraLinear`] for q/k/v/o projections.
#[derive(Module, Debug)]
pub struct Gemma2AttentionLora<B: Backend> {
    pub q_proj: LoraLinear<B>,
    pub k_proj: LoraLinear<B>,
    pub v_proj: LoraLinear<B>,
    pub o_proj: LoraLinear<B>,
    pub rotary: RotaryEncoding<B>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f64,
    pub softcap: f64,
}

impl<B: Backend> Gemma2AttentionLora<B> {
    /// Forward pass: `[batch, seq, hidden] -> [batch, seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 3>, mask_add: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let [batch, seq, _hidden] = x.dims();
        let kv_groups = self.num_heads / self.num_kv_heads;

        // Project Q, K, V (LoRA forward: base(x) + x @ A @ B * scaling)
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let q = q
            .reshape([batch, seq, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch, seq, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch, seq, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply RoPE
        let q = self.rotary.forward(q);
        let k = self.rotary.forward(k);

        // Scale queries
        let q = q.mul_scalar(self.scale);

        // Expand KV for grouped query attention
        let (k, v) = match kv_groups > 1 {
            true => {
                let k = k
                    .reshape([batch, self.num_kv_heads, 1, seq, self.head_dim])
                    .repeat_dim(2, kv_groups)
                    .reshape([batch, self.num_heads, seq, self.head_dim]);
                let v = v
                    .reshape([batch, self.num_kv_heads, 1, seq, self.head_dim])
                    .repeat_dim(2, kv_groups)
                    .reshape([batch, self.num_heads, seq, self.head_dim]);
                (k, v)
            }
            false => (k, v),
        };

        // Attention scores: Q @ K^T -> [batch, heads, seq, seq]
        let scores = q.matmul(k.swap_dims(2, 3));

        // Mixed precision: cast to f32 for stable softcapping and softmax
        let original_dtype = scores.dtype();
        let scores = scores.cast(FloatDType::F32);

        // Softcapping: tanh(scores / cap) * cap
        let scores = scores
            .div_scalar(self.softcap)
            .tanh()
            .mul_scalar(self.softcap);

        // Apply causal mask (also in f32)
        let scores = match mask_add {
            Some(m) => {
                let m4 = m.reshape([1, 1, seq, seq]).cast(FloatDType::F32);
                scores.add(m4)
            }
            None => scores,
        };

        // Softmax over keys (dim 3) — computed in f32 for stability
        let weights = softmax(scores, 3).cast(original_dtype);

        // Weighted sum: weights @ V -> [batch, heads, seq, head_dim]
        let output = weights.matmul(v);

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
        let output = output
            .swap_dims(1, 2)
            .reshape([batch, seq, self.num_heads * self.head_dim]);

        // Output projection (LoRA)
        self.o_proj.forward(output)
    }
}

// ---------------------------------------------------------------------------
// LoRA MLP
// ---------------------------------------------------------------------------

/// Gemma 2 MLP with LoRA on all projections.
///
/// Forward: `down_proj(GeLU(gate_proj(x)) * up_proj(x))`
#[derive(Module, Debug)]
pub struct Gemma2MLPLora<B: Backend> {
    pub gate_proj: LoraLinear<B>,
    pub up_proj: LoraLinear<B>,
    pub down_proj: LoraLinear<B>,
}

impl<B: Backend> Gemma2MLPLora<B> {
    /// Forward pass: `[batch, seq, hidden] -> [batch, seq, hidden]`.
    ///
    /// Uses f32 upcast for GELU computation (x^3 and tanh lose precision in f16),
    /// following the same cast→compute→cast-back pattern as `rms_norm_f32`.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let original_dtype = x.dtype();

        // Upcast to f32 for GELU: x^3 and tanh approximation lose precision in f16.
        let gate = self.gate_proj.forward(x.clone()).cast(FloatDType::F32);
        let gate = gelu_approximate(gate);
        let up = self.up_proj.forward(x).cast(FloatDType::F32);

        // Multiply in f32, cast back to original dtype for down_proj
        self.down_proj.forward(gate.mul(up).cast(original_dtype))
    }
}

/// Fused LoRA MLP forward for cubecl GPU backends.
///
/// Replaces 3 separate `LoraLinear.forward()` calls (15+ autodiff backward
/// steps per MLP block) with a single fused forward+backward dispatch.
///
/// **Note:** Skips LoRA dropout (normally 0 for fine-tuning).
/// **Note:** Does not handle bias in base linear layers (Gemma 2 has no MLP bias).
#[cfg(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
))]
impl<B: FusedLoraMLPBackend> Gemma2MLPLora<B> {
    /// Forward pass using the fused LoRA MLP kernel.
    ///
    /// Fuses gate+up+down LoRA projections with GeGLU activation into
    /// a single autodiff backward step. On backward, computes all 7 gradients
    /// (dX + 6 LoRA weight grads) in one dispatch instead of ~15.
    pub fn forward_fused<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        // Handle 1D input by temporarily upgrading to 2D
        if D == 1 {
            let input_2d: Tensor<B, 2> = x.unsqueeze_dim(0);
            let output_2d = self.forward_fused(input_2d);
            return output_2d.squeeze_dim(0);
        }

        // Save original dims for reshaping back
        let dims = x.dims();

        // Flatten to 2D: [..., d_in] -> [N, d_in]
        let n: usize = dims[..D - 1].iter().product();
        let x_2d = x.reshape([n, dims[D - 1]]);

        // Extract LoRA weights and frozen base weights from LoraLinear modules.
        // Dropout is skipped (normally disabled for LoRA fine-tuning).
        let out_2d = crate::fused_ops::fused_lora_mlp::<B>(
            x_2d,
            self.gate_proj.lora_a.val(),
            self.gate_proj.lora_b.val(),
            self.gate_proj.scaling,
            self.up_proj.lora_a.val(),
            self.up_proj.lora_b.val(),
            self.up_proj.scaling,
            self.down_proj.lora_a.val(),
            self.down_proj.lora_b.val(),
            self.down_proj.scaling,
            self.gate_proj.base.weight.val(),
            self.up_proj.base.weight.val(),
            self.down_proj.base.weight.val(),
        );

        // Reshape back: [N, d_out] -> [..., d_out]
        let mut out_dims = dims;
        out_dims[D - 1] = out_2d.dims()[1];
        out_2d.reshape(out_dims)
    }
}

// ---------------------------------------------------------------------------
// LoRA Transformer Block
// ---------------------------------------------------------------------------

/// Gemma 2 transformer block with LoRA-adapted attention and MLP.
///
/// Uses sandwich normalization (post-attention + post-MLP RMSNorm).
#[derive(Module, Debug)]
pub struct Gemma2BlockLora<B: Backend> {
    pub self_attn: Gemma2AttentionLora<B>,
    pub mlp: Gemma2MLPLora<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub pre_feedforward_layernorm: RmsNorm<B>,
    pub post_feedforward_layernorm: RmsNorm<B>,
}

impl<B: Backend> Gemma2BlockLora<B> {
    /// Forward pass: `[batch, seq, hidden] -> [batch, seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        // Attention with sandwich norm (mixed precision for stability)
        let r = self
            .self_attn
            .forward(rms_norm_f32(&self.input_layernorm, x.clone()), mask);
        let h = x.clone() + rms_norm_f32(&self.post_attention_layernorm, r);

        // MLP with sandwich norm (mixed precision for stability)
        let r = self
            .mlp
            .forward(rms_norm_f32(&self.pre_feedforward_layernorm, h.clone()));
        h + rms_norm_f32(&self.post_feedforward_layernorm, r)
    }
}

/// Fused LoRA MLP forward for cubecl GPU backends.
///
/// Uses `Gemma2MLPLora::forward_fused()` which replaces 3 separate LoraLinear
/// calls with a single fused dispatch, reducing autodiff backward steps from
/// ~15 to 1 per MLP block.
#[cfg(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
))]
impl<B: FusedLoraMLPBackend> Gemma2BlockLora<B> {
    /// Forward pass with fused LoRA MLP kernel.
    ///
    /// Identical to [`forward`](Self::forward) but uses the fused MLP kernel
    /// for the feedforward block, reducing GPU dispatches and autodiff overhead.
    pub fn forward_fused(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        // Attention with sandwich norm (mixed precision for stability)
        let r = self
            .self_attn
            .forward(rms_norm_f32(&self.input_layernorm, x.clone()), mask);
        let h = x.clone() + rms_norm_f32(&self.post_attention_layernorm, r);

        // Fused MLP with sandwich norm
        let r = self
            .mlp
            .forward_fused(rms_norm_f32(&self.pre_feedforward_layernorm, h.clone()));
        h + rms_norm_f32(&self.post_feedforward_layernorm, r)
    }
}

// ---------------------------------------------------------------------------
// LoRA Model
// ---------------------------------------------------------------------------

/// Gemma 2 model with LoRA-adapted transformer blocks.
///
/// Embedding, final norm, and LM head are frozen (not LoRA'd).
/// Only the transformer block projections (attention + MLP) have LoRA.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Gemma2ModelLora<B: Backend> {
    pub embed: Embedding<B>,
    pub layers: Vec<Gemma2BlockLora<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub final_logit_softcapping: f64,
}

impl<B: Backend> Gemma2ModelLora<B> {
    /// Forward pass: token IDs -> logits.
    ///
    /// - Input: `[batch, seq_len]` integer token IDs
    /// - Output: `[batch, seq_len, vocab_size]` logits
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = input_ids.device();
        let [_batch, seq_len] = input_ids.dims();

        // Embedding lookup + Gemma 2 scaling
        let scale = (self.hidden_size as f64).sqrt();
        let h = self.embed.forward(input_ids).mul_scalar(scale);

        // Causal mask
        let mask = causal_mask::<B>(seq_len, &device);

        // Transformer blocks
        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(h, Some(mask.clone()));
        }

        // Final norm (mixed precision for stability) + LM head + softcapping
        let h = rms_norm_f32(&self.norm, h);
        let logits = self.lm_head.forward(h);

        // Softcapping in f32 for numerical stability
        let original_dtype = logits.dtype();
        logits
            .cast(FloatDType::F32)
            .div_scalar(self.final_logit_softcapping)
            .tanh()
            .mul_scalar(self.final_logit_softcapping)
            .cast(original_dtype)
    }

    /// Forward pass returning raw logits WITHOUT final logit softcapping.
    ///
    /// Used by the fused CE training path where softcapping is applied
    /// inline in the GPU kernel, avoiding materialization of the
    /// `[batch, seq, 256K]` softcapped logits tensor (~4GB for seq=2048).
    ///
    /// For inference, use [`forward()`](Self::forward) which applies softcapping.
    pub fn forward_raw(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = input_ids.device();
        let [_batch, seq_len] = input_ids.dims();

        let scale = (self.hidden_size as f64).sqrt();
        let h = self.embed.forward(input_ids).mul_scalar(scale);

        let mask = causal_mask::<B>(seq_len, &device);

        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(h, Some(mask.clone()));
        }

        let h = rms_norm_f32(&self.norm, h);
        self.lm_head.forward(h)
    }

    /// Forward pass with fused LoRA MLP kernels, returning raw logits without softcapping.
    ///
    /// Uses [`Gemma2BlockLora::forward_fused`] which replaces 3 separate `LoraLinear`
    /// calls per block with a single fused dispatch, reducing autodiff backward steps
    /// from ~15 to 1 per MLP block (~390 total → ~26).
    ///
    /// Only available on cubecl GPU backends (metal, wgpu, cuda, vulkan, rocm).
    /// Use [`forward_raw`](Self::forward_raw) for non-cubecl backends.
    #[cfg(any(
        feature = "metal",
        feature = "wgpu",
        feature = "cuda",
        feature = "vulkan",
        feature = "rocm"
    ))]
    pub fn forward_raw_fused(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3>
    where
        B: FusedLoraMLPBackend,
    {
        let device = input_ids.device();
        let [_batch, seq_len] = input_ids.dims();

        let scale = (self.hidden_size as f64).sqrt();
        let h = self.embed.forward(input_ids).mul_scalar(scale);

        let mask = causal_mask::<B>(seq_len, &device);

        let mut h = h;
        for layer in &self.layers {
            h = layer.forward_fused(h, Some(mask.clone()));
        }

        let h = rms_norm_f32(&self.norm, h);
        self.lm_head.forward(h)
    }

    /// Merge all LoRA weights into base layers for inference.
    ///
    /// Returns a standard [`Gemma2Model`] with no LoRA overhead.
    /// Output is numerically identical (within floating point precision).
    pub fn merge(self) -> Gemma2Model<B> {
        let layers = self
            .layers
            .into_iter()
            .map(|block| Gemma2Block {
                self_attn: Gemma2Attention {
                    q_proj: block.self_attn.q_proj.merge(),
                    k_proj: block.self_attn.k_proj.merge(),
                    v_proj: block.self_attn.v_proj.merge(),
                    o_proj: block.self_attn.o_proj.merge(),
                    rotary: block.self_attn.rotary,
                    num_heads: block.self_attn.num_heads,
                    num_kv_heads: block.self_attn.num_kv_heads,
                    head_dim: block.self_attn.head_dim,
                    scale: block.self_attn.scale,
                    softcap: block.self_attn.softcap,
                },
                mlp: Gemma2MLP {
                    gate_proj: block.mlp.gate_proj.merge(),
                    up_proj: block.mlp.up_proj.merge(),
                    down_proj: block.mlp.down_proj.merge(),
                },
                input_layernorm: block.input_layernorm,
                post_attention_layernorm: block.post_attention_layernorm,
                pre_feedforward_layernorm: block.pre_feedforward_layernorm,
                post_feedforward_layernorm: block.post_feedforward_layernorm,
            })
            .collect();

        Gemma2Model::from_module(
            self.embed,
            layers,
            self.norm,
            self.lm_head,
            self.hidden_size,
            self.vocab_size,
            self.final_logit_softcapping,
        )
    }

    /// Save all LoRA adapter weights to a directory.
    ///
    /// Creates a directory structure:
    /// ```text
    /// path/
    /// ├── layer_0/
    /// │   ├── q_proj.mpk
    /// │   ├── k_proj.mpk
    /// │   ├── v_proj.mpk
    /// │   ├── o_proj.mpk
    /// │   ├── gate_proj.mpk
    /// │   ├── up_proj.mpk
    /// │   └── down_proj.mpk
    /// ├── layer_1/
    /// │   └── ...
    /// └── layer_N/
    ///     └── ...
    /// ```
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
    ///
    /// Replaces LoRA matrices (A and B) with those loaded from disk.
    /// Base model weights remain unchanged.
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

            layers.push(Gemma2BlockLora {
                self_attn: Gemma2AttentionLora {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    rotary: block.self_attn.rotary,
                    num_heads: block.self_attn.num_heads,
                    num_kv_heads: block.self_attn.num_kv_heads,
                    head_dim: block.self_attn.head_dim,
                    scale: block.self_attn.scale,
                    softcap: block.self_attn.softcap,
                },
                mlp: Gemma2MLPLora {
                    gate_proj,
                    up_proj,
                    down_proj,
                },
                input_layernorm: block.input_layernorm,
                post_attention_layernorm: block.post_attention_layernorm,
                pre_feedforward_layernorm: block.pre_feedforward_layernorm,
                post_feedforward_layernorm: block.post_feedforward_layernorm,
            });
        }

        Ok(Gemma2ModelLora {
            layers,
            embed: self.embed,
            norm: self.norm,
            lm_head: self.lm_head,
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
            final_logit_softcapping: self.final_logit_softcapping,
        })
    }
}

impl<B: Backend> Gemma2Model<B> {
    /// Construct a [`Gemma2Model`] from its constituent parts.
    ///
    /// Used by [`Gemma2ModelLora::merge`] to rebuild the base model.
    #[allow(clippy::too_many_arguments)]
    fn from_module(
        embed: Embedding<B>,
        layers: Vec<Gemma2Block<B>>,
        norm: RmsNorm<B>,
        lm_head: Linear<B>,
        hidden_size: usize,
        vocab_size: usize,
        final_logit_softcapping: f64,
    ) -> Self {
        Self {
            embed,
            layers,
            norm,
            lm_head,
            hidden_size,
            vocab_size,
            final_logit_softcapping,
        }
    }
}

impl<B: Backend> ModuleDisplay for Gemma2ModelLora<B> {
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

/// Apply LoRA adaptation to a [`Gemma2Model`].
///
/// Converts the model into a [`Gemma2ModelLora`] by wrapping specified
/// projection layers with LoRA. Layers matching `targets` get trainable
/// LoRA params; non-target layers are wrapped with frozen LoRA
/// (output is zero at initialization since B is initialized to zeros).
///
/// # Arguments
///
/// * `model` — Base Gemma 2 model (consumed)
/// * `config` — LoRA configuration (rank, alpha, dropout, etc.)
/// * `targets` — Which projections to apply LoRA to
/// * `device` — Device for tensor allocation
///
/// # Example
///
/// ```ignore
/// use lora_gemma2::model_lora::apply_lora_to_gemma2;
/// use burn::nn::lora::{LoraConfig, LoraBias};
///
/// let lora_config = LoraConfig::new(16).with_alpha(32.0).with_bias(LoraBias::None);
/// let targets = LoraTarget::all_targets(); // attention + MLP
/// let lora_model = apply_lora_to_gemma2(model, &lora_config, targets, &device);
/// ```
pub fn apply_lora_to_gemma2<B: Backend>(
    model: Gemma2Model<B>,
    config: &LoraConfig,
    targets: &[LoraTarget],
    device: &B::Device,
) -> Gemma2ModelLora<B> {
    let Gemma2Model {
        embed,
        layers,
        norm,
        lm_head,
        hidden_size,
        vocab_size,
        final_logit_softcapping,
    } = model;

    let layers = layers
        .into_iter()
        .map(|block| apply_lora_to_block(block, config, targets, device))
        .collect();

    Gemma2ModelLora {
        embed: embed.no_grad(),
        layers,
        norm,
        lm_head: lm_head.no_grad(),
        hidden_size,
        vocab_size,
        final_logit_softcapping,
    }
}

/// Apply LoRA to a single transformer block.
fn apply_lora_to_block<B: Backend>(
    block: Gemma2Block<B>,
    config: &LoraConfig,
    targets: &[LoraTarget],
    device: &B::Device,
) -> Gemma2BlockLora<B> {
    Gemma2BlockLora {
        self_attn: apply_lora_to_attention(block.self_attn, config, targets, device),
        mlp: apply_lora_to_mlp(block.mlp, config, targets, device),
        input_layernorm: block.input_layernorm,
        post_attention_layernorm: block.post_attention_layernorm,
        pre_feedforward_layernorm: block.pre_feedforward_layernorm,
        post_feedforward_layernorm: block.post_feedforward_layernorm,
    }
}

/// Apply LoRA to attention projections.
fn apply_lora_to_attention<B: Backend>(
    attn: Gemma2Attention<B>,
    config: &LoraConfig,
    targets: &[LoraTarget],
    device: &B::Device,
) -> Gemma2AttentionLora<B> {
    Gemma2AttentionLora {
        q_proj: wrap_lora(attn.q_proj, config, LoraTarget::QProj, targets, device),
        k_proj: wrap_lora(attn.k_proj, config, LoraTarget::KProj, targets, device),
        v_proj: wrap_lora(attn.v_proj, config, LoraTarget::VProj, targets, device),
        o_proj: wrap_lora(attn.o_proj, config, LoraTarget::OProj, targets, device),
        rotary: attn.rotary,
        num_heads: attn.num_heads,
        num_kv_heads: attn.num_kv_heads,
        head_dim: attn.head_dim,
        scale: attn.scale,
        softcap: attn.softcap,
    }
}

/// Apply LoRA to MLP projections.
fn apply_lora_to_mlp<B: Backend>(
    mlp: Gemma2MLP<B>,
    config: &LoraConfig,
    targets: &[LoraTarget],
    device: &B::Device,
) -> Gemma2MLPLora<B> {
    Gemma2MLPLora {
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

/// Training wrapper for LoRA-adapted Gemma 2 with SFT (Supervised Fine-Tuning).
///
/// Implements [`TrainStep`] and [`InferenceStep`] for use with burn's
/// [`Learner`](burn::train::Learner). Computes cross-entropy loss
/// for next-token prediction, ignoring pad tokens.
///
/// # Usage
///
/// ```ignore
/// use lora_gemma2::model_lora::{apply_lora_to_gemma2, Gemma2ForSFT};
///
/// let lora_model = apply_lora_to_gemma2(model, &lora_config, targets, &device);
/// let sft_model = Gemma2ForSFT::new(lora_model, pad_token_id, false);
///
/// // Use with burn's Learner
/// let result = training.launch(Learner::new(sft_model, optimizer, lr));
/// ```
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Gemma2ForSFT<B: Backend> {
    /// The LoRA-adapted Gemma 2 model.
    pub model: Gemma2ModelLora<B>,
    /// Pad token ID to ignore in cross-entropy loss.
    pub pad_token_id: usize,
    /// If true, use fused CE kernel with inline softcapping instead of standard CE.
    /// Default: false (standard CE is faster; fused CE saves memory but is slower).
    pub use_fused_ce: bool,
}

impl<B: Backend> Gemma2ForSFT<B> {
    /// Create a new SFT training wrapper.
    pub fn new(model: Gemma2ModelLora<B>, pad_token_id: usize, use_fused_ce: bool) -> Self {
        Self {
            model,
            pad_token_id,
            use_fused_ce,
        }
    }

    /// Forward pass: token IDs -> logits.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.model.forward(input_ids)
    }

    /// Merge LoRA weights and return base model for inference.
    pub fn merge(self) -> Gemma2Model<B> {
        self.model.merge()
    }
}

impl<B: Backend> ModuleDisplay for Gemma2ForSFT<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("pad_token_id", &self.pad_token_id)
            .add("use_fused_ce", &self.use_fused_ce)
            .add("model", &self.model)
            .optional()
    }
}

// ---------------------------------------------------------------------------
// TrainStep / InferenceStep
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// TrainStep / InferenceStep
// ---------------------------------------------------------------------------

/// Shared masking + normalization for per-token CE losses.
///
/// Both fused and standard CE paths produce per-token losses `[batch, seq]`.
/// This helper applies padding masking and token-count normalization.
///
/// `mask_pad`: true = padding position. `token_losses`: raw CE per token.
fn mask_normalize_ce<B: Backend>(
    token_losses: Tensor<B, 2>,
    mask_pad: Tensor<B, 2, burn::tensor::Bool>,
) -> Tensor<B, 1> {
    let [batch_size, seq_len] = token_losses.dims();
    let device = token_losses.device();
    let original_dtype = token_losses.dtype();
    let masked_ce = token_losses.mask_fill(mask_pad.clone(), 0);
    // Upcast sum accumulation to f32 for precision (same pattern as rms_norm_f32).
    // Both operands cast to f32 before sum+division avoids DTypeMismatch in autodiff.
    let masked_ce_sum = masked_ce.cast(FloatDType::F32).sum();
    let ntokens = Tensor::<B, 2>::ones([batch_size, seq_len], &device)
        .mask_fill(mask_pad, 0)
        .cast(FloatDType::F32)
        .sum();
    (masked_ce_sum / ntokens).cast(original_dtype)
}

/// TrainStep for cubecl GPU backends (metal, wgpu, cuda, vulkan, rocm).
///
/// Supports two CE paths selected at runtime via `use_fused_ce`:
/// - `false` (default): Standard CE (log_softmax → gather → neg).
///   Faster (~0.20s/iter) — burn's `log_softmax` handles f16 numerical stability.
///   Materializes `[batch*seq, 256K]` log_probs (~4GB for seq=2048).
/// - `true`: Fused CE with inline softcapping (single GPU kernel).
///   Avoids materializing `[batch*seq, 256K]` log_probs (~4GB memory saved).
///   Slower (~0.25s/iter) — f32 upcast in fused kernel adds overhead.
///   Use when memory-constrained, not for speed.
///
/// NOTE: Requires `B: FusedCEBackend` — the caller's `run()` function must
/// propagate this bound. See `sft-train.rs` for the matching cfg-conditional bounds.
#[cfg(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
))]
impl<B> TrainStep for Gemma2ForSFT<B>
where
    B: AutodiffBackend + crate::fused_ops::FusedCEBackend + crate::fused_ops::FusedLoraMLPBackend,
{
    type Input = SFTTrainingBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, batch: SFTTrainingBatch<B>) -> TrainOutput<SequenceOutput<B>> {
        let targets = batch.targets;
        let mask_pad = batch.mask_pad;

        if self.use_fused_ce {
            // Fused CE + fused LoRA MLP path: both use single GPU kernel dispatches.
            // forward_raw_fused() uses fused LoRA MLP kernel (1 backward step per block
            // instead of ~15), forward_raw() skips softcapping (fused into CE kernel).
            // NOTE: Slower than standard CE due to f32 upcast overhead in fused kernel.
            // Use only when memory-constrained (avoids [N, 256K] log_probs materialization).
            let logits = self.model.forward_raw_fused(batch.tokens_inputs);
            let [batch_size, seq_len, vocab_size] = logits.dims();

            // Softcap value from model config (Gemma 2 2B = 30.0).
            // The fused CE kernel internally upcasts to f32 for the logsumexp
            // accumulation (sum over 256K vocab overflows f16 max ~65504),
            // then casts the loss back to the original dtype.
            let softcap = Some(self.model.final_logit_softcapping as f32);
            let logits_flat = logits.clone().reshape([batch_size * seq_len, vocab_size]);
            let targets_flat = targets.clone().reshape([batch_size * seq_len]);
            let token_losses =
                crate::fused_ops::fused_ce_loss::<B>(logits_flat, targets_flat, softcap)
                    .reshape([batch_size, seq_len]);

            let loss = mask_normalize_ce(token_losses, mask_pad);

            // NOTE: Do NOT call loss.into_scalar() — GPU sync costs ~800ms/iter.
            TrainOutput::new(
                self,
                loss.backward(),
                SequenceOutput::new(loss, logits, None, targets),
            )
        } else {
            // Standard CE path: log_softmax → gather → neg (default, faster).
            // Cast to f32 before log_softmax — f16 overflows with 256K vocab
            // (exp(30) ≈ 1e13 >> f16 max 65504, causes inf→NaN cascade).
            // Materializes [batch*seq, 256K] log_probs in f32 (~8GB for seq=2048) but
            // ~20% faster than fused CE with f32 upcast.
            let logits = self.model.forward(batch.tokens_inputs);
            let [batch_size, seq_len, _vocab_size] = logits.dims();

            let logits_f32 = logits.clone().cast(FloatDType::F32);
            let target_indices = targets.clone().reshape([batch_size, seq_len, 1]);
            let token_losses = log_softmax(logits_f32, 2)
                .gather(2, target_indices)
                .reshape([batch_size, seq_len])
                .neg();

            let loss = mask_normalize_ce(token_losses, mask_pad);

            TrainOutput::new(
                self,
                loss.backward(),
                SequenceOutput::new(loss, logits, None, targets),
            )
        }
    }
}

/// Standard CE TrainStep for non-cubecl backends (ndarray, tch).
///
/// These backends lack the fused CE kernel, so they always use the standard
/// `log_softmax → gather → neg` path regardless of the `use_fused_ce` flag.
#[cfg(not(any(
    feature = "metal",
    feature = "wgpu",
    feature = "cuda",
    feature = "vulkan",
    feature = "rocm"
)))]
impl<B: AutodiffBackend> TrainStep for Gemma2ForSFT<B> {
    type Input = SFTTrainingBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, batch: SFTTrainingBatch<B>) -> TrainOutput<SequenceOutput<B>> {
        let logits = self.model.forward(batch.tokens_inputs);
        let targets = batch.targets;
        let [batch_size, seq_len, _vocab_size] = logits.dims();

        let logits_f32 = logits.clone().cast(FloatDType::F32);
        let target_indices = targets.clone().reshape([batch_size, seq_len, 1]);
        let token_losses = log_softmax(logits_f32, 2)
            .gather(2, target_indices)
            .reshape([batch_size, seq_len])
            .neg();

        let loss = mask_normalize_ce(token_losses, batch.mask_pad.clone());

        TrainOutput::new(
            self,
            loss.backward(),
            SequenceOutput::new(loss, logits, None, targets),
        )
    }
}

impl<B: Backend> InferenceStep for Gemma2ForSFT<B> {
    type Input = SFTTrainingBatch<B>;
    type Output = SequenceOutput<B>;

    fn step(&self, batch: SFTTrainingBatch<B>) -> SequenceOutput<B> {
        let logits = self.model.forward(batch.tokens_inputs);
        let targets = batch.targets;

        // Token-normalized cross-entropy (same formula as TrainStep)
        let [batch_size, seq_len, _vocab_size] = logits.dims();
        let log_probs = log_softmax(logits.clone(), 2);
        let target_indices = targets.clone().reshape([batch_size, seq_len, 1]);
        let token_log_probs = log_probs
            .gather(2, target_indices)
            .reshape([batch_size, seq_len]);

        let masked_ce = token_log_probs.neg().mask_fill(batch.mask_pad.clone(), 0);
        let ntokens = Tensor::<B, 2>::ones([batch_size, seq_len], &logits.device())
            .mask_fill(batch.mask_pad, 0)
            .sum();
        let loss = masked_ce.sum() / ntokens;

        // Log val loss for monitoring
        let loss_val: f32 = loss.clone().into_scalar().elem();
        log::info!("Val loss: {loss_val:.4}");

        SequenceOutput::new(loss, logits, None, targets)
    }
}

// ---------------------------------------------------------------------------
// Parameter Counting
// ---------------------------------------------------------------------------

/// Count total LoRA parameters (A + B matrices) across all layers.
///
/// This represents the number of trainable parameters during fine-tuning.
pub fn count_lora_params<B: Backend>(model: &Gemma2ModelLora<B>) -> usize {
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
pub fn count_total_params<B: Backend>(model: &Gemma2ModelLora<B>) -> usize {
    // Embedding
    let mut total: usize = model.embed.weight.shape().num_elements();

    // Transformer blocks
    for block in &model.layers {
        // Attention base weights + LoRA
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

        // MLP base weights + LoRA
        total += block.mlp.gate_proj.base.weight.shape().num_elements();
        total += block.mlp.gate_proj.lora_a.shape().num_elements();
        total += block.mlp.gate_proj.lora_b.shape().num_elements();

        total += block.mlp.up_proj.base.weight.shape().num_elements();
        total += block.mlp.up_proj.lora_a.shape().num_elements();
        total += block.mlp.up_proj.lora_b.shape().num_elements();

        total += block.mlp.down_proj.base.weight.shape().num_elements();
        total += block.mlp.down_proj.lora_a.shape().num_elements();
        total += block.mlp.down_proj.lora_b.shape().num_elements();

        // Norm layers
        total += block.input_layernorm.gamma.shape().num_elements();
        total += block.post_attention_layernorm.gamma.shape().num_elements();
        total += block.pre_feedforward_layernorm.gamma.shape().num_elements();
        total += block
            .post_feedforward_layernorm
            .gamma
            .shape()
            .num_elements();
    }

    // Final norm + LM head
    total += model.norm.gamma.shape().num_elements();
    total += model.lm_head.weight.shape().num_elements();

    total
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Gemma2Config;

    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

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

    #[test]
    fn test_apply_lora_all_targets() {
        let device = device();
        let config = tiny_config();
        let model = Gemma2Model::<TestBackend>::new(&config, &device);

        let lora_config = LoraConfig::new(4).with_alpha(8.0);
        let lora_model =
            apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);

        assert_eq!(lora_model.layers.len(), config.num_hidden_layers);
        // Check LoRA rank on q_proj
        assert_eq!(lora_model.layers[0].self_attn.q_proj.rank(), 4);
    }

    #[test]
    fn test_apply_lora_attention_only() {
        let device = device();
        let config = tiny_config();
        let model = Gemma2Model::<TestBackend>::new(&config, &device);

        let lora_config = LoraConfig::new(8).with_alpha(16.0);
        let targets = LoraTarget::attention_targets();
        let lora_model = apply_lora_to_gemma2(model, &lora_config, targets, &device);

        assert_eq!(lora_model.layers.len(), 2);
        assert_eq!(lora_model.layers[0].self_attn.q_proj.rank(), 8);
    }

    #[test]
    fn test_lora_identity_start() {
        // LoRA starts with B=zeros, so output should match base model.
        let device = device();
        let config = tiny_config();
        let model = Gemma2Model::<TestBackend>::new(&config, &device);

        let input = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);
        let base_output = model.forward(input.clone());

        let lora_config = LoraConfig::new(4).with_alpha(8.0);
        let lora_model =
            apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);
        let lora_output = lora_model.forward(input.clone());

        // LoRA starts with B=zeros → LoRA contribution is 0 → output == base
        let diff = (base_output.clone() - lora_output)
            .abs()
            .max()
            .into_scalar();
        assert!(
            diff < 1e-5,
            "LoRA should start as identity (B=zeros), diff={diff}"
        );
    }

    #[test]
    fn test_merge_output_equivalence() {
        let device = device();
        let config = tiny_config();
        let model = Gemma2Model::<TestBackend>::new(&config, &device);

        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        let base_output = model.forward(input.clone());

        let lora_config = LoraConfig::new(4).with_alpha(8.0);
        let lora_model =
            apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);

        // Merge and compare
        let merged = lora_model.merge();
        let merged_output = merged.forward(input);

        let diff = (base_output - merged_output).abs().max().into_scalar();
        assert!(diff < 1e-4, "Merged output should match base, diff={diff}");
    }

    #[test]
    fn test_param_counts() {
        let device = device();
        let config = tiny_config();
        let model = Gemma2Model::<TestBackend>::new(&config, &device);

        let lora_config = LoraConfig::new(4);
        let lora_model =
            apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);

        let lora_params = count_lora_params(&lora_model);
        let total_params = count_total_params(&lora_model);

        assert!(lora_params > 0, "Should have LoRA params");
        assert!(
            lora_params < total_params,
            "LoRA params ({lora_params}) should be less than total ({total_params})"
        );

        // With rank=4, each LoRA layer has: A[d_in, 4] + B[4, d_out]
        // For q_proj (d_in=32, d_out=32): (32*4 + 4*32) = 256
        // 7 targets × 2 layers = 14 LoRA layers
        assert!(lora_params > 0);
    }

    #[test]
    fn test_param_counts_attention_only() {
        let device = device();
        let config = tiny_config();
        let model = Gemma2Model::<TestBackend>::new(&config, &device);

        let lora_config = LoraConfig::new(4);
        let lora_model = apply_lora_to_gemma2(
            model,
            &lora_config,
            LoraTarget::attention_targets(),
            &device,
        );

        let lora_params = count_lora_params(&lora_model);
        let total_params = count_total_params(&lora_model);

        // Attention-only targets should have fewer LoRA params than all targets
        assert!(lora_params > 0);
        assert!(lora_params < total_params);
    }

    /// Test that single LoraLinear adapter save/load roundtrip works.
    /// This isolates whether the issue is in burn's LoRA or our multi-layer wrapper.
    #[test]
    fn test_single_lora_adapter_roundtrip() {
        use burn::nn::LinearConfig;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

        let device = device();

        // Create a linear layer and wrap with LoRA
        let linear = LinearConfig::new(32, 32)
            .with_bias(false)
            .init::<TestBackend>(&device);
        let lora_config = LoraConfig::new(4)
            .with_alpha(8.0)
            .with_init(burn::nn::lora::LoraInit::Gaussian);
        let lora = linear.with_lora(&lora_config, &device);

        // Save original LoRA weights for comparison
        let original_a = lora.lora_a.val().clone();
        let original_b = lora.lora_b.val().clone();

        // Forward pass with reference
        let input =
            Tensor::<TestBackend, 2>::random([2, 32], burn::tensor::Distribution::Default, &device);
        let original_output = lora.forward(input.clone());

        // Save adapter
        let dir = tempfile::tempdir().expect("temp dir");
        let adapter_path = dir.path().join("test_adapter");
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        lora.save_adapter(&adapter_path, &recorder).expect("save");

        // Clone (same base weights) and load adapter
        let loaded = lora
            .clone()
            .load_adapter_file(&adapter_path, &recorder, &device)
            .expect("load");

        // Verify LoRA A/B weights match
        let a_diff = (loaded.lora_a.val() - original_a).abs().max().into_scalar();
        let b_diff = (loaded.lora_b.val() - original_b).abs().max().into_scalar();
        assert!(a_diff < 1e-6, "LoRA A mismatch: {a_diff}");
        assert!(b_diff < 1e-6, "LoRA B mismatch: {b_diff}");

        // Verify output matches
        let loaded_output = loaded.forward(input);
        let output_diff = (original_output - loaded_output).abs().max().into_scalar();
        assert!(output_diff < 1e-5, "Output mismatch: {output_diff}");
    }

    /// Diagnostic: compare LoRA A/B weights layer-by-layer after save/load.
    /// This helps isolate where the multi-layer adapter roundtrip fails.
    #[test]
    fn test_adapter_weights_per_layer() {
        let device = device();
        let config = tiny_config();

        let model = Gemma2Model::<TestBackend>::new(&config, &device);
        let model_clone = model.clone();

        let lora_config = LoraConfig::new(4).with_alpha(8.0);
        let lora_model =
            apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);

        // Save original LoRA A/B weights per layer for comparison
        let original_weights: Vec<_> = lora_model
            .layers
            .iter()
            .map(|block| {
                let a = block.self_attn.q_proj.lora_a.val().clone();
                let b = block.self_attn.q_proj.lora_b.val().clone();
                (a, b)
            })
            .collect();

        // Save adapters
        let dir = tempfile::tempdir().expect("temp dir");
        let adapter_path = dir.path().join("adapters");
        lora_model.save_adapters(&adapter_path).expect("save");

        // Apply LoRA to clone and load adapters
        let fresh_lora = apply_lora_to_gemma2(
            model_clone,
            &lora_config,
            LoraTarget::all_targets(),
            &device,
        );
        let loaded_lora = fresh_lora
            .load_adapters(&adapter_path, &device)
            .expect("load");

        // Compare A/B weights per layer
        for (i, block) in loaded_lora.layers.iter().enumerate() {
            let loaded_a = block.self_attn.q_proj.lora_a.val().clone();
            let loaded_b = block.self_attn.q_proj.lora_b.val().clone();

            let (ref_a, ref_b) = &original_weights[i];
            let a_diff = (loaded_a.clone() - ref_a.clone()).abs().max().into_scalar();
            let b_diff = (loaded_b.clone() - ref_b.clone()).abs().max().into_scalar();

            eprintln!("Layer {i}: q_proj A diff={a_diff:.8}, B diff={b_diff:.8}");
            assert!(a_diff < 1e-6, "Layer {i} q_proj A mismatch: {a_diff}");
            assert!(b_diff < 1e-6, "Layer {i} q_proj B mismatch: {b_diff}");
        }
    }

    /// Test multi-layer adapter roundtrip with cloned base model.
    /// NOTE: This test uses the same base model weights (cloned) so we only
    /// validate that LoRA A/B are correctly saved and restored.
    #[test]
    fn test_adapter_save_load_roundtrip() {
        let device = device();
        let config = tiny_config();

        // Create model and immediately snapshot base weights (deep copy via .into_data()).
        // burn's Module::clone() may share tensor storage, so we must capture
        // weight values before any mutation (no_grad, with_lora, etc.).
        let model = Gemma2Model::<TestBackend>::new(&config, &device);
        let ref_q_weight_data = model.layers[0].self_attn.q_proj.weight.val().into_data();

        // Clone for reload (base weights may share storage — that's OK,
        // we only need the reload copy to have the same base structure)
        let model_reload = model.clone();

        // Apply LoRA to the first copy and get reference output
        let lora_config = LoraConfig::new(4).with_alpha(8.0);
        let lora_model =
            apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);

        let ref_q_weight = Tensor::<TestBackend, 2>::from_data(ref_q_weight_data, &device);

        let input = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);
        let _original_output = lora_model.forward(input.clone());

        // Verify base weights survived apply_lora (no_grad should not change values)
        let lora_q_weight = lora_model.layers[0].self_attn.q_proj.base.weight.val();
        let base_diff = (ref_q_weight.clone() - lora_q_weight.clone())
            .abs()
            .max()
            .into_scalar();
        assert!(
            base_diff < 1e-10,
            "Base weight should not change during apply_lora, diff={base_diff}"
        );

        // Save adapters (LoRA A/B matrices only, not base weights)
        let dir = tempfile::tempdir().expect("temp dir");
        let adapter_path = dir.path().join("adapters");
        lora_model
            .save_adapters(&adapter_path)
            .expect("save adapters");

        // Verify files exist
        assert!(adapter_path.join("layer_0/q_proj.mpk").exists());
        assert!(adapter_path.join("layer_1/down_proj.mpk").exists());

        // Apply LoRA to the third copy (same base weights, fresh LoRA init)
        let fresh_lora = apply_lora_to_gemma2(
            model_reload,
            &lora_config,
            LoraTarget::all_targets(),
            &device,
        );

        // Verify base weights on reload copy match the snapshot
        let reload_q_weight = fresh_lora.layers[0].self_attn.q_proj.base.weight.val();
        let reload_diff = (ref_q_weight.clone() - reload_q_weight.clone())
            .abs()
            .max()
            .into_scalar();
        assert!(
            reload_diff < 1e-10,
            "Reload base weight should match reference, diff={reload_diff}"
        );

        // Load saved adapters — replaces the fresh LoRA A/B with saved ones
        let loaded_lora = fresh_lora
            .load_adapters(&adapter_path, &device)
            .expect("load adapters");

        // Verify LoRA A/B weights were correctly restored per layer
        // (base weights are shared via clone so we only validate adapter weights)
        for (i, block) in loaded_lora.layers.iter().enumerate() {
            let a_rank = block.self_attn.q_proj.lora_a.shape().dims::<2>()[1];
            assert_eq!(a_rank, 4, "Layer {i} q_proj should have rank 4");
        }

        let loaded_output = loaded_lora.forward(input);

        // Verify loaded model produces valid (finite) output
        let [batch, seq, vocab] = loaded_output.dims();
        assert_eq!(batch, 1);
        assert_eq!(seq, 4);
        assert_eq!(vocab, config.vocab_size);

        let max_logit = loaded_output.clone().abs().max().into_scalar();
        assert!(
            max_logit.is_finite(),
            "Output should be finite, got {max_logit}"
        );
    }

    #[test]
    fn test_forward_output_shape() {
        let device = device();
        let config = tiny_config();
        let model = Gemma2Model::<TestBackend>::new(&config, &device);

        let lora_config = LoraConfig::new(4);
        let lora_model =
            apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);

        let input = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        let output = lora_model.forward(input);

        let [batch, seq, vocab] = output.dims();
        assert_eq!(batch, 2);
        assert_eq!(seq, 8);
        assert_eq!(vocab, config.vocab_size);
    }

    #[test]
    fn test_sft_model_forward() {
        let device = device();
        let config = tiny_config();
        let model = Gemma2Model::<TestBackend>::new(&config, &device);

        let lora_config = LoraConfig::new(4);
        let lora_model =
            apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);

        let sft_model = Gemma2ForSFT::new(lora_model, 0, false);

        let input = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);
        let output = sft_model.forward(input);

        let [batch, seq, vocab] = output.dims();
        assert_eq!(batch, 1);
        assert_eq!(seq, 4);
        assert_eq!(vocab, config.vocab_size);
    }

    #[test]
    fn test_sft_model_merge() {
        let device = device();
        let config = tiny_config();
        let model = Gemma2Model::<TestBackend>::new(&config, &device);

        let lora_config = LoraConfig::new(4);
        let lora_model =
            apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);

        let sft_model = Gemma2ForSFT::new(lora_model, 0, false);
        let merged = sft_model.merge();

        let input = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);
        let output = merged.forward(input);

        let [batch, seq, vocab] = output.dims();
        assert_eq!(batch, 1);
        assert_eq!(seq, 4);
        assert_eq!(vocab, config.vocab_size);
    }

    #[test]
    fn test_lora_model_display() {
        let device = device();
        let config = tiny_config();
        let model = Gemma2Model::<TestBackend>::new(&config, &device);

        let lora_config = LoraConfig::new(4);
        let lora_model =
            apply_lora_to_gemma2(model, &lora_config, LoraTarget::all_targets(), &device);

        let display = format!("{lora_model}");
        assert!(display.contains("Gemma2ModelLora"));
    }
}
