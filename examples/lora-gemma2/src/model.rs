//! Gemma 2 model implementation for burn.
//!
//! Architecture reference: [Gemma 2 paper](https://arxiv.org/abs/2408.00118)
//! Implementation reference: [mlx-lm](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma2.py)
//!
//! Key Gemma 2 features:
//! - Grouped Query Attention (GQA)
//! - Attention logit softcapping
//! - Final logit softcapping
//! - Post-attention and post-MLP RMSNorm (sandwich architecture)
//! - RoPE (Rotary Position Embeddings)

use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding,
    RotaryEncodingConfig,
};
use burn::tensor::{
    FloatDType, Int, Tensor, activation::gelu_approximate, activation::softmax, backend::Backend,
};

use crate::types::Gemma2Config;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a causal attention mask `[seq_len, seq_len]`.
///
/// Lower triangular = 0.0 (attend), upper triangular = -inf (masked).
/// Used as additive mask before softmax.
fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    // Lower-triangular causal mask: (i,j) = 0.0 if j <= i, -inf if j > i
    let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device).float();
    let row = positions.clone().reshape([seq_len, 1]);
    let col = positions.reshape([1, seq_len]);
    // attend where col <= row (current and past tokens)
    let attend = col.lower_equal(row);
    // Start with zeros, fill future positions with -inf
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
// Attention
// ---------------------------------------------------------------------------

/// Gemma 2 multi-head attention with GQA, RoPE, and logit softcapping.
#[derive(Module, Debug)]
pub struct Gemma2Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub rotary: RotaryEncoding<B>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f64,
    pub softcap: f64,
}

impl<B: Backend> Gemma2Attention<B> {
    pub fn new(config: &Gemma2Config, device: &B::Device) -> Self {
        let q_proj = LinearConfig::new(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        )
        .with_bias(false)
        .init(device);
        let k_proj = LinearConfig::new(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
        )
        .with_bias(false)
        .init(device);
        let v_proj = LinearConfig::new(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
        )
        .with_bias(false)
        .init(device);
        let o_proj = LinearConfig::new(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
        )
        .with_bias(false)
        .init(device);

        let rotary = RotaryEncodingConfig::new(8192, config.head_dim)
            .with_theta(config.rope_theta)
            .init(device);

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            scale: config.attention_scale(),
            softcap: config.attn_logit_softcapping,
        }
    }

    /// Forward pass: `[batch, seq, hidden] -> [batch, seq, hidden]`.
    pub fn forward(&self, x: Tensor<B, 3>, mask_add: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let [batch, seq, _hidden] = x.dims();
        let kv_groups = self.num_heads / self.num_kv_heads;

        // Project Q, K, V
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
        let (k, v) = if kv_groups > 1 {
            let k = k
                .reshape([batch, self.num_kv_heads, 1, seq, self.head_dim])
                .repeat_dim(2, kv_groups)
                .reshape([batch, self.num_heads, seq, self.head_dim]);
            let v = v
                .reshape([batch, self.num_kv_heads, 1, seq, self.head_dim])
                .repeat_dim(2, kv_groups)
                .reshape([batch, self.num_heads, seq, self.head_dim]);
            (k, v)
        } else {
            (k, v)
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

        // Output projection
        self.o_proj.forward(output)
    }
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// Gemma 2 MLP: gate_proj + up_proj -> GeLU(gate) * up -> down_proj.
#[derive(Module, Debug)]
pub struct Gemma2MLP<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Gemma2MLP<B> {
    pub fn new(config: &Gemma2Config, device: &B::Device) -> Self {
        let gate_proj = LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(false)
            .init(device);
        let up_proj = LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(false)
            .init(device);
        let down_proj = LinearConfig::new(config.intermediate_size, config.hidden_size)
            .with_bias(false)
            .init(device);

        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Forward pass: `[batch, seq, hidden] -> [batch, seq, hidden]`.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gate = gelu_approximate(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate.mul(up))
    }
}

// ---------------------------------------------------------------------------
// Transformer Block
// ---------------------------------------------------------------------------

/// Gemma 2 transformer block with sandwich normalization.
///
/// Forward:
/// ```text
/// r = self_attn(input_layernorm(x))
/// h = x + post_attention_layernorm(r)
/// r = mlp(pre_feedforward_layernorm(h))
/// out = h + post_feedforward_layernorm(r)
/// ```
#[derive(Module, Debug)]
pub struct Gemma2Block<B: Backend> {
    pub self_attn: Gemma2Attention<B>,
    pub mlp: Gemma2MLP<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub pre_feedforward_layernorm: RmsNorm<B>,
    pub post_feedforward_layernorm: RmsNorm<B>,
}

impl<B: Backend> Gemma2Block<B> {
    pub fn new(config: &Gemma2Config, device: &B::Device) -> Self {
        let norm_cfg = RmsNormConfig::new(config.hidden_size).with_epsilon(config.rms_norm_eps);

        Self {
            self_attn: Gemma2Attention::new(config, device),
            mlp: Gemma2MLP::new(config, device),
            input_layernorm: norm_cfg.init(device),
            post_attention_layernorm: norm_cfg.init(device),
            pre_feedforward_layernorm: norm_cfg.init(device),
            post_feedforward_layernorm: norm_cfg.init(device),
        }
    }

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

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

/// Gemma 2 model: embedding + transformer blocks + final norm + LM head.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Gemma2Model<B: Backend> {
    pub embed: Embedding<B>,
    pub layers: Vec<Gemma2Block<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub final_logit_softcapping: f64,
}

impl<B: Backend> Gemma2Model<B> {
    /// Create a new Gemma 2 model from config.
    ///
    /// **Note:** Weights are randomly initialized. Use a weight loader to load
    /// pretrained weights from HuggingFace safetensors.
    pub fn new(config: &Gemma2Config, device: &B::Device) -> Self {
        let embed = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let layers = (0..config.num_hidden_layers)
            .map(|_| Gemma2Block::new(config, device))
            .collect();
        let norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);
        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);

        Self {
            embed,
            layers,
            norm,
            lm_head,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            final_logit_softcapping: config.final_logit_softcapping,
        }
    }

    /// Forward pass: token IDs -> logits.
    ///
    /// - Input: `[batch, seq_len]` integer token IDs
    /// - Output: `[batch, seq_len, vocab_size]` logits
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = input_ids.device();
        let [_batch, seq_len] = input_ids.dims();

        // Embedding lookup + Gemma 2 scaling: h = embed(tokens) * sqrt(hidden_size)
        let scale = (self.hidden_size as f64).sqrt();
        let h = self.embed.forward(input_ids).mul_scalar(scale);

        // Create causal mask
        let mask = causal_mask::<B>(seq_len, &device);

        // Transformer blocks
        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(h, Some(mask.clone()));
        }

        // Final norm (mixed precision for stability)
        let h = rms_norm_f32(&self.norm, h);

        // LM head: [batch, seq, hidden] -> [batch, seq, vocab]
        let logits = self.lm_head.forward(h);

        // Final logit softcapping in f32: tanh(logits / cap) * cap
        let original_dtype = logits.dtype();
        logits
            .cast(FloatDType::F32)
            .div_scalar(self.final_logit_softcapping)
            .tanh()
            .mul_scalar(self.final_logit_softcapping)
            .cast(original_dtype)
    }

    /// Forward pass returning hidden states (before LM head).
    ///
    /// Useful for LoRA fine-tuning where we compute loss on the logits separately.
    /// - Input: `[batch, seq_len]` integer token IDs
    /// - Output: `[batch, seq_len, hidden_size]` hidden states
    pub fn forward_hidden(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = input_ids.device();
        let [_batch, seq_len] = input_ids.dims();

        let scale = (self.hidden_size as f64).sqrt();
        let h = self.embed.forward(input_ids).mul_scalar(scale);

        let mask = causal_mask::<B>(seq_len, &device);

        let mut h = h;
        for layer in &self.layers {
            h = layer.forward(h, Some(mask.clone()));
        }

        // Final norm (mixed precision for stability)
        rms_norm_f32(&self.norm, h)
    }

    /// Compute logits from hidden states (after LM head + softcapping in f32).
    pub fn hidden_to_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let logits = self.lm_head.forward(hidden);
        // Softcapping in f32 for numerical stability
        let original_dtype = logits.dtype();
        logits
            .cast(FloatDType::F32)
            .div_scalar(self.final_logit_softcapping)
            .tanh()
            .mul_scalar(self.final_logit_softcapping)
            .cast(original_dtype)
    }
}

impl<B: Backend> ModuleDisplay for Gemma2Model<B> {
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::ops::FloatElem;
    use burn::tensor::{Distribution, Int, Shape, Tolerance};
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;
    type FT = FloatElem<TestBackend>;

    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn tiny_config() -> Gemma2Config {
        // Minimal config for fast unit tests
        Gemma2Config::new(
            256, // vocab_size
            64,  // hidden_size
            2,   // num_hidden_layers
            128, // intermediate_size
            4,   // num_attention_heads
            2,   // num_key_value_heads
            16,  // head_dim
        )
    }

    #[test]
    fn test_attention_forward_shapes() {
        let device = device();
        let config = tiny_config();
        let attn: Gemma2Attention<TestBackend> = Gemma2Attention::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::random([2, 8, 64], Distribution::Default, &device);
        let mask = causal_mask::<TestBackend>(8, &device);
        let out = attn.forward(x, Some(mask));

        assert_eq!(out.shape(), Shape::new([2, 8, 64]));
    }

    #[test]
    fn test_mlp_forward_shapes() {
        let device = device();
        let config = tiny_config();
        let mlp: Gemma2MLP<TestBackend> = Gemma2MLP::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::random([2, 8, 64], Distribution::Default, &device);
        let out = mlp.forward(x);

        assert_eq!(out.shape(), Shape::new([2, 8, 64]));
    }

    #[test]
    fn test_block_forward_shapes() {
        let device = device();
        let config = tiny_config();
        let block: Gemma2Block<TestBackend> = Gemma2Block::new(&config, &device);

        let x = Tensor::<TestBackend, 3>::random([2, 8, 64], Distribution::Default, &device);
        let mask = causal_mask::<TestBackend>(8, &device);
        let out = block.forward(x, Some(mask));

        assert_eq!(out.shape(), Shape::new([2, 8, 64]));
    }

    #[test]
    fn test_model_forward_shapes() {
        let device = device();
        let config = tiny_config();
        let model: Gemma2Model<TestBackend> = Gemma2Model::new(&config, &device);

        // Token IDs: [batch=2, seq=8]
        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &device);
        let logits = model.forward(input_ids);

        assert_eq!(logits.shape(), Shape::new([2, 8, 256]));
    }

    #[test]
    fn test_causal_mask_values() {
        let device = device();
        let mask = causal_mask::<TestBackend>(4, &device);

        // Expected:
        // [  0, -inf, -inf, -inf]
        // [  0,    0, -inf, -inf]
        // [  0,    0,    0, -inf]
        // [  0,    0,    0,    0]
        let data = mask.into_data();
        let values = data.as_slice::<f32>().unwrap();

        // (0,0) = 0
        assert!((values[0] - 0.0).abs() < 1e-6, "mask[0,0] should be 0");
        // (0,1) = -inf
        assert!(
            values[1].is_infinite() && values[1].is_sign_negative(),
            "mask[0,1] should be -inf"
        );
        // (1,0) = 0
        assert!((values[4] - 0.0).abs() < 1e-6, "mask[1,0] should be 0");
        // (3,3) = 0
        assert!((values[15] - 0.0).abs() < 1e-6, "mask[3,3] should be 0");
    }

    #[test]
    fn test_model_hidden_forward() {
        let device = device();
        let config = tiny_config();
        let model: Gemma2Model<TestBackend> = Gemma2Model::new(&config, &device);

        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);
        let hidden = model.forward_hidden(input_ids.clone());
        assert_eq!(hidden.shape(), Shape::new([1, 4, 64]));

        let logits = model.hidden_to_logits(hidden);
        assert_eq!(logits.shape(), Shape::new([1, 4, 256]));

        // Should match direct forward
        let direct_logits = model.forward(input_ids);
        direct_logits
            .into_data()
            .assert_approx_eq::<FT>(&logits.into_data(), Tolerance::default());
    }

    #[test]
    fn test_gqa_expansion() {
        let device = device();
        // 4 query heads, 2 kv heads -> groups = 2
        let config = tiny_config();
        assert_eq!(config.num_kv_groups(), 2);

        let attn: Gemma2Attention<TestBackend> = Gemma2Attention::new(&config, &device);

        // Verify projection shapes
        // q_proj: [hidden=64, num_heads*head_dim=64]
        let [d_in, d_out] = attn.q_proj.weight.shape().dims::<2>();
        assert_eq!(d_in, 64);
        assert_eq!(d_out, 64);

        // k_proj: [hidden=64, num_kv_heads*head_dim=32]
        let [d_in, d_out] = attn.k_proj.weight.shape().dims::<2>();
        assert_eq!(d_in, 64);
        assert_eq!(d_out, 32);
    }

    #[test]
    fn test_model_display() {
        let device = device();
        let config = tiny_config();
        let model: Gemma2Model<TestBackend> = Gemma2Model::new(&config, &device);

        let display = alloc::format!("{model}");
        assert!(display.contains("hidden_size: 64"));
        assert!(display.contains("vocab_size: 256"));
        assert!(display.contains("num_layers: 2"));
    }
}
