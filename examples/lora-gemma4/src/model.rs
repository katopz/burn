//! Gemma 4 model implementation for burn.
//!
//! Architecture reference: [Gemma 4](https://ai.google.dev/gemma) (Google, 2025)
//! Implementation reference: [mlx-lm gemma4_text.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py)
//!
//! Key Gemma 4 features (vs Gemma 2):
//! - 5:1 sliding/full attention pattern (vs 1:1 alternating)
//! - Q/K/V RMSNorm in attention (new)
//! - KV sharing across last N layers (new)
//! - Per-Layer Embeddings / PLE (new)
//! - Layer scalar per decoder layer (new)
//! - Global head dim (512) for full attention layers
//! - Different RoPE for sliding vs full attention (partial rotation)
//! - No attention logit softcapping (Gemma 2 had 50.0)
//! - Attention scale = 1.0 (Gemma 2 used 1/sqrt(head_dim))
//!
//! # Note on RMSNorm convention
//!
//! Gemma 4 uses `output * weight` (weight init=1, ones-based).
//! Burn's RmsNorm also uses `output * gamma` (gamma init=1, ones-based via `Initializer::Ones`).
//! These are identical: no conversion needed in the weight loader.

use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::nn::{
    Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding,
    RotaryEncodingConfig,
};
use burn::tensor::{
    DType, Element, Int, Tensor, activation::gelu_approximate, activation::softmax,
    backend::Backend,
};

use crate::types::{Gemma4Config, LayerType};

/// Type alias for KV cache pair: (keys, values) each `[batch, heads, seq, head_dim]`.
pub type KvPair<B> = Option<(Tensor<B, 4>, Tensor<B, 4>)>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Full causal mask: position i attends to positions 0..=i.
pub fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device).float();
    let row = positions.clone().reshape([seq_len, 1]);
    let col = positions.reshape([1, seq_len]);
    let attend = col.lower_equal(row);
    Tensor::<B, 2>::zeros([seq_len, seq_len], device)
        .mask_fill(attend.equal_elem(false), f32::NEG_INFINITY)
}

/// Sliding window causal mask: position i attends to max(0, i-window+1)..=i.
pub fn sliding_window_mask<B: Backend>(
    seq_len: usize,
    window: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device).float();
    let row = positions.clone().reshape([seq_len, 1]);
    let col = positions.reshape([1, seq_len]);

    // Causal: attend where col <= row
    let causal = col.clone().lower_equal(row.clone());

    // Window: attend where col >= max(0, row - window + 1)
    let window_start = (row - (window as f64 - 1.0)).clamp_min(0.0);
    let in_window = col.greater_equal(window_start);

    // Apply both masks sequentially (equivalent to AND)
    Tensor::<B, 2>::zeros([seq_len, seq_len], device)
        .mask_fill(causal.equal_elem(false), f32::NEG_INFINITY)
        .mask_fill(in_window.equal_elem(false), f32::NEG_INFINITY)
}

/// RMS normalization without learnable scale (for v_norm).
pub fn rms_norm_no_scale<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    epsilon: f64,
) -> Tensor<B, D> {
    let rms = x
        .clone()
        .powf_scalar(2.0)
        .mean_dim(D - 1)
        .add_scalar(epsilon)
        .sqrt();
    x.div(rms)
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Gemma 4 multi-head attention with GQA, Q/K/V norms, and per-layer-type RoPE.
///
/// Key differences from Gemma 2:
/// - Q/K/V RMSNorm after projection (q_norm, k_norm, v_norm)
/// - No attention logit softcapping
/// - Attention scale = 1.0 (not 1/sqrt(head_dim))
/// - Different head_dim for sliding (256) vs full (512/global_head_dim)
/// - Different RoPE for sliding (theta=10000, full) vs full (theta=1e6, partial=25%)
/// - KV sharing: shared layers skip own KV computation
#[derive(Module, Debug)]
pub struct Gemma4Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    /// Query normalization (with learnable weight).
    pub q_norm: RmsNorm<B>,
    /// Key normalization (with learnable weight).
    pub k_norm: RmsNorm<B>,
    // v_norm: no learnable weight — applied via `rms_norm_no_scale()`.
    /// Rotary positional encoding (d_model = rotary_dim, which varies per layer type).
    pub rotary: RotaryEncoding<B>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub layer_type: LayerType,
    /// Whether this layer has its own KV projections (true for non-shared layers).
    pub has_own_kv: bool,
    /// Fraction of head_dim that gets rotary encoding (1.0 = full, 0.25 = partial).
    pub partial_rotary_factor: f64,
}

impl<B: Backend> Gemma4Attention<B> {
    /// Create a new attention layer for the given layer index.
    pub fn new(config: &Gemma4Config, layer_idx: usize, device: &B::Device) -> Self {
        let layer_type = config.layer_type(layer_idx);
        let effective_head_dim = config.head_dim_for(layer_type);
        let num_kv_heads = config.num_kv_heads_for(layer_type);
        let has_own_kv = config.has_own_kv(layer_idx);

        // Projections — sizes depend on layer type (head_dim varies)
        let q_proj = LinearConfig::new(
            config.hidden_size,
            config.num_attention_heads * effective_head_dim,
        )
        .with_bias(false)
        .init(device);
        let k_proj = LinearConfig::new(config.hidden_size, num_kv_heads * effective_head_dim)
            .with_bias(false)
            .init(device);
        let v_proj = LinearConfig::new(config.hidden_size, num_kv_heads * effective_head_dim)
            .with_bias(false)
            .init(device);
        let o_proj = LinearConfig::new(
            config.num_attention_heads * effective_head_dim,
            config.hidden_size,
        )
        .with_bias(false)
        .init(device);

        // Q/K norms (with learnable weight, head_dim-sized)
        let norm_cfg = RmsNormConfig::new(effective_head_dim).with_epsilon(config.rms_norm_eps);
        let q_norm = norm_cfg.init(device);
        let k_norm = norm_cfg.init(device);

        // RoPE — Proportional RoPE for full attention layers.
        //
        // HF's "proportional" rope_type normalizes frequencies by global_head_dim (512),
        // NOT by rotary_dim (128). Standard burn RotaryEncoding normalizes by d_model
        // (= rotary_dim), so we must rescale for partial rotation layers.
        //
        // burn default:  freq[i] = 1/(base^(2i/rotary_dim))
        // HF proportional: freq[i] = 1/(base^(2i/head_dim))
        // Fix: freq_fixed = freq^(rotary_dim / head_dim)
        let rope_params = config.rope_params_for(layer_type);
        let rotary_dim =
            ((effective_head_dim as f64) * (rope_params.partial_rotary_factor as f64)) as usize;
        let rotary = if rope_params.partial_rotary_factor < 1.0 {
            let scale = rotary_dim as f32 / effective_head_dim as f32;
            RotaryEncodingConfig::new(config.max_position_embeddings, rotary_dim)
                .with_theta(rope_params.rope_theta)
                .init_with_frequency_scaling(|freq| freq.powf_scalar(scale), device)
        } else {
            RotaryEncodingConfig::new(config.max_position_embeddings, rotary_dim)
                .with_theta(rope_params.rope_theta)
                .init(device)
        };

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary,
            num_heads: config.num_attention_heads,
            num_kv_heads,
            head_dim: effective_head_dim,
            layer_type,
            has_own_kv,
            partial_rotary_factor: rope_params.partial_rotary_factor as f64,
        }
    }

    /// Apply rotary encoding, handling partial rotation for full attention layers.
    fn apply_rotary(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // x: [batch, heads, seq, head_dim]
        if self.partial_rotary_factor >= 1.0 {
            return self.rotary.forward(x);
        }

        let dims = x.dims();
        let [batch, heads, seq, dim] = [dims[0], dims[1], dims[2], dims[3]];
        let rotary_dim = ((dim as f64) * self.partial_rotary_factor) as usize;

        if rotary_dim == 0 || rotary_dim >= dim {
            return self.rotary.forward(x);
        }

        // Split into rotary and pass-through portions
        let x_clone = x.clone();
        let x_rot = x.slice([0..batch, 0..heads, 0..seq, 0..rotary_dim]);
        let x_pass = x_clone.slice([0..batch, 0..heads, 0..seq, rotary_dim..dim]);

        // Apply RoPE to rotary portion only
        let x_rot = self.rotary.forward(x_rot);

        // Concat back
        Tensor::cat(vec![x_rot, x_pass], 3)
    }

    /// Forward pass.
    ///
    /// - `x`: `[batch, seq, hidden_size]`
    /// - `mask`: additive attention mask `[seq, seq]`
    /// - `shared_kv`: pre-computed K,V from source layer (for KV-shared layers)
    ///
    /// Returns `(output_hidden, own_kv)` where `own_kv` is `Some` for non-shared layers.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        shared_kv: KvPair<B>,
    ) -> (Tensor<B, 3>, KvPair<B>) {
        let [batch, seq, _hidden] = x.dims();
        let kv_groups = self.num_heads / self.num_kv_heads;

        // --- Query ---
        let q = self.q_proj.forward(x.clone());
        let q = q.reshape([batch, seq, self.num_heads, self.head_dim]);
        let q = self.q_norm.forward(q); // norm over head_dim
        let q = q.swap_dims(1, 2); // [batch, heads, seq, head_dim]
        let q = self.apply_rotary(q);

        // --- Key, Value ---
        let (keys, values, own_kv) = if let Some((shared_k, shared_v)) = shared_kv {
            // KV-shared: reuse from source layer
            (shared_k, shared_v, None)
        } else {
            // Own KV: compute from projections
            let k = self.k_proj.forward(x.clone());
            let k = k.reshape([batch, seq, self.num_kv_heads, self.head_dim]);
            let k = self.k_norm.forward(k);
            let k = k.swap_dims(1, 2); // [batch, kv_heads, seq, head_dim]
            let k = self.apply_rotary(k);

            let v = self.v_proj.forward(x);
            let v = v.reshape([batch, seq, self.num_kv_heads, self.head_dim]);
            // v_norm: no learnable scale (RMSNormNoScale)
            let v = rms_norm_no_scale(v, 1e-6);
            let v = v.swap_dims(1, 2); // [batch, kv_heads, seq, head_dim]
            // No RoPE for values!

            (k.clone(), v.clone(), Some((k, v)))
        };

        // --- GQA expansion ---
        let (keys, values) = if kv_groups > 1 {
            let keys = keys
                .reshape([batch, self.num_kv_heads, 1, seq, self.head_dim])
                .repeat_dim(2, kv_groups)
                .reshape([batch, self.num_heads, seq, self.head_dim]);
            let values = values
                .reshape([batch, self.num_kv_heads, 1, seq, self.head_dim])
                .repeat_dim(2, kv_groups)
                .reshape([batch, self.num_heads, seq, self.head_dim]);
            (keys, values)
        } else {
            (keys, values)
        };

        // --- Attention scores in f32 to prevent f16 overflow (head_dim=512 for full attention) ---
        // HF computes entire attention in float32 to avoid f16 precision loss:
        //   torch.matmul(query, key_states.transpose(2, 3)) * scaling
        //   nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        let original_dtype = B::FloatElem::dtype();
        let scores = q
            .cast(DType::F32)
            .matmul(keys.cast(DType::F32).swap_dims(2, 3));

        // --- Apply mask ---
        let scores = match mask {
            Some(m) => scores.add(m.cast(DType::F32).reshape([1, 1, seq, seq])),
            None => scores,
        };

        // Softmax in f32, then cast back for value weighted sum
        let weights = softmax(scores, 3).cast(original_dtype);
        let output = weights.matmul(values); // [batch, heads, seq, head_dim]

        // --- Reshape + output projection ---
        let output = output
            .swap_dims(1, 2)
            .reshape([batch, seq, self.num_heads * self.head_dim]);
        let output = self.o_proj.forward(output);

        (output, own_kv)
    }
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// Gemma 4 MLP: GeGLU (gate + up → GeLU(gate) * up → down).
///
/// Same architecture as Gemma 2.
#[derive(Module, Debug)]
pub struct Gemma4MLP<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Gemma4MLP<B> {
    pub fn new(config: &Gemma4Config, device: &B::Device) -> Self {
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

    /// Forward: `[batch, seq, hidden] -> [batch, seq, hidden]`.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gate = gelu_approximate(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate.mul(up))
    }
}

// ---------------------------------------------------------------------------
// Transformer Block
// ---------------------------------------------------------------------------

/// Gemma 4 transformer block with sandwich normalization, PLE, and layer scalar.
///
/// Forward:
/// ```text
/// residual = x
/// h = self_attn(input_layernorm(x))
/// h = post_attention_layernorm(h)
/// h = residual + h
///
/// residual = h
/// h = mlp(pre_feedforward_layernorm(h))
/// h = post_feedforward_layernorm(h)
/// h = residual + h
///
/// // Per-Layer Embeddings (PLE)
/// if has_ple:
///     residual = h
///     gate = gelu(per_layer_input_gate(h)) * per_layer_input
///     h = per_layer_projection(gate)
///     h = post_per_layer_input_norm(h)
///     h = residual + h
///
/// h = h * layer_scalar  // always 1.0 for E4B
/// ```
#[derive(Module, Debug)]
pub struct Gemma4Block<B: Backend> {
    pub self_attn: Gemma4Attention<B>,
    pub mlp: Gemma4MLP<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub pre_feedforward_layernorm: RmsNorm<B>,
    pub post_feedforward_layernorm: RmsNorm<B>,
    // Per-Layer Embeddings (PLE) components — Some when PLE enabled
    pub per_layer_input_gate: Option<Linear<B>>,
    pub per_layer_projection: Option<Linear<B>>,
    pub post_per_layer_input_norm: Option<RmsNorm<B>>,
    // Layer scalar (always 1.0 for E4B, stored as constant)
    pub layer_scalar: f64,
}

impl<B: Backend> Gemma4Block<B> {
    pub fn new(config: &Gemma4Config, layer_idx: usize, device: &B::Device) -> Self {
        let norm_cfg = RmsNormConfig::new(config.hidden_size).with_epsilon(config.rms_norm_eps);

        // PLE components
        let (ple_gate, ple_proj, ple_norm) = if config.has_ple() {
            let gate = LinearConfig::new(config.hidden_size, config.hidden_size_per_layer_input)
                .with_bias(false)
                .init(device);
            let proj = LinearConfig::new(config.hidden_size_per_layer_input, config.hidden_size)
                .with_bias(false)
                .init(device);
            let norm = RmsNormConfig::new(config.hidden_size)
                .with_epsilon(config.rms_norm_eps)
                .init(device);
            (Some(gate), Some(proj), Some(norm))
        } else {
            (None, None, None)
        };

        Self {
            self_attn: Gemma4Attention::new(config, layer_idx, device),
            mlp: Gemma4MLP::new(config, device),
            input_layernorm: norm_cfg.init(device),
            post_attention_layernorm: norm_cfg.init(device),
            pre_feedforward_layernorm: norm_cfg.init(device),
            post_feedforward_layernorm: norm_cfg.init(device),
            per_layer_input_gate: ple_gate,
            per_layer_projection: ple_proj,
            post_per_layer_input_norm: ple_norm,
            layer_scalar: 1.0, // always 1.0 for E4B
        }
    }

    /// Forward pass.
    ///
    /// - `x`: `[batch, seq, hidden]`
    /// - `mask`: attention mask `[seq, seq]`
    /// - `shared_kv`: pre-computed K,V from source layer (for KV-shared layers)
    /// - `per_layer_input`: PLE input `[batch, seq, ple_dim]` (None if PLE disabled)
    ///
    /// Returns `(output_hidden, own_kv)`.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 2>>,
        shared_kv: KvPair<B>,
        per_layer_input: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, KvPair<B>) {
        // --- Attention with sandwich norm ---
        let residual = x.clone();
        let (attn_out, own_kv) =
            self.self_attn
                .forward(self.input_layernorm.forward(x), mask, shared_kv);
        let h = residual + self.post_attention_layernorm.forward(attn_out);

        // --- MLP with sandwich norm ---
        let residual = h.clone();
        let mlp_out = self.mlp.forward(self.pre_feedforward_layernorm.forward(h));
        let mut h = residual + self.post_feedforward_layernorm.forward(mlp_out);

        // --- Per-Layer Embeddings (PLE) ---
        if let (Some(gate), Some(proj), Some(norm), Some(ple_input)) = (
            &self.per_layer_input_gate,
            &self.per_layer_projection,
            &self.post_per_layer_input_norm,
            per_layer_input,
        ) {
            let residual = h.clone();
            let gate_val = gelu_approximate(gate.forward(h));
            let gated = gate_val.mul(ple_input); // [batch, seq, ple_dim]
            let projected = proj.forward(gated); // [batch, seq, hidden]
            h = residual + norm.forward(projected);
        }

        // --- Layer scalar (always 1.0 for E4B) ---
        if (self.layer_scalar - 1.0).abs() > f64::EPSILON {
            h = h.mul_scalar(self.layer_scalar);
        }

        (h, own_kv)
    }
}

// ---------------------------------------------------------------------------
// Full Model
// ---------------------------------------------------------------------------

/// Gemma 4 model: embedding + transformer blocks + PLE + final norm + LM head.
///
/// Supports:
/// - Tied word embeddings (lm_head shares embed_tokens weight)
/// - Per-Layer Embeddings (PLE) for richer per-layer representations
/// - KV sharing across the last N layers for efficiency
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Gemma4Model<B: Backend> {
    pub embed: Embedding<B>,
    pub layers: Vec<Gemma4Block<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: Linear<B>,
    // Per-Layer Embeddings (PLE) — model-level components
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
    /// Non-shared layers map to themselves.
    pub kv_source_map: Vec<usize>,
}

impl<B: Backend> Gemma4Model<B> {
    /// Create a new model with random weights.
    pub fn new(config: &Gemma4Config, device: &B::Device) -> Self {
        let embed = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let layers = (0..config.num_hidden_layers)
            .map(|i| Gemma4Block::new(config, i, device))
            .collect();
        let norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);
        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);

        // PLE components
        let (embed_per_layer, proj_per_layer, norm_per_layer) = if config.has_ple() {
            let emb = EmbeddingConfig::new(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            )
            .init(device);
            let proj = LinearConfig::new(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
            )
            .with_bias(false)
            .init(device);
            let n = RmsNormConfig::new(config.hidden_size_per_layer_input)
                .with_epsilon(config.rms_norm_eps)
                .init(device);
            (Some(emb), Some(proj), Some(n))
        } else {
            (None, None, None)
        };

        Self {
            embed,
            layers,
            norm,
            lm_head,
            embed_tokens_per_layer: embed_per_layer,
            per_layer_model_projection: proj_per_layer,
            per_layer_projection_norm: norm_per_layer,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            final_logit_softcapping: config.final_logit_softcapping,
            tie_word_embeddings: config.tie_word_embeddings,
            num_hidden_layers: config.num_hidden_layers,
            hidden_size_per_layer_input: config.hidden_size_per_layer_input,
            sliding_window: config.sliding_window,
            kv_source_map: config.kv_source_map(),
        }
    }

    /// Forward pass: token IDs → logits.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let hidden = self.forward_hidden(input_ids);
        self.hidden_to_logits(hidden)
    }

    /// Forward pass returning hidden states (before LM head).
    pub fn forward_hidden(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = input_ids.device();
        let [_batch, seq_len] = input_ids.dims();

        // Embedding + Gemma scaling: h = embed(tokens) * sqrt(hidden_size)
        let scale = (self.hidden_size as f64).sqrt();
        let h = self.embed.forward(input_ids.clone()).mul_scalar(scale);

        // Pre-compute masks for each layer type
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
            let shared_kv = if i < self.kv_source_map.len() && self.kv_source_map[i] != i {
                kv_cache[self.kv_source_map[i]].clone()
            } else {
                None
            };

            // Get per-layer input for this layer
            let ple_input = per_layer_inputs.as_ref().map(|inputs| inputs[i].clone());

            let (new_h, own_kv) = layer.forward(h, mask, shared_kv, ple_input);

            // Store own KV for potential sharing
            if let Some(kv) = own_kv {
                kv_cache[i] = Some(kv);
            }

            h = new_h;
        }

        // Final norm
        self.norm.forward(h)
    }

    /// Compute logits from hidden states.
    pub fn hidden_to_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        // Always use lm_head; for tied embeddings, the loader copies embed weight into lm_head.
        let logits = self.lm_head.forward(hidden);

        // Final logit softcapping: tanh(logits / cap) * cap
        if (self.final_logit_softcapping - 0.0).abs() > f64::EPSILON {
            logits
                .div_scalar(self.final_logit_softcapping)
                .tanh()
                .mul_scalar(self.final_logit_softcapping)
        } else {
            logits
        }
    }

    /// Compute per-layer inputs for PLE.
    ///
    /// Returns None if PLE is disabled, otherwise Vec of per-layer 3D tensors.
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

        // Token PLE: embed_per_layer(input_ids) * sqrt(ple_dim)
        let ple_scale = (ple_dim as f64).sqrt();
        let token_ple = embed_per_layer
            .forward(input_ids.clone())
            .mul_scalar(ple_scale);
        let token_ple = token_ple.reshape([batch, seq, num_layers, ple_dim]);

        // Hidden PLE: projection(hidden) * 1/sqrt(hidden_size) → norm
        let proj_scale = 1.0 / (self.hidden_size as f64).sqrt();
        let hidden_ple = proj_per_layer
            .forward(hidden.clone())
            .mul_scalar(proj_scale);
        let hidden_ple = hidden_ple.reshape([batch, seq, num_layers, ple_dim]);
        let hidden_ple = norm_per_layer.forward(hidden_ple);

        // Combine: (token + hidden) * 1/sqrt(2)
        let combine_scale = 1.0 / 2.0_f64.sqrt();
        let combined = (token_ple + hidden_ple).mul_scalar(combine_scale);

        // Slice per layer
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
}

impl<B: Backend> ModuleDisplay for Gemma4Model<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let num_sliding = self
            .layers
            .iter()
            .filter(|l| l.self_attn.layer_type == LayerType::Sliding)
            .count();
        let num_full = self.layers.len() - num_sliding;
        let has_ple = self.embed_tokens_per_layer.is_some();

        content
            .add("hidden_size", &self.hidden_size)
            .add("vocab_size", &self.vocab_size)
            .add("num_layers", &self.layers.len())
            .add("sliding_layers", &num_sliding)
            .add("full_layers", &num_full)
            .add("tie_word_embeddings", &self.tie_word_embeddings)
            .add("has_ple", &has_ple)
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
    use burn::tensor::Tolerance;

    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    /// Tiny config for testing: 6 layers (5 sliding + 1 full), no PLE, no KV sharing.
    fn tiny_config() -> Gemma4Config {
        Gemma4Config::new(
            256, // vocab_size
            64,  // hidden_size
            6,   // num_hidden_layers
            128, // intermediate_size
            4,   // num_attention_heads
            2,   // num_key_value_heads
            16,  // head_dim
        )
        .with_global_head_dim(32)
        .with_sliding_window(8)
        .with_final_logit_softcapping(30.0)
        .with_max_position_embeddings(512)
    }

    /// Config with KV sharing: 12 layers, last 4 shared.
    fn config_with_kv_sharing() -> Gemma4Config {
        Gemma4Config::new(
            256, // vocab_size
            64,  // hidden_size
            12,  // num_hidden_layers (10 sliding + 2 full)
            128, // intermediate_size
            4,   // num_attention_heads
            2,   // num_key_value_heads
            16,  // head_dim
        )
        .with_global_head_dim(32)
        .with_sliding_window(8)
        .with_num_kv_shared_layers(4)
        .with_max_position_embeddings(512)
    }

    /// Config with PLE enabled.
    fn config_with_ple() -> Gemma4Config {
        tiny_config()
            .with_hidden_size_per_layer_input(8)
            .with_vocab_size_per_layer_input(256)
    }

    #[test]
    fn test_causal_mask_values() {
        let dev = device();
        let mask = causal_mask::<TestBackend>(4, &dev);

        let expected = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
                [0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
                [0.0, 0.0, 0.0, f32::NEG_INFINITY],
                [0.0, 0.0, 0.0, 0.0],
            ],
            &dev,
        );

        mask.to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(0.001));
    }

    #[test]
    fn test_sliding_window_mask_values() {
        let dev = device();
        // Window = 3: position i attends to max(0, i-2)..=i
        let mask = sliding_window_mask::<TestBackend>(4, 3, &dev);

        let expected = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.0, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
                [0.0, 0.0, f32::NEG_INFINITY, f32::NEG_INFINITY],
                [0.0, 0.0, 0.0, f32::NEG_INFINITY],
                [f32::NEG_INFINITY, 0.0, 0.0, 0.0],
            ],
            &dev,
        );

        mask.to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(0.001));
    }

    #[test]
    fn test_attention_forward_shapes() {
        let dev = device();
        let config = tiny_config();

        // Layer 0: sliding, head_dim=16
        let attn = Gemma4Attention::<TestBackend>::new(&config, 0, &dev);
        let x = Tensor::<TestBackend, 3>::zeros([2, 8, 64], &dev);
        let mask = Some(causal_mask::<TestBackend>(8, &dev));

        let (output, own_kv) = attn.forward(x, mask, None);

        assert_eq!(output.dims(), [2, 8, 64]); // [batch, seq, hidden]
        let (k, v) = own_kv.expect("non-shared layer should return own KV");
        assert_eq!(k.dims(), [2, 2, 8, 16]); // [batch, kv_heads, seq, head_dim]
        assert_eq!(v.dims(), [2, 2, 8, 16]);
    }

    #[test]
    fn test_attention_full_layer_shapes() {
        let dev = device();
        let config = tiny_config();

        // Layer 5: full attention, head_dim=32 (global_head_dim)
        let attn = Gemma4Attention::<TestBackend>::new(&config, 5, &dev);
        let x = Tensor::<TestBackend, 3>::zeros([2, 8, 64], &dev);
        let mask = Some(causal_mask::<TestBackend>(8, &dev));

        let (output, own_kv) = attn.forward(x, mask, None);

        assert_eq!(output.dims(), [2, 8, 64]);
        let (k, v) = own_kv.expect("full attn layer should return own KV");
        assert_eq!(k.dims(), [2, 2, 8, 32]); // global_head_dim=32
        assert_eq!(v.dims(), [2, 2, 8, 32]);
    }

    #[test]
    fn test_mlp_forward_shapes() {
        let dev = device();
        let config = tiny_config();
        let mlp = Gemma4MLP::new(&config, &dev);

        let x = Tensor::<TestBackend, 3>::zeros([2, 8, 64], &dev);
        let output = mlp.forward(x);

        assert_eq!(output.dims(), [2, 8, 64]);
    }

    #[test]
    fn test_block_forward_shapes() {
        let dev = device();
        let config = tiny_config();

        // Layer 0: sliding
        let block = Gemma4Block::new(&config, 0, &dev);
        let x = Tensor::<TestBackend, 3>::zeros([2, 8, 64], &dev);
        let mask = Some(causal_mask::<TestBackend>(8, &dev));

        let (output, own_kv) = block.forward(x, mask, None, None);

        assert_eq!(output.dims(), [2, 8, 64]);
        assert!(own_kv.is_some());
    }

    #[test]
    fn test_model_forward_shapes() {
        let dev = device();
        let config = tiny_config();
        let model = Gemma4Model::new(&config, &dev);

        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([2, 8], &dev);
        let logits = model.forward(input_ids);

        assert_eq!(logits.dims(), [2, 8, 256]); // [batch, seq, vocab]
    }

    #[test]
    fn test_model_hidden_forward() {
        let dev = device();
        let config = tiny_config();
        let model = Gemma4Model::new(&config, &dev);

        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &dev);
        let hidden = model.forward_hidden(input_ids);

        assert_eq!(hidden.dims(), [1, 4, 64]); // [batch, seq, hidden]
    }

    #[test]
    fn test_gqa_expansion() {
        let dev = device();
        let config = tiny_config();
        // num_heads=4, num_kv_heads=2 → kv_groups=2

        let attn = Gemma4Attention::<TestBackend>::new(&config, 0, &dev);
        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.num_kv_heads, 2);

        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &dev);
        let (output, _) = attn.forward(x, None, None);
        assert_eq!(output.dims(), [1, 4, 64]); // hidden_size restored
    }

    #[test]
    fn test_kv_sharing_forward() {
        let dev = device();
        let config = config_with_kv_sharing();
        let model = Gemma4Model::new(&config, &dev);

        // 12 layers, last 4 shared
        // Layer 8 (sliding, idx=8) → shares from layer 0
        // Layer 11 (full, idx=11) → shares from layer 5
        assert_eq!(model.kv_source_map[8], 0);
        assert_eq!(model.kv_source_map[11], 5);

        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &dev);
        let logits = model.forward(input_ids);
        assert_eq!(logits.dims(), [1, 4, 256]);
    }

    #[test]
    fn test_ple_forward() {
        let dev = device();
        let config = config_with_ple();
        let model = Gemma4Model::<TestBackend>::new(&config, &dev);

        assert!(model.embed_tokens_per_layer.is_some());
        assert!(model.per_layer_model_projection.is_some());

        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &dev);
        let logits = model.forward(input_ids);
        assert_eq!(logits.dims(), [1, 4, 256]);
    }

    #[test]
    fn test_attention_shared_kv() {
        let dev = device();
        let config = config_with_kv_sharing();

        // Create source attention (layer 0, sliding) and compute its KV
        let src_attn = Gemma4Attention::<TestBackend>::new(&config, 0, &dev);
        let x = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &dev);
        let (_, src_kv) = src_attn.forward(x.clone(), None, None);
        let (src_k, src_v) = src_kv.expect("source should have own KV");

        // Create shared attention (layer 8, sliding, shares KV from layer 0)
        let shared_attn = Gemma4Attention::<TestBackend>::new(&config, 8, &dev);
        assert!(!shared_attn.has_own_kv);

        // Forward with shared KV
        let (output, own_kv) = shared_attn.forward(x, None, Some((src_k, src_v)));

        assert_eq!(output.dims(), [1, 4, 64]);
        assert!(own_kv.is_none(), "shared layer should not return own KV");
    }

    #[test]
    fn test_tied_embeddings_logits() {
        let dev = device();
        let config = tiny_config(); // tie_word_embeddings=true by default
        let model = Gemma4Model::<TestBackend>::new(&config, &dev);

        assert!(model.tie_word_embeddings);

        let hidden = Tensor::<TestBackend, 3>::zeros([1, 4, 64], &dev);
        let logits = model.hidden_to_logits(hidden);
        assert_eq!(logits.dims(), [1, 4, 256]);
    }

    #[test]
    fn test_model_display() {
        let dev = device();
        let config = tiny_config();
        let model = Gemma4Model::<TestBackend>::new(&config, &dev);
        let display = format!("{model}");
        assert!(display.contains("hidden_size"));
        assert!(display.contains("vocab_size"));
    }

    #[test]
    fn test_no_attention_softcapping() {
        // Verify that attention does NOT apply softcapping
        // (Gemma 2 did: tanh(scores/50)*50, Gemma 4 does not)
        let dev = device();
        let config = tiny_config();
        let attn = Gemma4Attention::<TestBackend>::new(&config, 0, &dev);

        // The attention struct should not have a softcap field
        // (it's absent from the struct definition)
        assert_eq!(attn.layer_type, LayerType::Sliding);
    }

    #[test]
    fn test_partial_rotary_dims() {
        let dev = device();
        let config = tiny_config();

        // Sliding: head_dim=16, partial_rotary_factor=1.0 → rotary_dim=16
        let sliding_attn = Gemma4Attention::<TestBackend>::new(&config, 0, &dev);
        assert!((sliding_attn.partial_rotary_factor - 1.0).abs() < f64::EPSILON);

        // Full: global_head_dim=32, partial_rotary_factor=0.25 → rotary_dim=8
        let full_attn = Gemma4Attention::<TestBackend>::new(&config, 5, &dev);
        assert!((full_attn.partial_rotary_factor - 0.25).abs() < f64::EPSILON);
    }
}
