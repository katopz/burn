//! Gemma 2 model types and configuration.
//!
//! Based on [mlx-lm Gemma 2 implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma2.py)
//! and [HuggingFace Gemma 2 config](https://huggingface.co/google/gemma-2-2b).

use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;

/// Gemma 2 model hyperparameters.
///
/// Matches the HuggingFace `config.json` fields for `google/gemma-2-2b`.
#[derive(Config, Debug)]
pub struct Gemma2Config {
    /// Vocabulary size (e.g., 256000 for Gemma 2).
    pub vocab_size: usize,

    /// Hidden dimension (e.g., 2304 for Gemma 2 2B).
    pub hidden_size: usize,

    /// Number of transformer decoder layers (e.g., 26 for Gemma 2 2B).
    pub num_hidden_layers: usize,

    /// Intermediate (MLP) dimension (e.g., 9216 for Gemma 2 2B).
    pub intermediate_size: usize,

    /// Number of query attention heads.
    pub num_attention_heads: usize,

    /// Number of key/value attention heads (GQA, e.g., 4 for Gemma 2 2B).
    pub num_key_value_heads: usize,

    /// Dimension per attention head (e.g., 256 for Gemma 2 2B).
    pub head_dim: usize,

    /// RMS normalization epsilon. Default: 1e-6.
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,

    /// RoPE base frequency. Default: 10000.0.
    #[config(default = 10000.0)]
    pub rope_theta: f32,

    /// Attention logit softcapping value. Default: 50.0.
    #[config(default = 50.0)]
    pub attn_logit_softcapping: f64,

    /// Final logit softcapping value. Default: 30.0.
    #[config(default = 30.0)]
    pub final_logit_softcapping: f64,
}

impl Gemma2Config {
    /// Returns the number of query heads per KV head (grouped query attention ratio).
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Returns the attention scale factor.
    /// Gemma 2 uses `1.0 / sqrt(query_pre_attn_scalar)` where `query_pre_attn_scalar = head_dim`.
    pub fn attention_scale(&self) -> f64 {
        1.0 / (self.head_dim as f64).sqrt()
    }
}

/// Preset configuration for Gemma 2 2B.
impl Gemma2Config {
    /// Gemma 2 2B configuration.
    pub fn gemma2_2b() -> Self {
        Self::new(
            256000, // vocab_size
            2304,   // hidden_size
            26,     // num_hidden_layers
            9216,   // intermediate_size
            8,      // num_attention_heads
            4,      // num_key_value_heads
            256,    // head_dim
        )
    }

    /// Gemma 2 9B configuration.
    pub fn gemma2_9b() -> Self {
        Self::new(
            256000, // vocab_size
            3584,   // hidden_size
            42,     // num_hidden_layers
            14336,  // intermediate_size
            16,     // num_attention_heads
            8,      // num_key_value_heads
            256,    // head_dim
        )
    }
}

/// Gemma 2 model output.
#[derive(Module, Debug)]
pub struct Gemma2Output<B: Backend> {
    /// Hidden states after the final RMS norm: `[batch, seq_len, hidden_size]`.
    pub hidden_states: burn::tensor::Tensor<B, 3>,
}

/// Enumeration of LoRA target modules for Gemma 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoraTarget {
    /// Query projection.
    QProj,
    /// Key projection.
    KProj,
    /// Value projection.
    VProj,
    /// Output projection.
    OProj,
    /// Gate projection in MLP.
    GateProj,
    /// Up projection in MLP.
    UpProj,
    /// Down projection in MLP.
    DownProj,
}

impl LoraTarget {
    /// Returns all attention targets (q, k, v, o projections).
    pub fn attention_targets() -> &'static [LoraTarget] {
        &[
            LoraTarget::QProj,
            LoraTarget::KProj,
            LoraTarget::VProj,
            LoraTarget::OProj,
        ]
    }

    /// Returns all MLP targets (gate, up, down projections).
    pub fn mlp_targets() -> &'static [LoraTarget] {
        &[
            LoraTarget::GateProj,
            LoraTarget::UpProj,
            LoraTarget::DownProj,
        ]
    }

    /// Returns all targets (attention + MLP).
    pub fn all_targets() -> &'static [LoraTarget] {
        &[
            LoraTarget::QProj,
            LoraTarget::KProj,
            LoraTarget::VProj,
            LoraTarget::OProj,
            LoraTarget::GateProj,
            LoraTarget::UpProj,
            LoraTarget::DownProj,
        ]
    }

    /// Returns the HF weight name suffix for this target.
    /// E.g., `q_proj` for `LoraTarget::QProj`.
    pub fn weight_name(&self) -> &'static str {
        match self {
            LoraTarget::QProj => "q_proj",
            LoraTarget::KProj => "k_proj",
            LoraTarget::VProj => "v_proj",
            LoraTarget::OProj => "o_proj",
            LoraTarget::GateProj => "gate_proj",
            LoraTarget::UpProj => "up_proj",
            LoraTarget::DownProj => "down_proj",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma2_2b_config() {
        let config = Gemma2Config::gemma2_2b();
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.hidden_size, 2304);
        assert_eq!(config.num_hidden_layers, 26);
        assert_eq!(config.intermediate_size, 9216);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_key_value_heads, 4);
        assert_eq!(config.head_dim, 256);
    }

    #[test]
    fn test_kv_groups() {
        let config = Gemma2Config::gemma2_2b();
        assert_eq!(config.num_kv_groups(), 2); // 8 / 4 = 2
    }

    #[test]
    fn test_attention_scale() {
        let config = Gemma2Config::gemma2_2b();
        let expected = 1.0 / 256.0_f64.sqrt();
        assert!((config.attention_scale() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_lora_target_names() {
        assert_eq!(LoraTarget::QProj.weight_name(), "q_proj");
        assert_eq!(LoraTarget::GateProj.weight_name(), "gate_proj");
        assert_eq!(LoraTarget::DownProj.weight_name(), "down_proj");
    }

    #[test]
    fn test_lora_all_targets_count() {
        assert_eq!(LoraTarget::all_targets().len(), 7);
        assert_eq!(LoraTarget::attention_targets().len(), 4);
        assert_eq!(LoraTarget::mlp_targets().len(), 3);
    }
}
