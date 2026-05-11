//! Gemma 4 model types and configuration.
//!
//! Based on [mlx-lm Gemma 4 implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py)
//! and [HuggingFace Gemma 4 config](https://huggingface.co/google/gemma-4-4b-it).
//!
//! Key differences from Gemma 2:
//! - Sliding (5:1) + full attention pattern (vs Gemma 2's 1:1 alternating)
//! - Q/K/V RMSNorm in attention (new)
//! - KV sharing across last N layers (new)
//! - Per-Layer Embeddings (PLE) (new)
//! - Layer scalar per decoder layer (new)
//! - Global head dim (512) for full attention layers
//! - Different RoPE for sliding vs full attention
//! - No attention logit softcapping (Gemma 2 had 50.0)
//! - Final logit softcapping kept at 30.0

use burn::config::Config;

// ---------------------------------------------------------------------------
// Layer Type
// ---------------------------------------------------------------------------

/// Attention type for a decoder layer.
///
/// Gemma 4 uses a 5:1 pattern: 5 sliding attention layers followed by 1 full attention layer.
/// This repeats across all 42 layers, giving 35 sliding + 7 full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerType {
    /// Sliding window attention (window=512).
    Sliding,
    /// Full attention (no window, global context).
    Full,
}

impl LayerType {
    /// Returns the layer type for a given layer index based on the 5:1 pattern.
    ///
    /// Pattern: [Sliding, Sliding, Sliding, Sliding, Sliding, Full] repeated.
    /// Layer 0-4: Sliding, Layer 5: Full, Layer 6-10: Sliding, Layer 11: Full, etc.
    pub fn from_index(layer_idx: usize) -> Self {
        // 5:1 pattern: every 6th layer (indices 5, 11, 17, 23, 29, 35, 41) is Full
        let pattern_period = 6;
        let pos_in_period = layer_idx % pattern_period;
        if pos_in_period == pattern_period - 1 {
            LayerType::Full
        } else {
            LayerType::Sliding
        }
    }

    /// Build the full layer type list for a given number of layers.
    pub fn build_layer_types(num_layers: usize) -> Vec<LayerType> {
        (0..num_layers).map(Self::from_index).collect()
    }
}

// ---------------------------------------------------------------------------
// RoPE Parameters
// ---------------------------------------------------------------------------

/// RoPE parameters for a specific attention type.
#[derive(Debug, Config)]
pub struct RopeParams {
    /// Base frequency for RoPE.
    pub rope_theta: f32,

    /// Fraction of head dimensions that get rotary embeddings.
    /// 1.0 = full rotation (sliding), 0.25 = partial rotation (full attention).
    pub partial_rotary_factor: f32,
}

impl RopeParams {
    /// Default RoPE for sliding attention: theta=10000, full rotation.
    pub fn sliding() -> Self {
        Self::new(10000.0, 1.0)
    }

    /// RoPE for full attention: theta=1000000, partial rotation (25%).
    pub fn full() -> Self {
        Self::new(1_000_000.0, 0.25)
    }
}

// ---------------------------------------------------------------------------
// Gemma 4 Config
// ---------------------------------------------------------------------------

/// Gemma 4 model hyperparameters.
///
/// Matches the HuggingFace `config.json` text_config fields for `google/gemma-4-4b-it`.
#[derive(Config, Debug)]
pub struct Gemma4Config {
    /// Vocabulary size (262,144 for Gemma 4).
    pub vocab_size: usize,

    /// Hidden dimension (2560 for Gemma 4 E4B).
    pub hidden_size: usize,

    /// Number of transformer decoder layers (42 for Gemma 4 E4B).
    pub num_hidden_layers: usize,

    /// Intermediate (MLP) dimension (10240 for Gemma 4 E4B).
    pub intermediate_size: usize,

    /// Number of query attention heads (8 for Gemma 4 E4B).
    pub num_attention_heads: usize,

    /// Number of key/value attention heads for sliding attention (2 for Gemma 4 E4B).
    pub num_key_value_heads: usize,

    /// Dimension per attention head for sliding attention (256 for Gemma 4 E4B).
    pub head_dim: usize,

    /// Dimension per attention head for full attention layers (512 for Gemma 4 E4B).
    /// When 0, defaults to `head_dim`.
    #[config(default = 0)]
    pub global_head_dim: usize,

    /// RMS normalization epsilon.
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,

    /// Final logit softcapping value (30.0 for Gemma 4).
    #[config(default = 30.0)]
    pub final_logit_softcapping: f64,

    /// Sliding window size for local attention (512 for Gemma 4).
    #[config(default = 512)]
    pub sliding_window: usize,

    /// Number of consecutive layers at the end that share KV projections.
    /// Gemma 4 E4B: 18 layers (layers 24-41 share KV from layers 0-23).
    #[config(default = 0)]
    pub num_kv_shared_layers: usize,

    /// Hidden size for per-layer input embeddings (PLE).
    /// 256 for Gemma 4 E4B, 0 to disable PLE.
    #[config(default = 0)]
    pub hidden_size_per_layer_input: usize,

    /// Vocabulary size for per-layer input embeddings.
    #[config(default = 0)]
    pub vocab_size_per_layer_input: usize,

    /// Maximum position embeddings (131072 for Gemma 4).
    #[config(default = 131072)]
    pub max_position_embeddings: usize,

    /// Whether to tie word embeddings (lm_head shares embed_tokens).
    #[config(default = true)]
    pub tie_word_embeddings: bool,
}

impl Gemma4Config {
    // -----------------------------------------------------------------------
    // Derived accessors
    // -----------------------------------------------------------------------

    /// Returns the effective head dimension for a given layer type.
    ///
    /// Full attention uses `global_head_dim` (512), sliding uses `head_dim` (256).
    pub fn head_dim_for(&self, layer_type: LayerType) -> usize {
        match layer_type {
            LayerType::Full if self.global_head_dim > 0 => self.global_head_dim,
            _ => self.head_dim,
        }
    }

    /// Returns the effective number of KV heads for a given layer type.
    /// Currently same for both, but could differ for future variants.
    pub fn num_kv_heads_for(&self, _layer_type: LayerType) -> usize {
        self.num_key_value_heads
    }

    /// Returns the KV group ratio (num_heads / num_kv_heads).
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Returns the RoPE parameters for a given layer type.
    pub fn rope_params_for(&self, layer_type: LayerType) -> RopeParams {
        match layer_type {
            LayerType::Sliding => RopeParams::sliding(),
            LayerType::Full => RopeParams::full(),
        }
    }

    /// Returns the layer type for a given layer index.
    pub fn layer_type(&self, layer_idx: usize) -> LayerType {
        LayerType::from_index(layer_idx)
    }

    /// Returns the full layer type list.
    pub fn layer_types(&self) -> Vec<LayerType> {
        LayerType::build_layer_types(self.num_hidden_layers)
    }

    /// Returns the first non-shared layer index (first layer with its own KV).
    /// Layers before this index have their own k_proj/v_proj.
    pub fn first_kv_shared_layer(&self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.num_kv_shared_layers)
    }

    /// Returns true if a layer has its own KV projections (not shared).
    pub fn has_own_kv(&self, layer_idx: usize) -> bool {
        layer_idx < self.first_kv_shared_layer()
    }

    /// Builds the KV sharing map: for each layer, the index of the layer it shares KV with.
    ///
    /// For non-shared layers, maps to themselves.
    /// For shared layers, maps to the first non-shared layer of the same type.
    pub fn kv_source_map(&self) -> Vec<usize> {
        let layer_types = self.layer_types();
        let first_shared = self.first_kv_shared_layer();

        // Find the first non-shared layer of each type
        let mut first_of_type = [None, None]; // [Sliding, Full]
        for (i, lt) in layer_types.iter().enumerate().take(first_shared) {
            let idx = match lt {
                LayerType::Sliding => 0,
                LayerType::Full => 1,
            };
            if first_of_type[idx].is_none() {
                first_of_type[idx] = Some(i);
            }
        }

        (0..self.num_hidden_layers)
            .map(|i| {
                if i < first_shared {
                    i // Own KV
                } else {
                    // Share KV from first non-shared layer of same type
                    let lt = layer_types[i];
                    let idx = match lt {
                        LayerType::Sliding => 0,
                        LayerType::Full => 1,
                    };
                    first_of_type[idx].unwrap_or(i)
                }
            })
            .collect()
    }

    /// Returns true if Per-Layer Embeddings (PLE) are enabled.
    pub fn has_ple(&self) -> bool {
        self.hidden_size_per_layer_input > 0
    }

    /// Attention scale factor: Gemma 4 uses 1.0 (no sqrt scaling).
    pub fn attention_scale(&self) -> f64 {
        1.0
    }

    // -----------------------------------------------------------------------
    // Presets
    // -----------------------------------------------------------------------

    /// Gemma 4 E4B (4.5B effective) configuration.
    ///
    /// 42 layers, 2560 hidden, 262K vocab, 5:1 sliding/full pattern,
    /// 18 KV-shared layers, Per-Layer Embeddings.
    pub fn gemma4_e4b() -> Self {
        Self::new(
            262144, // vocab_size
            2560,   // hidden_size
            42,     // num_hidden_layers
            10240,  // intermediate_size
            8,      // num_attention_heads
            2,      // num_key_value_heads
            256,    // head_dim
        )
        .with_global_head_dim(512)
        .with_sliding_window(512)
        .with_num_kv_shared_layers(18)
        .with_hidden_size_per_layer_input(256)
        .with_vocab_size_per_layer_input(262144)
        .with_max_position_embeddings(131072)
        .with_final_logit_softcapping(30.0)
        .with_tie_word_embeddings(true)
    }
}

// ---------------------------------------------------------------------------
// LoRA Target
// ---------------------------------------------------------------------------

/// Enumeration of LoRA target modules for Gemma 4.
///
/// Same 7 targets as Gemma 2: q/k/v/o projections (attention) + gate/up/down (MLP).
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma4_e4b_config() {
        let config = Gemma4Config::gemma4_e4b();
        assert_eq!(config.vocab_size, 262144);
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 42);
        assert_eq!(config.intermediate_size, 10240);
        assert_eq!(config.num_attention_heads, 8);
        assert_eq!(config.num_key_value_heads, 2);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.global_head_dim, 512);
        assert_eq!(config.sliding_window, 512);
        assert_eq!(config.num_kv_shared_layers, 18);
        assert_eq!(config.hidden_size_per_layer_input, 256);
        assert_eq!(config.final_logit_softcapping, 30.0);
    }

    #[test]
    fn test_layer_types_5_to_1_pattern() {
        let types = LayerType::build_layer_types(42);
        assert_eq!(types.len(), 42);
        assert_eq!(
            types.iter().filter(|t| **t == LayerType::Sliding).count(),
            35
        );
        assert_eq!(types.iter().filter(|t| **t == LayerType::Full).count(), 7);

        // Verify pattern: full at indices 5, 11, 17, 23, 29, 35, 41
        assert_eq!(types[0], LayerType::Sliding);
        assert_eq!(types[4], LayerType::Sliding);
        assert_eq!(types[5], LayerType::Full);
        assert_eq!(types[6], LayerType::Sliding);
        assert_eq!(types[10], LayerType::Sliding);
        assert_eq!(types[11], LayerType::Full);
        assert_eq!(types[41], LayerType::Full);
    }

    #[test]
    fn test_head_dim_per_layer_type() {
        let config = Gemma4Config::gemma4_e4b();

        // Sliding uses head_dim = 256
        assert_eq!(config.head_dim_for(LayerType::Sliding), 256);
        // Full uses global_head_dim = 512
        assert_eq!(config.head_dim_for(LayerType::Full), 512);
    }

    #[test]
    fn test_kv_groups() {
        let config = Gemma4Config::gemma4_e4b();
        assert_eq!(config.num_kv_groups(), 4); // 8 / 2 = 4
    }

    #[test]
    fn test_attention_scale_is_one() {
        let config = Gemma4Config::gemma4_e4b();
        assert_eq!(config.attention_scale(), 1.0);
    }

    #[test]
    fn test_rope_params() {
        let config = Gemma4Config::gemma4_e4b();

        let sliding = config.rope_params_for(LayerType::Sliding);
        assert_eq!(sliding.rope_theta, 10000.0);
        assert!((sliding.partial_rotary_factor - 1.0).abs() < f32::EPSILON);

        let full = config.rope_params_for(LayerType::Full);
        assert_eq!(full.rope_theta, 1000000.0);
        assert!((full.partial_rotary_factor - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_first_kv_shared_layer() {
        let config = Gemma4Config::gemma4_e4b();
        assert_eq!(config.first_kv_shared_layer(), 24); // 42 - 18 = 24
    }

    #[test]
    fn test_has_own_kv() {
        let config = Gemma4Config::gemma4_e4b();
        assert!(config.has_own_kv(0));
        assert!(config.has_own_kv(23));
        assert!(!config.has_own_kv(24));
        assert!(!config.has_own_kv(41));
    }

    #[test]
    fn test_kv_source_map() {
        let config = Gemma4Config::gemma4_e4b();
        let map = config.kv_source_map();

        assert_eq!(map.len(), 42);

        // Non-shared layers map to themselves
        assert_eq!(map[0], 0);
        assert_eq!(map[5], 5);
        assert_eq!(map[23], 23);

        // Layer 24 is sliding → shares from layer 0 (first sliding)
        assert_eq!(map[24], 0);
        // Layer 25 is sliding → shares from layer 0
        assert_eq!(map[25], 0);
        // Layer 29 is full → shares from layer 5 (first full)
        assert_eq!(map[29], 5);
        // Layer 30 is sliding → shares from layer 0
        assert_eq!(map[30], 0);
        // Layer 35 is full → shares from layer 5
        assert_eq!(map[35], 5);
        // Layer 41 is full → shares from layer 5
        assert_eq!(map[41], 5);
    }

    #[test]
    fn test_has_ple() {
        let config = Gemma4Config::gemma4_e4b();
        assert!(config.has_ple());

        let config_no_ple = Gemma4Config::new(262144, 2560, 42, 10240, 8, 2, 256);
        assert!(!config_no_ple.has_ple());
    }

    #[test]
    fn test_lora_target_names() {
        assert_eq!(LoraTarget::QProj.weight_name(), "q_proj");
        assert_eq!(LoraTarget::KProj.weight_name(), "k_proj");
        assert_eq!(LoraTarget::VProj.weight_name(), "v_proj");
        assert_eq!(LoraTarget::OProj.weight_name(), "o_proj");
        assert_eq!(LoraTarget::GateProj.weight_name(), "gate_proj");
        assert_eq!(LoraTarget::UpProj.weight_name(), "up_proj");
        assert_eq!(LoraTarget::DownProj.weight_name(), "down_proj");
    }

    #[test]
    fn test_lora_all_targets_count() {
        assert_eq!(LoraTarget::all_targets().len(), 7);
        assert_eq!(LoraTarget::attention_targets().len(), 4);
        assert_eq!(LoraTarget::mlp_targets().len(), 3);
    }

    #[test]
    fn test_layer_type_from_index_short_sequence() {
        // Verify pattern holds for small counts
        assert_eq!(LayerType::from_index(0), LayerType::Sliding);
        assert_eq!(LayerType::from_index(4), LayerType::Sliding);
        assert_eq!(LayerType::from_index(5), LayerType::Full);
        assert_eq!(LayerType::from_index(6), LayerType::Sliding);
    }
}
