//! Gemma 4 model with LoRA fine-tuning support for burn.
//!
//! Implements the Gemma 4 architecture (Google, 2025) with LoRA (Low-Rank Adaptation)
//! for parameter-efficient fine-tuning on macOS Metal via burn's WGPU backend.
//!
//! # Architecture
//!
//! Gemma 4 features (vs Gemma 2):
//! - 5:1 sliding/full attention pattern (vs 1:1 alternating)
//! - Q/K/V RMSNorm in attention (new)
//! - KV sharing across last N layers (new)
//! - Per-Layer Embeddings / PLE (new)
//! - Layer scalar per decoder layer (new)
//! - Global head dim (512) for full attention layers
//! - Different RoPE for sliding vs full attention (partial rotation)
//! - No attention logit softcapping (Gemma 2 had 50.0)
//! - Final logit softcapping kept at 30.0
//!
//! # Usage
//!
//! ```ignore
//! use lora_gemma4::{Gemma4Config, Gemma4Model};
//! use burn::backend::{Autodiff, Wgpu};
//!
//! type B = Autodiff<Wgpu>;
//! let device = Default::default();
//!
//! let config = Gemma4Config::gemma4_e4b();
//! let model = Gemma4Model::new(&config, &device);
//!
//! let input_ids = Tensor::<B, 2, Int>::zeros([1, 16], &device);
//! let logits = model.forward(input_ids);
//! ```

extern crate alloc;

pub mod batcher;
pub mod dataset;
pub mod loader;
pub mod model;
pub mod model_lora;
pub mod tokenizer;
pub mod types;

pub use batcher::{SFTBatch, SFTBatcher};
pub use dataset::{ChatItem, ChatMessageSerde, DatasetError, JsonlDataset};
pub use loader::{LoadError, LoadReport};
pub use model::{Gemma4Attention, Gemma4Block, Gemma4MLP, Gemma4Model};
pub use model_lora::{
    Gemma4ForSFT, Gemma4ModelLora, apply_lora_to_gemma4, count_lora_params, count_total_params,
};
pub use tokenizer::{ChatMessage, GemmaTokenizer, Role, TokenizerError};
pub use types::{Gemma4Config, LayerType, LoraTarget, RopeParams};
