//! Gemma 2 model with LoRA fine-tuning support for burn.
//!
//! Implements the Gemma 2 architecture (Google, 2024) with LoRA (Low-Rank Adaptation)
//! for parameter-efficient fine-tuning on macOS Metal via burn's WGPU backend.
//!
//! # Architecture
//!
//! Gemma 2 features:
//! - Grouped Query Attention (GQA)
//! - Attention logit softcapping
//! - Final logit softcapping
//! - Sandwich normalization (post-attention + post-MLP RMSNorm)
//! - RoPE (Rotary Position Embeddings)
//!
//! # Usage
//!
//! ```ignore
//! use lora_gemma2::{Gemma2Config, Gemma2Model};
//! use burn::backend::{Autodiff, Wgpu};
//!
//! type B = Autodiff<Wgpu>;
//! let device = Default::default();
//!
//! let config = Gemma2Config::gemma2_2b();
//! let model = Gemma2Model::new(&config, &device);
//!
//! let input_ids = Tensor::<B, 2, Int>::zeros([1, 16], &device);
//! let logits = model.forward(input_ids);
//! ```

extern crate alloc;

pub mod batcher;
pub mod dataset;
pub mod loader;
pub mod model;
pub mod tokenizer;
pub mod types;

pub use batcher::{SFTBatch, SFTBatcher, SFTTrainingBatch};
pub use dataset::{ChatItem, ChatMessageSerde, DatasetError, JsonlDataset};
pub use loader::{LoadError, LoadReport};
pub use model::{Gemma2Attention, Gemma2Block, Gemma2MLP, Gemma2Model};
pub use tokenizer::{ChatMessage, GemmaTokenizer, Role, TokenizerError};
pub use types::{Gemma2Config, LoraTarget};
