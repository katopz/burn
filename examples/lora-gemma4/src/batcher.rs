//! Supervised Fine-Tuning (SFT) batcher for LoRA training on Gemma 4.
//!
//! Converts [`ChatItem`] conversations into tensor batches for next-token
//! prediction training:
//!
//! 1. Format each conversation with the Gemma chat template
//! 2. Tokenize the formatted text
//! 3. Truncate to `max_seq_length` tokens
//! 4. Pad shorter sequences to uniform length
//! 5. Create input/target pairs (shift by 1 for autoregressive training)
//!
//! # Differences from Gemma 2
//!
//! Gemma 4's [`SFTTrainingBatch`](crate::model_lora::SFTTrainingBatch) has no
//! `mask_pad` tensor. Padding is handled by `CrossEntropyLossConfig::with_pad_tokens`
//! which ignores target positions matching `pad_token_id`.
//!
//! # Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use lora_gemma4::tokenizer::GemmaTokenizer;
//! use lora_gemma4::batcher::SFTBatcher;
//! use lora_gemma4::dataset::ChatItem;
//! use burn::data::dataloader::batcher::Batcher;
//!
//! let tokenizer = Arc::new(GemmaTokenizer::from_pretrained("unsloth/gemma-4-E4B-it")?);
//! let batcher = SFTBatcher::new(tokenizer, 2048);
//!
//! let items = vec![dataset.get(0).unwrap()];
//! let batch = batcher.batch(items, &device);
//! // batch.tokens_inputs: [batch, seq_len-1] input tokens
//! // batch.targets:       [batch, seq_len-1] target tokens (shifted by 1)
//! ```

use std::sync::Arc;

use burn::data::dataloader::batcher::Batcher;
use burn::nn::attention::generate_padding_mask;
use burn::prelude::*;

use crate::dataset::ChatItem;
use crate::model_lora::SFTTrainingBatch;
use crate::tokenizer::GemmaTokenizer;

// ---------------------------------------------------------------------------
// Inference Batch
// ---------------------------------------------------------------------------

/// Inference batch: padded tokens + padding mask.
///
/// Used for forward passes without training (e.g., evaluation, generation).
#[derive(Debug, Clone)]
pub struct SFTBatch<B: Backend> {
    /// Token IDs: `[batch_size, seq_len]`.
    pub tokens: Tensor<B, 2, Int>,
    /// Padding mask: `[batch_size, seq_len]`.
    /// `true` where tokens are padding (should be ignored).
    pub mask_pad: Tensor<B, 2, Bool>,
}

// ---------------------------------------------------------------------------
// Batcher
// ---------------------------------------------------------------------------

/// Supervised Fine-Tuning batcher for Gemma 4.
///
/// Converts [`ChatItem`] conversations into padded tensor batches for
/// Gemma 4 training. Uses the Gemma chat template for formatting.
///
/// Training batches use [`SFTTrainingBatch`] from `model_lora`, which relies
/// on `CrossEntropyLossConfig::with_pad_tokens` for padding handling
/// (no explicit `mask_pad` tensor).
pub struct SFTBatcher {
    tokenizer: Arc<GemmaTokenizer>,
    max_seq_length: usize,
}

impl SFTBatcher {
    /// Create a new SFT batcher.
    ///
    /// # Arguments
    /// * `tokenizer` — Shared tokenizer instance (wrapped in `Arc` for thread safety)
    /// * `max_seq_length` — Maximum sequence length. Sequences are truncated to this length.
    pub fn new(tokenizer: Arc<GemmaTokenizer>, max_seq_length: usize) -> Self {
        Self {
            tokenizer,
            max_seq_length,
        }
    }

    /// Reference to the underlying tokenizer.
    pub fn tokenizer(&self) -> &GemmaTokenizer {
        &self.tokenizer
    }

    /// Maximum sequence length.
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Tokenize a single [`ChatItem`] into token IDs using the Gemma chat template.
    ///
    /// Converts the item's messages to [`ChatMessage`](crate::tokenizer::ChatMessage)s,
    /// formats with the chat template, and encodes to token IDs.
    /// Truncates to `max_seq_length`.
    fn tokenize_item(&self, item: &ChatItem) -> Vec<usize> {
        let messages: Vec<_> = item
            .messages
            .iter()
            .filter_map(|m| m.to_chat_message())
            .collect();

        if messages.is_empty() {
            return vec![self.tokenizer.bos_token_id()];
        }

        let tokens = self.tokenizer.encode_chat(&messages, false);

        // Truncate to max_seq_length
        if tokens.len() > self.max_seq_length {
            tokens[..self.max_seq_length].to_vec()
        } else {
            tokens
        }
    }
}

// ---------------------------------------------------------------------------
// Batcher Trait Implementations
// ---------------------------------------------------------------------------

impl<B: Backend> Batcher<B, ChatItem, SFTBatch<B>> for SFTBatcher {
    fn batch(&self, items: Vec<ChatItem>, device: &B::Device) -> SFTBatch<B> {
        let tokens_list: Vec<Vec<usize>> =
            items.iter().map(|item| self.tokenize_item(item)).collect();

        let mask = generate_padding_mask(
            self.tokenizer.pad_token_id(),
            tokens_list,
            Some(self.max_seq_length),
            device,
        );

        SFTBatch {
            tokens: mask.tensor,
            mask_pad: mask.mask,
        }
    }
}

impl<B: Backend> Batcher<B, ChatItem, SFTTrainingBatch<B>> for SFTBatcher {
    fn batch(&self, items: Vec<ChatItem>, device: &B::Device) -> SFTTrainingBatch<B> {
        // First create an inference batch (padded tokens + mask)
        let inference_batch: SFTBatch<B> =
            Batcher::<B, ChatItem, SFTBatch<B>>::batch(self, items, device);

        let [batch_size, seq_length] = inference_batch.tokens.dims();

        // Ensure seq_length > 1 for the shift operation
        if seq_length <= 1 {
            panic!(
                "Sequence length must be > 1 for next-token prediction, got {seq_length}. \
                 Consider increasing max_seq_length or using longer training samples."
            );
        }

        // Next-token prediction: input = tokens[:-1], target = tokens[1:]
        //
        // Gemma 4 uses token-ID-based padding: the loss function ignores
        // target positions matching `pad_token_id` via `with_pad_tokens`.
        // No explicit `mask_pad` tensor needed in the training batch.
        let tokens_inputs = inference_batch
            .tokens
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);
        let targets = inference_batch.tokens.slice([0..batch_size, 1..seq_length]);

        SFTTrainingBatch {
            tokens_inputs,
            targets,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::ChatMessageSerde;

    /// Helper: create a ChatItem with user/assistant messages.
    fn make_item(user: &str, assistant: &str) -> ChatItem {
        ChatItem {
            messages: vec![
                ChatMessageSerde {
                    role: "user".into(),
                    content: user.into(),
                },
                ChatMessageSerde {
                    role: "assistant".into(),
                    content: assistant.into(),
                },
            ],
        }
    }

    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    #[test]
    fn test_tokenize_item_empty_messages() {
        let item = ChatItem { messages: vec![] };
        assert!(item.messages.is_empty());
    }

    #[test]
    fn test_training_batch_structure() {
        // Verify the shift operation logic
        // If tokens = [[1, 2, 3, 4]], then:
        //   inputs  = [[1, 2, 3]]
        //   targets = [[2, 3, 4]]
        // This is the standard next-token prediction pattern.
        let device = device();
        let tokens =
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::from([[1i64, 2, 3, 4]]), &device);

        let [batch_size, seq_length] = tokens.dims();
        assert_eq!(seq_length, 4);

        let tokens_inputs = tokens.clone().slice([0..batch_size, 0..seq_length - 1]);
        let targets = tokens.slice([0..batch_size, 1..seq_length]);

        let input_data: Vec<i64> = tokens_inputs.into_data().to_vec().unwrap();
        let target_data: Vec<i64> = targets.into_data().to_vec().unwrap();

        assert_eq!(input_data, vec![1, 2, 3]);
        assert_eq!(target_data, vec![2, 3, 4]);
    }

    #[test]
    fn test_training_batch_multi_sample() {
        let device = device();
        let tokens = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[10i64, 20, 30], [40, 50, 60]]),
            &device,
        );

        let [batch_size, seq_length] = tokens.dims();
        assert_eq!(batch_size, 2);
        assert_eq!(seq_length, 3);

        let tokens_inputs = tokens.clone().slice([0..batch_size, 0..seq_length - 1]);
        let targets = tokens.slice([0..batch_size, 1..seq_length]);

        let input_data: Vec<i64> = tokens_inputs.into_data().to_vec().unwrap();
        let target_data: Vec<i64> = targets.into_data().to_vec().unwrap();

        assert_eq!(input_data, vec![10, 20, 40, 50]);
        assert_eq!(target_data, vec![20, 30, 50, 60]);
    }

    #[test]
    fn test_chat_item_creation() {
        let item = make_item("Hello", "Hi there!");
        assert_eq!(item.messages.len(), 2);
        assert_eq!(item.messages[0].role, "user");
        assert_eq!(item.messages[1].role, "assistant");
    }

    #[test]
    fn test_sft_batch_clone() {
        let device = device();
        let tokens = Tensor::<TestBackend, 2, Int>::ones([2, 4], &device);
        let mask = Tensor::<TestBackend, 2, Bool>::zeros([2, 4], &device);

        let batch = SFTBatch {
            tokens: tokens.clone(),
            mask_pad: mask.clone(),
        };

        let cloned = batch.clone();
        assert_eq!(cloned.tokens.dims(), [2, 4]);
        assert_eq!(cloned.mask_pad.dims(), [2, 4]);
    }

    #[test]
    fn test_training_batch_clone() {
        let device = device();
        let tokens = Tensor::<TestBackend, 2, Int>::ones([2, 3], &device);

        let batch = SFTTrainingBatch {
            tokens_inputs: tokens.clone(),
            targets: tokens,
        };

        let cloned = batch.clone();
        assert_eq!(cloned.tokens_inputs.dims(), [2, 3]);
        assert_eq!(cloned.targets.dims(), [2, 3]);
    }
}
