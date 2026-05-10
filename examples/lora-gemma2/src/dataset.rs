//! JSONL dataset loader for LoRA fine-tuning.
//!
//! Loads training data from JSONL files in formats compatible with
//! [riir-burner](https://github.com/katopz/riir-burner) corpus output:
//!
//! - `{"messages": [{"role": "user", "content": "..."}, ...]}` — OpenAI format
//! - `{"conversations": [{"role": "user", "content": "..."}, ...]}` — alternative format
//! - `{"instruction": "...", "input": "...", "output": "..."}` — Alpaca format
//!
//! # Example
//!
//! ```ignore
//! use lora_gemma2::dataset::JsonlDataset;
//!
//! let dataset = JsonlDataset::from_file("input/train.jsonl").unwrap();
//! println!("Loaded {} samples", dataset.len());
//!
//! let (train, val) = dataset.split(0.9);
//! println!("Train: {}, Val: {}", train.len(), val.len());
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use burn::data::dataset::Dataset;
use serde::{Deserialize, Serialize};

use crate::tokenizer::{ChatMessage, Role};

// ---------------------------------------------------------------------------
// Dataset Item
// ---------------------------------------------------------------------------

/// A single training sample containing a conversation as chat messages.
///
/// Each item maps to one line in the JSONL file after format conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatItem {
    /// Ordered list of chat messages forming a conversation.
    pub messages: Vec<ChatMessageSerde>,
}

/// Serializable chat message (mirrors [`ChatMessage`] but with Serde support).
///
/// The `role` field accepts: `"user"`, `"assistant"`, `"model"`, `"system"`.
/// Both `"assistant"` and `"model"` map to [`Role::Assistant`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessageSerde {
    /// Sender role: `"user"`, `"assistant"` / `"model"`, `"system"`.
    pub role: String,
    /// Message text content.
    pub content: String,
}

impl ChatMessageSerde {
    /// Convert to a typed [`ChatMessage`].
    ///
    /// Returns `None` if the role string is unrecognized.
    pub fn to_chat_message(&self) -> Option<ChatMessage> {
        let role: Role = self.role.parse().ok()?;
        Some(ChatMessage::new(role, &self.content))
    }
}

// ---------------------------------------------------------------------------
// Raw JSONL formats (flexible deserialization)
// ---------------------------------------------------------------------------

/// Raw JSONL record supporting multiple corpus formats.
///
/// Fields are optional; the loader picks the first matching format:
/// 1. `messages` — OpenAI chat format
/// 2. `conversations` — Alternative chat format
/// 3. `instruction` + `output` — Alpaca instruction format
/// 4. `text` — Plain text (wrapped as a single user message)
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRecord {
    /// OpenAI-style messages array.
    messages: Option<Vec<ChatMessageSerde>>,
    /// Alternative conversations array.
    conversations: Option<Vec<ChatMessageSerde>>,
    /// Instruction text (Alpaca format).
    instruction: Option<String>,
    /// Optional input text (Alpaca format).
    input: Option<String>,
    /// Output/response text (Alpaca format).
    output: Option<String>,
    /// Pre-formatted text (wrapped as user message).
    text: Option<String>,
}

impl RawRecord {
    /// Convert raw record into a [`ChatItem`] with normalized messages.
    ///
    /// Returns `None` if the record has no usable content.
    fn to_chat_item(&self) -> Option<ChatItem> {
        // Format 1: messages array
        if let Some(messages) = &self.messages
            && !messages.is_empty()
        {
            return Some(ChatItem {
                messages: messages.clone(),
            });
        }

        // Format 2: conversations array
        if let Some(conversations) = &self.conversations
            && !conversations.is_empty()
        {
            return Some(ChatItem {
                messages: conversations.clone(),
            });
        }

        // Format 3: instruction/output (Alpaca)
        let instruction = self.instruction.as_deref().unwrap_or("").trim();
        let output = self.output.as_deref().unwrap_or("").trim();
        if !instruction.is_empty() || !output.is_empty() {
            let input_text = self.input.as_deref().unwrap_or("").trim();
            let content = if !input_text.is_empty() {
                format!("{instruction}\n\n{input_text}")
            } else {
                instruction.to_string()
            };

            return Some(ChatItem {
                messages: vec![
                    ChatMessageSerde {
                        role: "user".into(),
                        content,
                    },
                    ChatMessageSerde {
                        role: "assistant".into(),
                        content: output.to_string(),
                    },
                ],
            });
        }

        // Format 4: plain text
        if let Some(text) = &self.text
            && !text.trim().is_empty()
        {
            return Some(ChatItem {
                messages: vec![ChatMessageSerde {
                    role: "user".into(),
                    content: text.clone(),
                }],
            });
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Dataset Error
// ---------------------------------------------------------------------------

/// Errors that can occur during dataset loading.
#[derive(Debug)]
pub enum DatasetError {
    /// The specified file was not found.
    FileNotFound(String),
    /// An I/O error occurred.
    Io(std::io::Error),
    /// A JSON parse error on a specific line.
    Parse {
        /// Line number (1-based).
        line: usize,
        /// Error description.
        message: String,
    },
    /// No valid records found in the file.
    Empty,
}

impl std::fmt::Display for DatasetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetError::FileNotFound(path) => write!(f, "File not found: {path}"),
            DatasetError::Io(e) => write!(f, "I/O error: {e}"),
            DatasetError::Parse { line, message } => {
                write!(f, "JSON parse error at line {line}: {message}")
            }
            DatasetError::Empty => write!(f, "No valid records found in file"),
        }
    }
}

impl std::error::Error for DatasetError {}

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

/// In-memory JSONL dataset of chat items.
///
/// Loads the entire file into memory for fast random access via the [`Dataset`] trait.
/// Suitable for datasets that fit in RAM (typical for LoRA fine-tuning).
#[derive(Debug)]
pub struct JsonlDataset {
    items: Vec<ChatItem>,
}

impl JsonlDataset {
    /// Load a JSONL dataset from a file path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, DatasetError> {
        let path_str = path.as_ref().display().to_string();
        if !path.as_ref().exists() {
            return Err(DatasetError::FileNotFound(path_str));
        }

        let file = File::open(path).map_err(DatasetError::Io)?;
        let reader = BufReader::new(file);
        Self::from_reader(reader)
    }

    /// Load from any `BufRead` source (file, string, etc.).
    pub fn from_reader<R: BufRead>(reader: R) -> Result<Self, DatasetError> {
        let mut items = Vec::new();
        let mut warnings = 0u32;

        for (line_num, line_result) in reader.lines().enumerate() {
            let line_num = line_num + 1; // 1-based
            let line = match line_result {
                Ok(l) => l,
                Err(e) => return Err(DatasetError::Io(e)),
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            match serde_json::from_str::<RawRecord>(trimmed) {
                Ok(raw) => {
                    if let Some(item) = raw.to_chat_item() {
                        items.push(item);
                    } else {
                        warnings += 1;
                        log::debug!("Skipping empty record at line {line_num}");
                    }
                }
                Err(e) => {
                    return Err(DatasetError::Parse {
                        line: line_num,
                        message: format!("{e}"),
                    });
                }
            }
        }

        if warnings > 0 {
            log::warn!("Skipped {warnings} empty/malformed records");
        }

        if items.is_empty() {
            return Err(DatasetError::Empty);
        }

        log::info!("Loaded {} chat items from JSONL", items.len());
        Ok(Self { items })
    }

    /// Split the dataset into train and validation sets.
    ///
    /// `train_ratio` is the fraction for training (e.g., 0.9 for 90% train, 10% val).
    /// The split is deterministic (no shuffling) — shuffle beforehand if needed.
    pub fn split(&self, train_ratio: f32) -> (JsonlDataset, JsonlDataset) {
        assert!(
            (0.0..=1.0).contains(&train_ratio),
            "train_ratio must be between 0.0 and 1.0"
        );

        let split_point = (self.items.len() as f32 * train_ratio).round() as usize;
        let train_items = self.items[..split_point].to_vec();
        let val_items = self.items[split_point..].to_vec();

        log::info!(
            "Split: {} train, {} val (ratio={train_ratio})",
            train_items.len(),
            val_items.len()
        );

        (
            JsonlDataset { items: train_items },
            JsonlDataset { items: val_items },
        )
    }

    /// Number of items in the dataset.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get a reference to the underlying items.
    pub fn items(&self) -> &[ChatItem] {
        &self.items
    }

    /// Create a dataset from a vector of items (for testing).
    pub fn from_items(items: Vec<ChatItem>) -> Self {
        Self { items }
    }
}

impl Dataset<ChatItem> for JsonlDataset {
    fn get(&self, index: usize) -> Option<ChatItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Helper: create a reader from a JSONL string.
    fn reader(jsonl: &str) -> Cursor<&str> {
        Cursor::new(jsonl)
    }

    // --- RawRecord conversion tests ---

    #[test]
    fn test_messages_format() {
        let jsonl = r#"{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"}]}"#;
        let dataset = JsonlDataset::from_reader(reader(jsonl)).unwrap();
        assert_eq!(dataset.len(), 1);
        let item = dataset.get(0).unwrap();
        assert_eq!(item.messages.len(), 2);
        assert_eq!(item.messages[0].role, "user");
        assert_eq!(item.messages[0].content, "Hello");
        assert_eq!(item.messages[1].role, "assistant");
        assert_eq!(item.messages[1].content, "Hi");
    }

    #[test]
    fn test_conversations_format() {
        let jsonl =
            r#"{"conversations":[{"role":"user","content":"Q"},{"role":"model","content":"A"}]}"#;
        let dataset = JsonlDataset::from_reader(reader(jsonl)).unwrap();
        assert_eq!(dataset.len(), 1);
        let item = dataset.get(0).unwrap();
        assert_eq!(item.messages[0].role, "user");
        assert_eq!(item.messages[1].role, "model");
    }

    #[test]
    fn test_alpaca_format() {
        let jsonl = r#"{"instruction":"What is Rust?","output":"A language"}"#;
        let dataset = JsonlDataset::from_reader(reader(jsonl)).unwrap();
        assert_eq!(dataset.len(), 1);
        let item = dataset.get(0).unwrap();
        assert_eq!(item.messages.len(), 2);
        assert_eq!(item.messages[0].role, "user");
        assert_eq!(item.messages[0].content, "What is Rust?");
        assert_eq!(item.messages[1].role, "assistant");
        assert_eq!(item.messages[1].content, "A language");
    }

    #[test]
    fn test_alpaca_format_with_input() {
        let jsonl = r#"{"instruction":"Translate","input":"hello","output":"hola"}"#;
        let dataset = JsonlDataset::from_reader(reader(jsonl)).unwrap();
        let item = dataset.get(0).unwrap();
        assert_eq!(item.messages[0].content, "Translate\n\nhello");
    }

    #[test]
    fn test_text_format() {
        let jsonl = r#"{"text":"Some raw text content"}"#;
        let dataset = JsonlDataset::from_reader(reader(jsonl)).unwrap();
        let item = dataset.get(0).unwrap();
        assert_eq!(item.messages.len(), 1);
        assert_eq!(item.messages[0].role, "user");
        assert_eq!(item.messages[0].content, "Some raw text content");
    }

    #[test]
    fn test_multi_line_jsonl() {
        let jsonl = r#"{"messages":[{"role":"user","content":"A"}]}
{"messages":[{"role":"user","content":"B"}]}
{"messages":[{"role":"user","content":"C"}]}"#;
        let dataset = JsonlDataset::from_reader(reader(jsonl)).unwrap();
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get(0).unwrap().messages[0].content, "A");
        assert_eq!(dataset.get(2).unwrap().messages[0].content, "C");
    }

    #[test]
    fn test_empty_lines_skipped() {
        let jsonl = r#"{"messages":[{"role":"user","content":"A"}]}

{"messages":[{"role":"user","content":"B"}]}"#;
        let dataset = JsonlDataset::from_reader(reader(jsonl)).unwrap();
        assert_eq!(dataset.len(), 2);
    }

    #[test]
    fn test_empty_record_skipped() {
        // instruction + output both empty => no usable content
        let jsonl = r#"{"instruction":"","output":""}
{"messages":[{"role":"user","content":"Valid"}]}"#;
        let dataset = JsonlDataset::from_reader(reader(jsonl)).unwrap();
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_parse_error() {
        let jsonl = r#"{"invalid json"#;
        let result = JsonlDataset::from_reader(reader(jsonl));
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            DatasetError::Parse { line, .. } => assert_eq!(line, 1),
            other => panic!("Expected Parse error, got: {other}"),
        }
    }

    #[test]
    fn test_empty_file() {
        let jsonl = "";
        let result = JsonlDataset::from_reader(reader(jsonl));
        assert!(matches!(result, Err(DatasetError::Empty)));
    }

    #[test]
    fn test_file_not_found() {
        let result = JsonlDataset::from_file("/nonexistent/path/train.jsonl");
        assert!(matches!(result, Err(DatasetError::FileNotFound(_))));
    }

    // --- Split tests ---

    #[test]
    fn test_split_90_10() {
        let items: Vec<ChatItem> = (0..100)
            .map(|i| ChatItem {
                messages: vec![ChatMessageSerde {
                    role: "user".into(),
                    content: format!("item_{i}"),
                }],
            })
            .collect();

        let dataset = JsonlDataset::from_items(items);
        let (train, val) = dataset.split(0.9);

        assert_eq!(train.len(), 90);
        assert_eq!(val.len(), 10);
        // Verify content preserved
        assert_eq!(train.get(0).unwrap().messages[0].content, "item_0");
        assert_eq!(val.get(0).unwrap().messages[0].content, "item_90");
    }

    #[test]
    fn test_split_single_item() {
        let dataset = JsonlDataset::from_items(vec![ChatItem {
            messages: vec![ChatMessageSerde {
                role: "user".into(),
                content: "only".into(),
            }],
        }]);
        let (train, val) = dataset.split(0.9);
        assert_eq!(train.len(), 1);
        assert_eq!(val.len(), 0);
    }

    // --- ChatMessageSerde conversion ---

    #[test]
    fn test_message_to_chat_message() {
        let serde_msg = ChatMessageSerde {
            role: "user".into(),
            content: "Hello".into(),
        };
        let chat_msg = serde_msg.to_chat_message().unwrap();
        assert_eq!(chat_msg.role, Role::User);
        assert_eq!(chat_msg.content, "Hello");
    }

    #[test]
    fn test_message_to_chat_message_model_role() {
        let serde_msg = ChatMessageSerde {
            role: "model".into(),
            content: "Response".into(),
        };
        let chat_msg = serde_msg.to_chat_message().unwrap();
        assert_eq!(chat_msg.role, Role::Assistant);
    }

    #[test]
    fn test_message_to_chat_message_unknown_role() {
        let serde_msg = ChatMessageSerde {
            role: "unknown".into(),
            content: "???".into(),
        };
        assert!(serde_msg.to_chat_message().is_none());
    }

    // --- Error display tests ---

    #[test]
    fn test_dataset_error_display() {
        let err = DatasetError::FileNotFound("missing.jsonl".into());
        assert!(err.to_string().contains("missing.jsonl"));

        let err = DatasetError::Parse {
            line: 42,
            message: "bad json".into(),
        };
        assert!(err.to_string().contains("line 42"));

        let err = DatasetError::Empty;
        assert!(err.to_string().contains("No valid records"));
    }

    // --- Format priority test ---

    #[test]
    fn test_messages_format_priority_over_instruction() {
        // When both messages and instruction are present, messages takes priority
        let jsonl = r#"{"messages":[{"role":"user","content":"from messages"}],"instruction":"from instruction","output":"from output"}"#;
        let dataset = JsonlDataset::from_reader(reader(jsonl)).unwrap();
        let item = dataset.get(0).unwrap();
        assert_eq!(item.messages[0].content, "from messages");
    }
}
