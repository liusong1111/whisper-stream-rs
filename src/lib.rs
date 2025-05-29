//! whisper-stream-rs
//!
//! A library for performing real-time transcription using Whisper ASR models.
//! It handles audio capture, processing, and streaming results.

mod audio;
mod model;
mod error;
mod audio_utils;
mod score;
mod whisper_stream;
// New public API
pub use whisper_stream::{WhisperStream, Event};
pub use error::WhisperStreamError;
pub use model::Model;
