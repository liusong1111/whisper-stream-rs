//! Real-time streaming transcription library using whisper-rs and cpal

mod model;
pub mod audio;
pub mod stream;

pub use stream::{start_transcription_stream, TranscriptionStreamEvent};
