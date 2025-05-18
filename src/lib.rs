//! Real-time streaming transcription library using whisper-rs and cpal

mod model;
mod audio;
mod stream;

pub use stream::start_transcription_stream;
