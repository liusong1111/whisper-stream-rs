use std::sync::mpsc::{self, Receiver};
use std::thread;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
use hound;

use crate::audio::{AudioInput};
use crate::model::ensure_model;

// Default values for TranscriptionStreamParams
const DEFAULT_RECORD_TO_WAV: Option<String> = None;
// const DEFAULT_LANGUAGE: Option<String> = Some("en".to_string()); // This cannot be const, handled in impl Default
const DEFAULT_STEP_MS: u32 = 800;
const DEFAULT_LENGTH_MS: u32 = 5000;
const DEFAULT_KEEP_MS: u32 = 200;
const DEFAULT_MAX_TOKENS: i32 = 32;
const DEFAULT_N_THREADS: i32 = 4; // Fallback, actual default uses available_parallelism
const DEFAULT_AUDIO_DEVICE_NAME: Option<String> = None;

/// Configuration for the transcription stream.
#[derive(Debug, Clone)]
pub struct TranscriptionStreamParams {
    /// Optional path to save the recorded audio as a WAV file.
    pub record_to_wav: Option<String>,
    /// Target language for transcription. `None` for auto-detection.
    pub language: Option<String>,
    /// Duration of each audio chunk processed by `AudioInput` in milliseconds.
    pub step_ms: u32,
    /// Total duration of the audio window considered for a single transcription, in milliseconds.
    pub length_ms: u32,
    /// Duration of audio from the previous segment to keep for context, in milliseconds.
    pub keep_ms: u32,
    /// Maximum number of tokens to generate per audio segment.
    pub max_tokens: i32,
    /// Number of threads to use for Whisper model computation.
    pub n_threads: i32,
    /// Optional human-readable device name passed to `AudioInput::new`; `None` means default system device.
    pub audio_device_name: Option<String>,
}

impl Default for TranscriptionStreamParams {
    fn default() -> Self {
        // For non-const defaults like String options, initialize them directly here.
        let default_lang = Some("en".to_string());
        let default_n_threads = std::thread::available_parallelism()
            .map(|nzu| nzu.get() as i32)
            .unwrap_or(DEFAULT_N_THREADS); // Fallback if parallelism can't be determined

        Self {
            record_to_wav: DEFAULT_RECORD_TO_WAV, // This is None, which is const
            language: default_lang, // Initialized above
            step_ms: DEFAULT_STEP_MS,
            length_ms: DEFAULT_LENGTH_MS,
            keep_ms: DEFAULT_KEEP_MS,
            max_tokens: DEFAULT_MAX_TOKENS,
            n_threads: default_n_threads,
            audio_device_name: DEFAULT_AUDIO_DEVICE_NAME, // This is None, which is const
        }
    }
}

/// Configuration for the transcription stream.
#[derive(Debug, Clone)]
pub enum TranscriptionStreamEvent {
    Transcript {
        text: String,
        is_final: bool,
    },
    SystemMessage(String),
    Error(String),
}

fn send_err<T: Into<String>>(tx: &mpsc::Sender<TranscriptionStreamEvent>, msg: T) {
    let _ = tx.send(TranscriptionStreamEvent::Error(msg.into()));
}

/// Starts the transcription stream and returns a receiver for stream events.
///
/// # Arguments
/// * `params`: Configuration parameters for the audio capture and transcription process.
pub fn start_transcription_stream(params: TranscriptionStreamParams) -> Receiver<TranscriptionStreamEvent> {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let config = params;

        // 1. Ensure model is present
        let model_path = match ensure_model() {
            Ok(p) => p,
            Err(e) => {
                send_err(&tx, format!("Model error: {e}"));
                return;
            }
        };
        // 2. Initialize whisper context
        let ctx = match WhisperContext::new_with_params(
            model_path.to_str().unwrap(),
            WhisperContextParameters::default(),
        ) {
            Ok(c) => c,
            Err(e) => {
                send_err(&tx, format!("Failed to load model: {e}"));
                return;
            }
        };
        // 3. Start audio capture
        let audio_input = match AudioInput::new(config.audio_device_name.as_deref(), config.step_ms) {
            Ok(input) => input,
            Err(e) => {
                send_err(&tx, format!("Failed to initialize audio input: {e}"));
                return;
            }
        };
        let audio_rx = audio_input.start_capture_16k();
        let sample_rate = 16000;
        let n_samples_window = (sample_rate as f32 * (config.length_ms as f32 / 1000.0)) as usize;
        let n_samples_overlap = (sample_rate as f32 * (config.keep_ms as f32 / 1000.0)) as usize;
        let mut segment_window: Vec<f32> = Vec::with_capacity(n_samples_window);
        let mut state = match ctx.create_state() {
            Ok(s) => s,
            Err(e) => {
                send_err(&tx, format!("Failed to create state: {e}"));
                return;
            }
        };
        // WAV writer setup (record original device audio, not resampled)
        let mut wav_writer = if let Some(path) = config.record_to_wav.clone() {
            println!("[Recording] Saving transcribed audio to {path}...");
            let spec = hound::WavSpec {
                channels: audio_input.channels,
                sample_rate: audio_input.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            Some(hound::WavWriter::create(path, spec).expect("Failed to create WAV file"))
        } else {
            None
        };
        for pcmf32_new in audio_rx {
            // Write to WAV if enabled (convert f32 to i16 for WAV)
            if let Some(writer) = wav_writer.as_mut() {
                for &sample in &pcmf32_new {
                    let s = (sample * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                    writer.write_sample(s).expect("Failed to write sample");
                }
            }
            // Rolling segment window: append new samples
            segment_window.extend_from_slice(&pcmf32_new);
            // Emit partial transcript for the current segment window
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_n_threads(config.n_threads);
            params.set_max_tokens(config.max_tokens);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            if let Some(ref lang) = config.language {
                params.set_language(Some(lang));
            }
            let res = state.full(params, &segment_window);
            let mut text = String::new();
            match res {
                Ok(_) => {
                    let num_segments = match state.full_n_segments() {
                        Ok(n) => n,
                        Err(e) => {
                            send_err(&tx, format!("Segment error: {e}"));
                            continue;
                        }
                    };
                    for i in 0..num_segments {
                        match state.full_get_segment_text(i) {
                            Ok(seg) => text.push_str(&seg),
                            Err(e) => {
                                send_err(&tx, format!("Segment text error: {e}"));
                                continue;
                            }
                        }
                    }
                }
                Err(e) => {
                    send_err(&tx, format!("Transcription error: {e}"));
                    continue;
                }
            }
            // Emit the current segment window transcript as partial
            if !text.trim().is_empty() {
                let _ = tx.send(TranscriptionStreamEvent::Transcript { text: text.clone(), is_final: false });
            }
            // If the segment window reaches max length, emit final and start new segment window with overlap
            if segment_window.len() >= n_samples_window {
                // Emit final transcript for this segment
                let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                params.set_n_threads(config.n_threads);
                params.set_max_tokens(config.max_tokens);
                params.set_print_special(false);
                params.set_print_progress(false);
                params.set_print_realtime(false);
                params.set_print_timestamps(false);
                if let Some(ref lang) = config.language {
                    params.set_language(Some(lang));
                }
                let res = state.full(params, &segment_window);
                let mut text = String::new();
                if let Ok(_) = res {
                    if let Ok(num_segments) = state.full_n_segments() {
                        for i in 0..num_segments {
                            if let Ok(seg) = state.full_get_segment_text(i) {
                                text.push_str(&seg);
                            }
                        }
                    }
                }
                if !text.trim().is_empty() {
                    let _ = tx.send(TranscriptionStreamEvent::Transcript { text, is_final: true });
                }
                // Start new segment window with overlap
                if n_samples_overlap > 0 && segment_window.len() > n_samples_overlap {
                    segment_window = segment_window[segment_window.len() - n_samples_overlap..].to_vec();
                } else {
                    segment_window.clear();
                }
            }
        }
        // After the loop, emit final for any remaining segment window
        if !segment_window.is_empty() {
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_n_threads(config.n_threads);
            params.set_max_tokens(config.max_tokens);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            if let Some(ref lang) = config.language {
                params.set_language(Some(lang));
            }
            let res = state.full(params, &segment_window);
            let mut text = String::new();
            if let Ok(_) = res {
                if let Ok(num_segments) = state.full_n_segments() {
                    for i in 0..num_segments {
                        if let Ok(seg) = state.full_get_segment_text(i) {
                            text.push_str(&seg);
                        }
                    }
                }
            }
            if !text.trim().is_empty() {
                let _ = tx.send(TranscriptionStreamEvent::Transcript { text, is_final: true });
            }
        }
        // Finalize WAV writer if used
        if let Some(writer) = wav_writer {
            writer.finalize().expect("Failed to finalize WAV file");
            println!("[Done] Saved transcribed audio");
        }
    });
    rx
}