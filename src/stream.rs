use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy, WhisperError};
use hound;

use crate::audio::{AudioInput};
use crate::model::ensure_model;
use crate::error::WhisperStreamError;

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
#[derive(Debug)]
pub enum TranscriptionStreamEvent {
    Transcript {
        text: String,
        is_final: bool,
    },
    SystemMessage(String),
    Error(WhisperStreamError),
}

fn send_custom_error(tx: &Sender<TranscriptionStreamEvent>, error: WhisperStreamError) {
    let _ = tx.send(TranscriptionStreamEvent::Error(error));
}

/// Starts the transcription stream and returns a receiver for stream events.
///
/// # Arguments
/// * `params`: Configuration parameters for the audio capture and transcription process.
pub fn start_transcription_stream(params: TranscriptionStreamParams) -> Receiver<TranscriptionStreamEvent> {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let config = params;

        let model_path = match ensure_model() {
            Ok(p) => p,
            Err(e) => {
                send_custom_error(&tx, e);
                return;
            }
        };

        let ctx = match WhisperContext::new_with_params(
            model_path.to_str().unwrap_or("invalid_model_path"),
            WhisperContextParameters::default(),
        ) {
            Ok(c) => c,
            Err(e) => {
                send_custom_error(&tx, WhisperStreamError::from(e));
                return;
            }
        };

        let audio_input = match AudioInput::new(config.audio_device_name.as_deref(), config.step_ms) {
            Ok(input) => input,
            Err(e) => {
                send_custom_error(&tx, e);
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
                send_custom_error(&tx, WhisperStreamError::from(e));
                return;
            }
        };
        let mut wav_writer = if let Some(path_str) = config.record_to_wav.clone() {
            println!("[Recording] Saving transcribed audio to {path_str}...");
            let spec = hound::WavSpec {
                channels: audio_input.channels,
                sample_rate: audio_input.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            match hound::WavWriter::create(&path_str, spec) {
                Ok(writer) => {
                    if tx.send(TranscriptionStreamEvent::SystemMessage(format!("[Recording] Saving transcribed audio to {}...", path_str))).is_err() {
                        return;
                    }
                    Some(writer)
                }
                Err(e) => {
                    send_custom_error(&tx, WhisperStreamError::Hound { source: e });
                    None
                }
            }
        } else {
            None
        };
        for pcmf32_new_result in audio_rx {
            let pcmf32_new = match pcmf32_new_result {
                Ok(audio_data) => {
                    if audio_data.is_empty() {
                        continue;
                    }
                    audio_data
                }
                Err(audio_err) => {
                    send_custom_error(&tx, audio_err);
                    continue;
                }
            };
            segment_window.extend_from_slice(&pcmf32_new);
            let mut params_full = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params_full.set_n_threads(config.n_threads);
            params_full.set_max_tokens(config.max_tokens);
            params_full.set_print_special(false);
            params_full.set_print_progress(false);
            params_full.set_print_realtime(false);
            params_full.set_print_timestamps(false);
            if let Some(ref lang) = config.language {
                params_full.set_language(Some(lang));
            }
            if let Err(e) = state.full(params_full.clone(), &segment_window) {
                send_custom_error(&tx, WhisperStreamError::from(e));
                continue;
            }
            let mut current_text = String::new();
            match state.full_n_segments() {
                Ok(num_segments) => {
                    for i in 0..num_segments {
                        match state.full_get_segment_text(i) {
                            Ok(seg) => current_text.push_str(&seg),
                            Err(e) => {
                                send_custom_error(&tx, WhisperStreamError::from(e));
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    send_custom_error(&tx, WhisperStreamError::from(e));
                    continue;
                }
            }
            if !current_text.trim().is_empty() {
                let _ = tx.send(TranscriptionStreamEvent::Transcript { text: current_text.clone(), is_final: false });
            }
            if segment_window.len() >= n_samples_window {
                if !current_text.trim().is_empty() {
                    let _ = tx.send(TranscriptionStreamEvent::Transcript { text: current_text, is_final: true });
                }
                if n_samples_overlap > 0 && segment_window.len() > n_samples_overlap {
                    segment_window = segment_window[segment_window.len() - n_samples_overlap..].to_vec();
                } else {
                    segment_window.clear();
                }
            }
        }
        if !segment_window.is_empty() {
            let mut params_final = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params_final.set_n_threads(config.n_threads);
            params_final.set_max_tokens(config.max_tokens);
            if let Some(ref lang) = config.language {
                params_final.set_language(Some(lang));
            }
            if let Err(e) = state.full(params_final, &segment_window) {
                send_custom_error(&tx, WhisperStreamError::from(e));
            } else {
                let mut final_text = String::new();
                match state.full_n_segments() {
                    Ok(num_segments) => {
                        for i in 0..num_segments {
                            match state.full_get_segment_text(i) {
                                Ok(seg) => final_text.push_str(&seg),
                                Err(e) => {
                                    send_custom_error(&tx, WhisperStreamError::from(e));
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        send_custom_error(&tx, WhisperStreamError::from(e));
                    }
                }
                if !final_text.trim().is_empty() {
                    let _ = tx.send(TranscriptionStreamEvent::Transcript { text: final_text, is_final: true });
                }
            }
        }
        if let Some(mut writer) = wav_writer {
            if let Err(e) = writer.finalize() {
                send_custom_error(&tx, WhisperStreamError::from(e));
            }
        }
    });
    rx
}