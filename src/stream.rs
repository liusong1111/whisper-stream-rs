use std::sync::mpsc::{self, Receiver};
use std::thread;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
use hound;

use crate::audio::start_audio_capture;
use crate::model::ensure_model;

/// Configuration for the transcription stream.
#[derive(Debug, Clone)]
pub struct TranscriptionConfig {
    pub record_to_wav: Option<String>,
    pub language: Option<String>,
    pub step_ms: u32,
    pub length_ms: u32,
    pub keep_ms: u32,
    pub max_tokens: i32,
    pub n_threads: i32,
}

impl Default for TranscriptionConfig {
    fn default() -> Self {
        Self {
            record_to_wav: None,
            language: Some("en".to_string()),
            step_ms: 3000,
            length_ms: 10000,
            keep_ms: 200,
            max_tokens: 32,
            n_threads: 4,
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
pub fn start_transcription_stream(config: TranscriptionConfig) -> Receiver<TranscriptionStreamEvent> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
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
        let audio_rx = start_audio_capture(config.step_ms);
        let sample_rate = 16000;
        let n_samples_len = (sample_rate as f32 * (config.length_ms as f32 / 1000.0)) as usize;
        let n_samples_keep = (sample_rate as f32 * (config.keep_ms as f32 / 1000.0)) as usize;
        let mut pcmf32_old: Vec<f32> = Vec::with_capacity(n_samples_len);
        let mut state = match ctx.create_state() {
            Ok(s) => s,
            Err(e) => {
                send_err(&tx, format!("Failed to create state: {e}"));
                return;
            }
        };
        let mut last_text = None;
        // WAV writer setup
        let mut wav_writer = if let Some(path) = config.record_to_wav.clone() {
            println!("[Recording] Saving transcribed audio to {path}...");
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            Some(hound::WavWriter::create(path, spec).expect("Failed to create WAV file"))
        } else {
            None
        };
        for pcmf32_new in audio_rx {
            // Write to WAV if enabled
            if let Some(writer) = wav_writer.as_mut() {
                for &sample in &pcmf32_new {
                    let s = (sample * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                    writer.write_sample(s).expect("Failed to write sample");
                }
            }
            // Take up to n_samples_len audio from previous iteration
            let n_samples_take = std::cmp::min(pcmf32_old.len(), n_samples_keep + n_samples_len - pcmf32_new.len());
            let mut pcmf32 = Vec::with_capacity(pcmf32_new.len() + n_samples_take);
            if n_samples_take > 0 {
                pcmf32.extend_from_slice(&pcmf32_old[pcmf32_old.len() - n_samples_take..]);
            }
            pcmf32.extend_from_slice(&pcmf32_new);
            // Truncate to n_samples_len if needed
            if pcmf32.len() > n_samples_len {
                pcmf32 = pcmf32[pcmf32.len() - n_samples_len..].to_vec();
            }
            // 4. Transcribe
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
            let res = state.full(params, &pcmf32);
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
            // Send the previous chunk as final, if any
            if let Some(prev) = last_text.take() {
                let _ = tx.send(TranscriptionStreamEvent::Transcript { text: prev, is_final: true });
            }
            // Send the current chunk as intermediate
            if !text.trim().is_empty() {
                let _ = tx.send(TranscriptionStreamEvent::Transcript { text: text.clone(), is_final: false });
            }
            last_text = Some(text);
            // Keep part of the audio for next iteration
            if pcmf32.len() > n_samples_keep {
                pcmf32_old = pcmf32[pcmf32.len() - n_samples_keep..].to_vec();
            } else {
                pcmf32_old = pcmf32;
            }
        }
        // After the loop, send the last chunk as final if any
        if let Some(prev) = last_text.take() {
            let _ = tx.send(TranscriptionStreamEvent::Transcript { text: prev, is_final: true });
        }
        // Finalize WAV writer if used
        if let Some(writer) = wav_writer {
            writer.finalize().expect("Failed to finalize WAV file");
            println!("[Done] Saved transcribed audio");
        }
    });
    rx
}