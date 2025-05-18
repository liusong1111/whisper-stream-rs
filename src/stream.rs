use std::sync::mpsc::{self, Receiver};
use std::thread;
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

use crate::audio::start_audio_capture;
use crate::model::ensure_model;

const N_THREADS: i32 = 4;
const STEP_MS: u32 = 3000;
const LENGTH_MS: u32 = 10000;
const KEEP_MS: u32 = 200;
const MAX_TOKENS: i32 = 32;

#[derive(Debug, Clone)]
pub enum TranscriptionStreamEvent {
    Transcript {
        text: String,
        is_final: bool,
    },
    SystemMessage(String),
    Error(String),
}

/// Starts the transcription stream and returns a receiver for stream events.
pub fn start_transcription_stream() -> Receiver<TranscriptionStreamEvent> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        // 1. Ensure model is present
        let model_path = match ensure_model() {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.send(TranscriptionStreamEvent::Error(format!("Model error: {e}")));
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
                let _ = tx.send(TranscriptionStreamEvent::Error(format!("Failed to load model: {e}")));
                return;
            }
        };
        // 3. Start audio capture
        let audio_rx = start_audio_capture(STEP_MS);
        let sample_rate = 16000;
        let n_samples_step = (sample_rate as f32 * (STEP_MS as f32 / 1000.0)) as usize;
        let n_samples_len = (sample_rate as f32 * (LENGTH_MS as f32 / 1000.0)) as usize;
        let n_samples_keep = (sample_rate as f32 * (KEEP_MS as f32 / 1000.0)) as usize;
        let mut pcmf32_old: Vec<f32> = Vec::with_capacity(n_samples_len);
        let mut state = match ctx.create_state() {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(TranscriptionStreamEvent::Error(format!("Failed to create state: {e}")));
                return;
            }
        };
        let mut last_text = None;
        for pcmf32_new in audio_rx {
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
            params.set_n_threads(N_THREADS);
            params.set_max_tokens(MAX_TOKENS);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            params.set_language(Some("en"));
            let res = state.full(params, &pcmf32);
            let mut text = String::new();
            match res {
                Ok(_) => {
                    let num_segments = match state.full_n_segments() {
                        Ok(n) => n,
                        Err(e) => {
                            let _ = tx.send(TranscriptionStreamEvent::Error(format!("Segment error: {e}")));
                            continue;
                        }
                    };
                    for i in 0..num_segments {
                        match state.full_get_segment_text(i) {
                            Ok(seg) => text.push_str(&seg),
                            Err(e) => {
                                let _ = tx.send(TranscriptionStreamEvent::Error(format!("Segment text error: {e}")));
                                continue;
                            }
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(TranscriptionStreamEvent::Error(format!("Transcription error: {e}")));
                    continue;
                }
            }
            // Send the previous chunk as final, if any
            if let Some(prev) = last_text.take() {
                let _ = tx.send(TranscriptionStreamEvent::Transcript { text: prev, is_final: true });
            }
            // Send the current chunk as intermediate
            let _ = tx.send(TranscriptionStreamEvent::Transcript { text: text.clone(), is_final: false });
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
    });
    rx
}