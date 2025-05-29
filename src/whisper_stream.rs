use std::sync::mpsc::{self, Receiver};
use std::thread;
use crate::model::Model;

/// Events emitted by the transcription stream.
///
/// These are sent through the channel returned by [`WhisperStreamBuilder::build`].
#[derive(Debug)]
pub enum Event {
    /// A provisional, live text update. This is an intermediate result, suitable for displaying
    /// real-time feedback. It is subject to change and will be superseded by subsequent
    /// `ProvisionalLiveUpdate` messages (providing more refined guesses for the same ongoing audio)
    /// or ultimately by a `SegmentTranscript` for that audio segment.
    /// These should not be stored or considered definitive.
    ///
    /// `is_low_quality` is true if the text is considered low quality by the detector.
    ProvisionalLiveUpdate { text: String, is_low_quality: bool },

    /// The final and complete transcription for a specific audio segment window.
    /// This is the version of the transcript that should be considered the actual output for that portion of audio.
    ///
    /// `is_low_quality` is true if the text is considered low quality by the detector.
    SegmentTranscript { text: String, is_low_quality: bool },

    /// System messages (e.g., recording status, warnings).
    SystemMessage(String),
    /// Errors encountered during processing.
    Error(crate::error::WhisperStreamError),
}

/// Main entry point for configuring and running a Whisper transcription stream.
///
/// Use [`WhisperStream::builder()`] to create a [`WhisperStreamBuilder`], configure options,
/// and call `.build()` to start streaming and receive events.
///
/// Example:
/// ```no_run
/// use whisper_stream_rs::{WhisperStream, Event, Model};
/// let (_stream, rx) = WhisperStream::builder().model(Model::TinyEn).build().unwrap();
/// for event in rx {
///     match event {
///         Event::SegmentTranscript { text, .. } => println!("Final: {}", text),
///         _ => {}
///     }
/// }
/// ```
pub struct WhisperStream {
    // Will own the background thread and config
}

/// Builder for [`WhisperStream`].
///
/// All configuration is set via builder methods. Call `.build()` to start streaming and receive events.
///
/// Example:
/// ```no_run
/// use whisper_stream_rs::{WhisperStream, Model};
/// let (_stream, rx) = WhisperStream::builder()
///     .device("MacBook Pro Microphone")
///     .language("en")
///     .model(Model::SmallEn)
///     .build()
///     .unwrap();
/// ```
pub struct WhisperStreamBuilder {
    device: Option<String>,
    language: Option<String>,
    record_to_wav: Option<String>,
    step_ms: u32,
    length_ms: u32,
    keep_ms: u32,
    max_tokens: i32,
    n_threads: i32,
    compute_partials: bool,
    logging_enabled: bool,
    model: Option<Model>,
}

impl WhisperStreamBuilder {
    pub fn device(mut self, name: &str) -> Self {
        self.device = Some(name.to_string());
        self
    }
    pub fn language(mut self, lang: &str) -> Self {
        self.language = Some(lang.to_string());
        self
    }
    pub fn record_to_wav(mut self, path: &str) -> Self {
        self.record_to_wav = Some(path.to_string());
        self
    }
    pub fn step_ms(mut self, ms: u32) -> Self {
        self.step_ms = ms;
        self
    }
    pub fn length_ms(mut self, ms: u32) -> Self {
        self.length_ms = ms;
        self
    }
    pub fn keep_ms(mut self, ms: u32) -> Self {
        self.keep_ms = ms;
        self
    }
    pub fn max_tokens(mut self, n: i32) -> Self {
        self.max_tokens = n;
        self
    }
    pub fn n_threads(mut self, n: i32) -> Self {
        self.n_threads = n;
        self
    }
    pub fn compute_partials(mut self, enabled: bool) -> Self {
        self.compute_partials = enabled;
        self
    }
    pub fn disable_logging(mut self) -> Self {
        self.logging_enabled = false;
        self
    }
    pub fn model(mut self, model: Model) -> Self {
        self.model = Some(model);
        self
    }
    pub fn build(self) -> Result<(WhisperStream, Receiver<Event>), crate::error::WhisperStreamError> {
        // Set up logging if enabled
        if self.logging_enabled {
            // Safe to call multiple times; only installs once
            whisper_rs::install_logging_hooks();
        }

        let (tx, rx) = mpsc::channel();
        let config = self;
        let selected_model = config.model.unwrap_or(Model::BaseEn);
        thread::spawn(move || {
            use crate::model::ensure_model;
            use crate::audio::{AudioInput};
            use crate::audio_utils::{pad_audio_if_needed, WavAudioRecorder};
            use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};
            use log::info;
            use std::sync::Arc;

            const MIN_WHISPER_SAMPLES: usize = 16800; // 1050ms at 16kHz (increased buffer)

            let model_path = match ensure_model(selected_model) {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.send(Event::Error(e));
                    return;
                }
            };

            let system_info = whisper_rs::print_system_info();
            info!("Whisper System Info: \n{}", system_info);

            let ctx = match WhisperContext::new_with_params(
                model_path.to_str().unwrap_or("invalid_model_path"),
                WhisperContextParameters::default(),
            ) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Event::Error(crate::error::WhisperStreamError::from(e)));
                    return;
                }
            };

            let audio_input = match AudioInput::new(config.device.as_deref(), config.step_ms) {
                Ok(input) => input,
                Err(e) => {
                    let _ = tx.send(Event::Error(e));
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
                    let _ = tx.send(Event::Error(crate::error::WhisperStreamError::from(e)));
                    return;
                }
            };

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
            let arc_params_full = Arc::new(params_full);

            let mut wav_audio_recorder = match WavAudioRecorder::new(config.record_to_wav.as_deref()) {
                Ok(recorder) => recorder,
                Err(e) => {
                    let _ = tx.send(Event::Error(e));
                    match WavAudioRecorder::new(None) {
                        Ok(no_op_recorder) => no_op_recorder,
                        Err(_) => return,
                    }
                }
            };

            if wav_audio_recorder.is_recording() {
                if let Some(path_str) = config.record_to_wav.as_ref() {
                    info!("[Recording] Saving transcribed audio to {}...", path_str);
                    let _ = tx.send(Event::SystemMessage(format!("[Recording] Saving transcribed audio to {}...", path_str)));
                }
            }

            for pcmf32_new_result in audio_rx {
                let pcmf32_new = match pcmf32_new_result {
                    Ok(audio_data) => {
                        if audio_data.is_empty() {
                            continue;
                        }
                        audio_data
                    }
                    Err(audio_err) => {
                        let _ = tx.send(Event::Error(audio_err));
                        continue;
                    }
                };

                if wav_audio_recorder.is_recording() {
                    if let Err(e) = wav_audio_recorder.write_audio_chunk(&pcmf32_new) {
                        let _ = tx.send(Event::Error(e));
                    }
                }

                segment_window.extend_from_slice(&pcmf32_new);
                let audio_for_processing = pad_audio_if_needed(&segment_window, MIN_WHISPER_SAMPLES);

                if let Err(e) = state.full(arc_params_full.as_ref().clone(), &audio_for_processing) {
                    let _ = tx.send(Event::Error(crate::error::WhisperStreamError::from(e)));
                    continue;
                }

                let mut current_text = String::new();
                match state.full_n_segments() {
                    Ok(num_segments) => {
                        for i in 0..num_segments {
                            match state.full_get_segment_text(i) {
                                Ok(seg) => current_text.push_str(&seg),
                                Err(e) => {
                                    let _ = tx.send(Event::Error(crate::error::WhisperStreamError::from(e)));
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Event::Error(crate::error::WhisperStreamError::from(e)));
                        continue;
                    }
                }

                if !current_text.trim().is_empty() {
                    let is_low_quality = crate::score::is_low_quality_output(&current_text);
                    if segment_window.len() >= n_samples_window {
                        let _ = tx.send(Event::SegmentTranscript { text: current_text.clone(), is_low_quality });
                    } else if config.compute_partials {
                        let _ = tx.send(Event::ProvisionalLiveUpdate { text: current_text.clone(), is_low_quality });
                    }
                }

                if segment_window.len() >= n_samples_window {
                    if n_samples_overlap > 0 && segment_window.len() > n_samples_overlap {
                        segment_window = segment_window[segment_window.len() - n_samples_overlap..].to_vec();
                    } else {
                        segment_window.clear();
                    }
                }
            }

            if !segment_window.is_empty() {
                let final_audio_for_processing = pad_audio_if_needed(&segment_window, MIN_WHISPER_SAMPLES);
                if let Err(e) = state.full(arc_params_full.as_ref().clone(), &final_audio_for_processing) {
                    let _ = tx.send(Event::Error(crate::error::WhisperStreamError::from(e)));
                } else {
                    let mut final_text = String::new();
                    match state.full_n_segments() {
                        Ok(num_segments) => {
                            for i in 0..num_segments {
                                match state.full_get_segment_text(i) {
                                    Ok(seg) => final_text.push_str(&seg),
                                    Err(e) => {
                                        let _ = tx.send(Event::Error(crate::error::WhisperStreamError::from(e)));
                                        break;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Event::Error(crate::error::WhisperStreamError::from(e)));
                        }
                    }
                    if !final_text.trim().is_empty() {
                        let is_low_quality = crate::score::is_low_quality_output(&final_text);
                        let _ = tx.send(Event::SegmentTranscript { text: final_text, is_low_quality });
                    }
                }
            }

            match wav_audio_recorder.finalize() {
                Ok(Some(msg)) => {
                    info!("{}", msg);
                    let _ = tx.send(Event::SystemMessage(msg));
                }
                Ok(None) => { /* No recording was active, nothing to report */ }
                Err(e) => {
                    let _ = tx.send(Event::Error(e));
                }
            }
        });
        Ok((WhisperStream {}, rx))
    }
}

impl WhisperStream {
    pub fn builder() -> WhisperStreamBuilder {
        WhisperStreamBuilder {
            device: None,
            language: Some("en".to_string()),
            record_to_wav: None,
            step_ms: 800,
            length_ms: 5000,
            keep_ms: 200,
            max_tokens: 32,
            n_threads: std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4),
            compute_partials: true,
            logging_enabled: true,
            model: None,
        }
    }
    pub fn list_devices() -> Result<Vec<String>, crate::error::WhisperStreamError> {
        crate::audio::AudioInput::available_input_devices()
    }
    pub fn list_models() -> Vec<Model> {
        Model::list()
    }
    pub fn start(&mut self) -> Result<(), crate::error::WhisperStreamError> {
        // Will start the background thread in next phase
        Ok(())
    }
    pub fn stop(&mut self) -> Result<(), crate::error::WhisperStreamError> {
        // Will stop the background thread in next phase
        Ok(())
    }
}