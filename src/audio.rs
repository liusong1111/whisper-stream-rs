//! Handles audio capture, processing, and streaming for real-time transcription.
//!
//! This module interfaces with system audio input, captures raw data,
//! and processes it into a standard format for speech recognition:
//! 1. Captures audio in `step_ms` chunks.
//! 2. Downmixes to mono.
//! 3. Resamples to 16kHz for Whisper ASR.
//!
//! Processed audio chunks (`Vec<f32>`) are streamed via a channel Receiver
//! for consumption by other parts of the application (e.g., `stream.rs`)
//! for continuous transcription.

use std::sync::mpsc::{self, Receiver, Sender};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, StreamConfig, InputCallbackInfo, StreamError as CpalStreamError};
use rubato::{FftFixedInOut, Resampler};
use crate::error::WhisperStreamError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use log::{info, warn, error, debug};

/// Encapsulates audio input device information and configuration.
pub struct AudioInput {
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
    step_duration_ms: u32,
}

impl AudioInput {
    /// Lists names of available audio input devices.
    pub fn available_input_devices() -> Result<Vec<String>, WhisperStreamError> {
        let host = cpal::default_host();
        let devices = host.input_devices().map_err(WhisperStreamError::from)?;
        let mut device_names = Vec::new();
        for (index, device) in devices.enumerate() {
            let name = match device.name() {
                Ok(n) => n,
                Err(e) => {
                    warn!("[Audio] Warning: Could not get name for input device #{}: {}", index, e);
                    format!("<Unknown Device #{}>", index)
                }
            };
            device_names.push(name);
        }
        Ok(device_names)
    }

    /// Creates a new `AudioInput`.
    ///
    /// Uses default input if `device_name_opt` is `None`.
    /// `step_ms` defines the audio chunk duration.
    /// Fetches device config and warns if not mono, as mono is the target format.
    pub fn new(device_name_opt: Option<&str>, step_ms: u32) -> Result<Self, WhisperStreamError> {
        let host = cpal::default_host();

        // Always list devices first to potentially trigger permission prompt
        info!("[Audio] Available input devices:");
        let mut found_devices = Vec::new();

        // Get the default device name first for comparison
        let default_name = host.default_input_device()
            .and_then(|d| d.name().ok())
            .unwrap_or_else(|| "<unknown default>".to_string());
        info!("[Audio] Default input device is: {}", default_name);

        for device in host.input_devices().map_err(WhisperStreamError::from)? {
            if let Ok(name) = device.name() {
                info!("[Audio] Found device: {} {}",
                    name,
                    if name == default_name { "(default)" } else { "" }
                );
                found_devices.push((name, device));
            }
        }

        if found_devices.is_empty() {
            return Err(WhisperStreamError::AudioDevice("No input devices available".to_string()));
        }

        let (device_name, device) = match device_name_opt {
            Some(name) => {
                // Try to find the exact device
                found_devices.into_iter()
                    .find(|(dev_name, _)| dev_name.eq_ignore_ascii_case(name))
                    .ok_or_else(|| WhisperStreamError::AudioDevice(format!("Input device named '{}' not found (case-insensitive search)", name)))?
            }
            None => {
                // Try to find a non-default device first (prefer external microphones)
                info!("[Audio] No device specified, looking for best available device...");

                // Find index of preferred device
                let preferred_idx = found_devices.iter()
                    .position(|(name, _)| {
                        name != &default_name && (
                            name.contains("MacBook Pro-Mikrofon")
                        )
                    });

                // Remove and return either the preferred device or the first one
                if let Some(idx) = preferred_idx {
                    let (name, device) = found_devices.remove(idx);
                    info!("[Audio] Selected preferred device: {}", name);
                    (name, device)
                } else {
                    let (name, device) = found_devices.remove(0);
                    info!("[Audio] No preferred device found, using: {}", name);
                    (name, device)
                }
            }
        };

        let config = device
            .default_input_config()
            .map_err(WhisperStreamError::from)?;

        // Try to get supported configs for this device specifically
        info!("[Audio] Checking supported configurations for device: {}", device_name);
        if let Ok(supported_configs) = device.supported_input_configs() {
            for supported_config in supported_configs {
                info!("[Audio] - Supported: Rate {:?}, Channels {}, Format {:?}",
                    supported_config.min_sample_rate()..=supported_config.max_sample_rate(),
                    supported_config.channels(),
                    supported_config.sample_format()
                );
            }
        }

        let sample_rate = config.sample_rate().0;
        let channels = config.channels();

        info!("[Audio] Attempting to use device: {}", device_name);
        info!("[Audio] Selected config: SampleRate({}), Channels({}), Format({})",
            sample_rate, channels, config.sample_format());

        // Try to verify device access
        info!("[Audio] Verifying device access...");
        if let Err(e) = device.default_input_config() {
            error!("[Audio] Failed to access device configuration: {}", e);
            return Err(WhisperStreamError::AudioDevice(
                format!("Failed to access device '{}'. Please check microphone permissions: {}", device_name, e)
            ));
        }
        info!("[Audio] Successfully verified device access");

        if channels != 1 {
            warn!("[Audio][Warning] Source input device '{}' has {} channels. It will be downmixed to mono.", device_name, channels);
        }

        Ok(Self {
            device_name,
            sample_rate,
            channels,
            step_duration_ms: step_ms,
        })
    }

    /// Internal helper for cpal input stream and audio processing.
    ///
    /// Sets up cpal stream, converts samples to `f32`, buffers,
    /// downmixes to mono, resamples to 16kHz (if needed),
    /// and sends `Vec<f32>` chunks via `tx`.
    ///
    /// # Type Parameters
    /// * `T`: cpal sample type (`f32`, `i16`, `u16`).
    /// * `F`: Closure to convert `T` to `f32`.
    fn process_audio_stream_internal<T, F>(
        device: &cpal::Device,
        config: &StreamConfig,
        tx: Sender<Result<Vec<f32>, WhisperStreamError>>,
        audio_channels: usize,
        device_samples_per_step: usize, // Native samples for one processing step.
        mut resampler_opt: Option<FftFixedInOut<f32>>,
        err_fn_outer_tx: Sender<Result<Vec<f32>, WhisperStreamError>>,
        convert_sample: F,
        stop_processing_signal: Arc<AtomicBool>,
    ) -> Result<cpal::Stream, WhisperStreamError>
    where
        T: cpal::SizedSample,
        F: Fn(&T) -> f32 + Send + Sync + 'static,
    {
        let mut main_buffer: Vec<f32> = Vec::with_capacity(device_samples_per_step * audio_channels * 2);
        let callback_stop_signal = stop_processing_signal.clone();

        // Pre-allocate buffers to reduce allocations in the hot loop
        let mut interleaved_chunk_buffer: Vec<f32> = Vec::with_capacity(device_samples_per_step * audio_channels);
        let mut mono_chunk_buffer: Vec<f32> = Vec::with_capacity(device_samples_per_step);

        device.build_input_stream(
            config,
            move |data: &[T], _: &InputCallbackInfo| {
                if callback_stop_signal.load(Ordering::Relaxed) {
                    return;
                }

                if log::log_enabled!(log::Level::Debug) {
                    let mut min = f32::INFINITY;
                    let mut max = f32::NEG_INFINITY;
                    for sample in data {
                        let s = convert_sample(sample);
                        min = min.min(s);
                        max = max.max(s);
                        main_buffer.push(s);
                    }
                    debug!(
                        "[Audio] Input chunk: len={}, range=[{:.3}, {:.3}]",
                        data.len(),
                        min,
                        max
                    );
                } else {
                    main_buffer.extend(data.iter().map(&convert_sample));
                }

                while main_buffer.len() >= device_samples_per_step * audio_channels {
                    if callback_stop_signal.load(Ordering::Relaxed) {
                        main_buffer.clear();
                        return;
                    }

                    interleaved_chunk_buffer.clear();
                    interleaved_chunk_buffer.extend(main_buffer.drain(..device_samples_per_step * audio_channels));

                    mono_chunk_buffer.clear();
                    let downmix_result: Result<(), WhisperStreamError> = if audio_channels == 1 {
                        mono_chunk_buffer.extend_from_slice(&interleaved_chunk_buffer);
                        Ok(())
                    } else if audio_channels == 2 {
                        match whisper_rs::convert_stereo_to_mono_audio(&interleaved_chunk_buffer) {
                            Ok(converted_mono) => {
                                mono_chunk_buffer.extend_from_slice(&converted_mono);
                                Ok(())
                            }
                            Err(e) => {
                                Err(WhisperStreamError::AudioStreamRuntime(format!("Stereo to mono conversion failed: {:?}", e)))
                            }
                        }
                    } else { // audio_channels > 2, manual downmix
                        // Ensure enough samples for full frames before processing.
                        // This check might be overly defensive if input length calculation is always correct.
                        if interleaved_chunk_buffer.chunks_exact(audio_channels).next().is_none() && !interleaved_chunk_buffer.is_empty() {
                            warn!("[Audio] Incomplete frame for multi-channel downmix. Buffer len: {}, channels: {}. Skipping.", interleaved_chunk_buffer.len(), audio_channels);
                        }
                        for frame in interleaved_chunk_buffer.chunks_exact(audio_channels) {
                            mono_chunk_buffer.push(frame.iter().sum::<f32>() / audio_channels as f32);
                        }
                        Ok(())
                    };

                    if let Err(e) = downmix_result {
                        error!("[Audio] Downmixing audio failed: {}", e);
                        if tx.send(Err(e)).is_err() {
                            error!("[Audio] Receiver dropped while sending downmix error. Signalling to stop.");
                        }
                        callback_stop_signal.store(true, Ordering::Relaxed);
                        return;
                    }

                    let final_chunk_data_result = if let Some(resampler) = resampler_opt.as_mut() {
                        match resampler.process(&[&mono_chunk_buffer], None) { // Pass as slice of slices
                            Ok(mut output_frames) => {
                                if output_frames.is_empty() {
                                    // Can happen if input chunk is too small for resampler.
                                    debug!("[Audio] Resampler returned no frames. Input size: {}. Accumulating more data.", mono_chunk_buffer.len());
                                    continue;
                                }
                                Ok(output_frames.remove(0))
                            }
                            Err(e) => {
                                Err(WhisperStreamError::AudioResampling(format!("Resample failed: {:?}", e)))
                            }
                        }
                    } else {
                        Ok(mono_chunk_buffer.clone()) // Clone as mono_chunk_buffer is reused
                    };

                    match final_chunk_data_result {
                        Ok(final_chunk) => {
                            if !final_chunk.is_empty() {
                                if tx.send(Ok(final_chunk)).is_err() {
                                    error!("[Audio] Receiver dropped. Signalling to stop audio processing.");
                                    callback_stop_signal.store(true, Ordering::Relaxed);
                                    return;
                                }
                            } else {
                                debug!("[Audio] Final chunk is empty after processing/resampling. Skipping send.");
                            }
                        }
                        Err(e) => {
                             error!("[Audio] Processing chunk failed: {}", e);
                             if tx.send(Err(e)).is_err() {
                                error!("[Audio] Receiver dropped while sending processing error. Signalling to stop.");
                            }
                            callback_stop_signal.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                }
            },
            move |err: CpalStreamError| {
                error!("[Audio] Stream error: {}", err);
                let _ = err_fn_outer_tx.send(Err(WhisperStreamError::from(err)));
            },
            None,
        ).map_err(WhisperStreamError::from)
    }

    /// Starts audio capture, returning a `Receiver` for 16kHz mono f32 audio chunks.
    ///
    /// This is the primary method to get audio for Whisper ASR.
    /// `step_duration_ms` determines chunk size.
    /// In a spawned thread, it:
    /// 1. Initializes cpal input stream.
    /// 2. Sets up resampler if native sample rate != 16kHz.
    /// 3. Continuously collects, converts, downmixes, resamples, and sends audio chunks.
    pub fn start_capture_16k(&self) -> Receiver<Result<Vec<f32>, WhisperStreamError>> {
        let (tx, rx) = mpsc::channel();
        let device_name_clone = self.device_name.clone();
        let step_duration_ms_clone = self.step_duration_ms;

        #[cfg(target_os = "macos")]
        {
            // Double-check permissions before starting capture
            let host = cpal::default_host();
            if let Err(e) = host.input_devices() {
                error!("[Audio] No microphone access. Please check System Settings > Privacy & Security > Microphone: {}", e);
                let _ = tx.send(Err(WhisperStreamError::AudioDevice(
                    "No microphone access. Please check System Settings > Privacy & Security > Microphone".to_string()
                )));
                return rx;
            }
        }

        let stop_processing_signal = Arc::new(AtomicBool::new(false));
        // _thread_stop_signal kept for potential future use or if stream lifetime needs explicit management with it.
        let _thread_stop_signal = stop_processing_signal.clone();

        let tx_for_err_fn = tx.clone();
        let tx_for_data_cb = tx.clone();

        info!("[Audio] Starting capture for device '{}'. Native SR: {}, Target SR: {}, Channels: {}, Chunk ms: {}",
            self.device_name, self.sample_rate, 16000, self.channels, self.step_duration_ms);

        std::thread::spawn(move || {
            let host = cpal::default_host();
            let device_result = host.input_devices().map_err(WhisperStreamError::from)
                .and_then(|mut devices| {
                    devices.find(|d| d.name().is_ok_and(|n| n == device_name_clone))
                        .ok_or_else(|| WhisperStreamError::AudioDevice(format!("Named input device '{}' not found.", device_name_clone)))
                })
                .or_else(|_| { // If named device not found or error in listing, try default
                    host.default_input_device()
                        .ok_or_else(|| WhisperStreamError::AudioDevice(format!("Audio device '{}' not found and no default available.", device_name_clone)))
                });

            let device = match device_result {
                Ok(d) => d,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };

                        let default_config = match device.default_input_config().map_err(WhisperStreamError::from) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            let native_sample_rate = default_config.sample_rate().0;

            // Create a specific config instead of using default
            let config = StreamConfig {
                channels: 1, // Force mono
                sample_rate: cpal::SampleRate(native_sample_rate),
                buffer_size: cpal::BufferSize::Default,
            };
            debug!("[Audio] Using explicit stream config: channels={}, rate={}",
                config.channels, config.sample_rate.0);
            let audio_channels = config.channels as usize;
            let target_sample_rate = 16000;
            // device_samples_per_step is the number of *native* samples for one chunk before any processing.
            let device_samples_per_step = (native_sample_rate as f32 * (step_duration_ms_clone as f32 / 1000.0)) as usize;

                        debug!("[Audio] Device config: Format={:?}, Rate={}, Channels={}",
                default_config.sample_format(), native_sample_rate, audio_channels);

            // Try to get supported configs
            if let Ok(supported_configs) = device.supported_input_configs() {
                debug!("[Audio] Supported input configurations:");
                for conf in supported_configs {
                    debug!("  - Rate: {:?}, Channels: {}, Format: {:?}",
                        conf.min_sample_rate()..=conf.max_sample_rate(),
                        conf.channels(),
                        conf.sample_format());
                }
            }

            let resampler_opt = if native_sample_rate != target_sample_rate {
                match FftFixedInOut::<f32>::new(
                    native_sample_rate as usize,
                    target_sample_rate as usize,
                    device_samples_per_step, // Max input chunk size for resampler
                    1, // Output is always mono
                ) {
                    Ok(r) => Some(r),
                    Err(e) => {
                        let err_msg = format!("Failed to create audio resampler: {}", e);
                        error!("[Audio] Error: {}", err_msg);
                        let _ = tx.send(Err(WhisperStreamError::AudioResampling(err_msg)));
                        return;
                    }
                }
            } else {
                None
            };

            info!("[Audio] Starting capture. Native SR: {}, Target SR: {}, Channels: {}, Chunk Samples (native): {}", native_sample_rate, target_sample_rate, audio_channels, device_samples_per_step);

            let stream_result = match default_config.sample_format() {
                SampleFormat::F32 => {
                    info!("[Audio] Attempting to access audio device...");
                    Self::process_audio_stream_internal::<f32, _>(
                        &device, &config, tx_for_data_cb, audio_channels, device_samples_per_step, resampler_opt, tx_for_err_fn, |s: &f32| *s, stop_processing_signal.clone())
                }
                SampleFormat::I16 => {
                    info!("[Audio] Attempting to access audio device...");
                    Self::process_audio_stream_internal::<i16, _>(
                        &device, &config, tx_for_data_cb, audio_channels, device_samples_per_step, resampler_opt, tx_for_err_fn, |s: &i16| s.to_float_sample(), stop_processing_signal.clone())
                }
                SampleFormat::U16 => {
                    info!("[Audio] Attempting to access audio device...");
                    Self::process_audio_stream_internal::<u16, _>(
                        &device, &config, tx_for_data_cb, audio_channels, device_samples_per_step, resampler_opt, tx_for_err_fn, |s: &u16| s.to_float_sample(), stop_processing_signal.clone())
                }
                other_format => {
                    let err_msg = format!("Unsupported sample format: {:?}", other_format);
                    error!("[Audio] {}", err_msg);
                    let _ = tx.send(Err(WhisperStreamError::AudioStreamConfig(err_msg)));
                    return;
                }
            };

            match stream_result {
                Ok(stream) => {
                    info!("[Audio] Successfully created audio stream, attempting to start capture...");
                    if let Err(e) = stream.play() {
                        error!("[Audio] Failed to play stream for device: {} ({})", device_name_clone, e);
                        let _ = tx.send(Err(WhisperStreamError::AudioDevice(format!("Failed to play stream: {}", e))));
                        return;
                    }
                    info!("[Audio] Audio capture started successfully!");
                    // Stream is playing; cpal callbacks handle audio processing.
                    // This thread keeps alive to monitor stop_processing_signal.
                    loop {
                        if stop_processing_signal.load(Ordering::Relaxed) {
                            debug!("[Audio] Stop signal received in main capture thread for device '{}'. Stream will be dropped.", device_name_clone);
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                    // Dropping `stream` here stops cpal callbacks.
                }
                Err(e) => {
                    error!("[Audio] Failed to build input stream for device '{}': {}", device_name_clone, e);
                    let _ = tx.send(Err(e));
                    return;
                }
            }
             info!("[Audio] Audio capture thread for '{}' finished.", device_name_clone);
        });

        rx
    }

}