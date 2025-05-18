//! Handles audio capture, processing, and streaming for real-time transcription.
//!
//! This module is responsible for interfacing with the system's audio input devices,
//! capturing raw audio data, and then processing it into a standardized format suitable
//! for speech recognition. The key processing steps include:
//! 1. Capturing audio based on a specified `step_ms` duration.
//! 2. Downmixing multi-channel audio to mono.
//! 3. Resampling the audio to 16kHz, which is expected by Whisper ASR models.
//!
//! The processed audio is provided as a stream of `Vec<f32>` chunks via a channel Receiver.
//! This chunk-based delivery is designed to be consumed by other parts of the application
//! (e.g., `stream.rs`) that typically manage a rolling window of audio data for continuous
//! transcription.

use std::sync::mpsc::{self, Receiver, Sender};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, StreamConfig, InputCallbackInfo, StreamError as CpalStreamError};
use rubato::{FftFixedInOut, Resampler};
use crate::error::WhisperStreamError; // Import custom error
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use log::{info, warn, error, debug};

/// Encapsulates audio input device information and configuration for capture.
///
/// It stores details like the device name, native sample rate, and channel count,
/// and provides the primary method (`start_capture_16k`) to begin streaming
/// processed audio data.
pub struct AudioInput {
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub buffer_size: usize,
    step_duration_ms: u32,
}

impl AudioInput {
    /// Lists the names of all available audio input devices on the system.
    pub fn available_input_devices() -> Result<Vec<String>, WhisperStreamError> {
        let host = cpal::default_host();
        let devices = host.input_devices().map_err(WhisperStreamError::from)?; // Uses From<DevicesError>
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

    /// Creates a new `AudioInput` instance, optionally targeting a specific input device.
    ///
    /// If `device_name_opt` is `None`, it uses the default system input device.
    /// If `Some(name)`, it attempts to find and use the device with the given name.
    /// The `step_ms` parameter defines the target duration for each audio chunk processed by `start_capture_16k`.
    ///
    /// It fetches the selected device's name, default configuration (sample rate, channels),
    /// and calculates an initial buffer size based on the provided `step_ms`.
    /// It also prints warnings if the selected device does not have 1 channel (mono),
    /// as mono audio is the target format after processing.
    pub fn new(device_name_opt: Option<&str>, step_ms: u32) -> Result<Self, WhisperStreamError> {
        let host = cpal::default_host();
        let device = match device_name_opt {
            Some(name) => {
                let devices = host.input_devices().map_err(WhisperStreamError::from)?;
                devices.into_iter() // Use into_iter for owned iterator if needed, or iter()
                    .find(|d| d.name().map(|n| n.eq_ignore_ascii_case(name)).unwrap_or(false))
                    .ok_or_else(|| WhisperStreamError::AudioDevice(format!("Input device named '{}' not found (case-insensitive search)", name)))?
            }
            None => {
                host.default_input_device()
                    .ok_or_else(|| WhisperStreamError::AudioDevice("No default input device available".to_string()))?
            }
        };

        let actual_device_name = match device.name() {
            Ok(n) => n,
            Err(e) => {
                warn!("[Audio] Warning: Could not get device name: {}. Using <Unknown>.", e);
                "<Unknown>".to_string()
            }
        };

        let config = device
            .default_input_config()
            .map_err(WhisperStreamError::from)?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        let buffer_size = (sample_rate as f32 * (step_ms as f32 / 1000.0)) as usize * channels as usize;

        info!("[Audio] Using input device: {}", actual_device_name);
        info!("[Audio] Device input config: SampleRate({}), Channels({}), Format({})", sample_rate, channels, config.sample_format());

        if channels != 1 {
            warn!("[Audio][Warning] Source input device '{}' has {} channels. It will be downmixed to mono.", actual_device_name, channels);
        }

        Ok(Self {
            device_name: actual_device_name,
            sample_rate,
            channels,
            buffer_size,
            step_duration_ms: step_ms,
        })
    }

    /// Internal helper function to build and manage the cpal input stream for a specific sample format.
    ///
    /// This generic function handles the core audio processing logic once the sample format is known:
    /// - It sets up a cpal input stream with the provided `convert_sample` closure to transform
    ///   input samples into `f32`.
    /// - Buffers incoming audio data until `device_samples_per_step` frames are collected.
    /// - Downmixes to mono if the input `audio_channels` is greater than 1.
    /// - Resamples the mono audio to 16kHz if `resampler_opt` is `Some`.
    /// - Sends the final processed `Vec<f32>` chunk through the `tx` channel sender.
    ///
    /// # Type Parameters
    /// * `T`: The cpal sample type (e.g., `f32`, `i16`, `u16`).
    /// * `F`: A closure that converts a sample of type `T` to `f32`.
    fn process_audio_stream_internal<T, F>(
        device: &cpal::Device,
        config: &StreamConfig,
        tx: Sender<Result<Vec<f32>, WhisperStreamError>>, // Channel sends Result
        audio_channels: usize,
        device_samples_per_step: usize,
        mut resampler_opt: Option<FftFixedInOut<f32>>,
        err_fn_outer_tx: Sender<Result<Vec<f32>, WhisperStreamError>>, // Used to send stream errors
        convert_sample: F,
        stop_processing_signal: Arc<AtomicBool>,
    ) -> Result<cpal::Stream, WhisperStreamError> // Return WhisperStreamError
    where
        T: cpal::SizedSample,
        F: Fn(&T) -> f32 + Send + Sync + 'static,
    {
        let mut buffer: Vec<f32> = Vec::with_capacity(device_samples_per_step * audio_channels);
        let callback_stop_signal = stop_processing_signal.clone();


        device.build_input_stream(
            config,
            move |data: &[T], _: &InputCallbackInfo| {
                if callback_stop_signal.load(Ordering::Relaxed) {
                    return;
                }

                buffer.extend(data.iter().map(&convert_sample));

                while buffer.len() >= device_samples_per_step * audio_channels {
                    if callback_stop_signal.load(Ordering::Relaxed) {
                        buffer.clear();
                        return;
                    }
                    let mut chunk: Vec<f32> = buffer.drain(..device_samples_per_step * audio_channels).collect();

                    if audio_channels > 1 {
                        chunk = chunk
                            .chunks_exact(audio_channels)
                            .map(|frame| frame.iter().sum::<f32>() / audio_channels as f32)
                            .collect();
                    }

                    let final_chunk_result = if let Some(resampler) = resampler_opt.as_mut() {
                        match resampler.process(&[chunk], None) {
                            Ok(mut output_frames) => {
                                if output_frames.is_empty() {
                                    // This isn't an error per se, but no data. Could be logged.
                                    // For now, skip sending empty data.
                                    // eprintln!("[Audio] Resampler processed but returned no frames. Skipping chunk.");
                                    continue; // Skip sending this empty chunk
                                }
                                Ok(output_frames.remove(0))
                            }
                            Err(e) => {
                                Err(WhisperStreamError::AudioResampling(format!("Resample failed: {:?}", e)))
                            }
                        }
                    } else {
                        Ok(chunk)
                    };

                    match final_chunk_result {
                        Ok(final_chunk) => {
                            if !final_chunk.is_empty() && tx.send(Ok(final_chunk)).is_err() {
                                error!("[Audio] Receiver dropped. Signalling to stop audio processing.");
                                callback_stop_signal.store(true, Ordering::Relaxed);
                                return;
                            }
                        }
                        Err(e) => {
                             if tx.send(Err(e)).is_err() {
                                error!("[Audio] Receiver dropped while sending resample error. Signalling to stop.");
                                callback_stop_signal.store(true, Ordering::Relaxed);
                                return;
                            }
                        }
                    }
                }
            },
            move |err: CpalStreamError| { // Use err_fn_outer_tx to send error
                error!("[Audio] Stream error: {}", err);
                let _ = err_fn_outer_tx.send(Err(WhisperStreamError::from(err)));
                // Consider stopping processing here too via stop_processing_signal
            },
            None,
        ).map_err(WhisperStreamError::from) // Converts BuildStreamError
    }

    /// Starts the audio capture and processing thread, returning a `Receiver` for 16kHz mono f32 audio chunks.
    ///
    /// This is the primary method to begin receiving audio data suitable for Whisper ASR.
    /// It uses the `step_duration_ms` configured for this `AudioInput` instance to determine chunk size.
    /// It performs the following steps in a spawned thread:
    /// 1. Initializes an audio input stream from the default device using its native configuration.
    /// 2. Sets up a resampler if the device's native sample rate is not 16kHz.
    /// 3. In a loop, for each `step_ms` interval:
    ///    a. Collects raw audio samples.
    ///    b. Converts samples to `f32` format.
    ///    c. Downmixes to mono if the input is multi-channel.
    ///    d. Resamples to 16kHz mono if necessary.
    ///    e. Sends the resulting `Vec<f32>` audio chunk via the returned channel `Receiver`.
    ///
    /// The `step_ms` parameter determines the duration of each audio chunk processed and sent.
    /// This chunk-based streaming allows a consumer (like `stream.rs`) to manage a
    /// rolling window of audio for continuous real-time transcription.
    pub fn start_capture_16k(&self) -> Receiver<Result<Vec<f32>, WhisperStreamError>> { // Receiver sends Result
        let (tx, rx) = mpsc::channel();
        let device_name_clone = self.device_name.clone();
        let step_duration_ms_clone = self.step_duration_ms;

        let stop_processing_signal = Arc::new(AtomicBool::new(false));
        let _thread_stop_signal = stop_processing_signal.clone(); // Renamed to avoid unused variable warning, assuming it might be used later or was for debugging

        let tx_for_err_fn = tx.clone(); // For the stream error callback
        let tx_for_data_cb = tx.clone(); // For the data callback

        info!("[Audio] Starting capture for device '{}'. Native SR: {}, Target SR: {}, Channels: {}, Chunk ms: {}",
            self.device_name, self.sample_rate, 16000, self.channels, self.step_duration_ms);

        std::thread::spawn(move || {
            let host = cpal::default_host();
            let device = match host.input_devices().map_err(WhisperStreamError::from) {
                Ok(mut devices) => devices.find(|d| d.name().map(|n| n == device_name_clone).unwrap_or(false)),
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            }.or_else(|| host.default_input_device())
             .ok_or_else(|| WhisperStreamError::AudioDevice(format!("Audio device '{}' not found and no default available.", device_name_clone)));

            let device = match device {
                Ok(d) => d,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            let config = match device.default_input_config().map_err(WhisperStreamError::from) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };

            let native_sample_rate = config.sample_rate().0;
            let audio_channels = config.channels() as usize;
            let target_sample_rate = 16000;
            let device_samples_per_step = (native_sample_rate as f32 * (step_duration_ms_clone as f32 / 1000.0)) as usize;

            let resampler_opt = if native_sample_rate != target_sample_rate {
                match FftFixedInOut::<f32>::new(
                    native_sample_rate as usize,
                    target_sample_rate as usize,
                    device_samples_per_step, // Max chunk size for resampler input
                    1, // Corrected: Always use 1 channel for mono after downmixing
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

            info!("[Audio] Starting capture. Native SR: {}, Target SR: {}, Channels: {}, Chunk Samples: {}", native_sample_rate, target_sample_rate, audio_channels, device_samples_per_step);

            let stream_result = match config.sample_format() {
                SampleFormat::F32 => {
                    let convert_sample = |s: &f32| *s;
                    Self::process_audio_stream_internal::<f32, _>(
                        &device, &config.into(), tx_for_data_cb, audio_channels, device_samples_per_step, resampler_opt, tx_for_err_fn, convert_sample, stop_processing_signal.clone())
                }
                SampleFormat::I16 => {
                    let convert_sample = |s: &i16| s.to_float_sample();
                    Self::process_audio_stream_internal::<i16, _>(
                        &device, &config.into(), tx_for_data_cb, audio_channels, device_samples_per_step, resampler_opt, tx_for_err_fn, convert_sample, stop_processing_signal.clone())
                }
                SampleFormat::U16 => {
                    let convert_sample = |s: &u16| s.to_float_sample();
                    Self::process_audio_stream_internal::<u16, _>(
                        &device, &config.into(), tx_for_data_cb, audio_channels, device_samples_per_step, resampler_opt, tx_for_err_fn, convert_sample, stop_processing_signal.clone())
                }
                // Handling other formats might require specific conversions or error reporting.
                other_format => {
                    let err_msg = format!("Unsupported sample format: {:?}", other_format);
                    error!("[Audio] {}", err_msg);
                    let _ = tx.send(Err(WhisperStreamError::AudioStreamConfig(err_msg)));
                    return;
                }
            };

            match stream_result {
                Ok(stream) => {
                    if stream.play().is_err() {
                        error!("[Audio] Failed to play stream: {}", device_name_clone); // Simplified error message
                        let _ = tx.send(Err(WhisperStreamError::AudioDevice("Failed to play stream".to_string())));
                        return;
                    }
                    // The stream is now playing. We need to keep this thread alive to process stop signals.
                    // The actual audio processing happens in cpal's callback threads.
                    loop {
                        if stop_processing_signal.load(Ordering::Relaxed) {
                            debug!("[Audio] Stop signal received in main capture thread for device '{}'. Stream will be dropped.", device_name_clone);
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100)); // Check for stop signal
                    }
                    // Dropping the stream here will stop the cpal callbacks
                }
                Err(e) => {
                    error!("[Audio] Failed to build input stream for device '{}': {}", device_name_clone, e);
                    let _ = tx.send(Err(e)); // Send the specific error
                    return;
                }
            }
             info!("[Audio] Audio capture thread for '{}' finished.", device_name_clone);
        });

        rx
    }

    /// Signals the audio capture thread to stop processing and clean up.
    ///
    /// This sets an atomic flag that the capture thread periodically checks.
    /// Once the flag is set, the capture thread will stop processing new audio data,
    /// allow the cpal stream to be dropped (which stops hardware capture), and then exit.
    pub fn stop_capture(stop_signal: Arc<AtomicBool>) {
        debug!("[Audio] Setting stop signal for audio capture.");
        stop_signal.store(true, Ordering::Relaxed);
    }
}