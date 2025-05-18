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
use cpal::{SampleFormat, StreamConfig, SizedSample, InputCallbackInfo, StreamError};
use rubato::{FftFixedInOut, Resampler, ResampleError};
use anyhow::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
    pub fn available_input_devices() -> anyhow::Result<Vec<String>> {
        let host = cpal::default_host();
        let devices = host.input_devices().map_err(|e| anyhow::anyhow!("Failed to list input devices: {}", e))?;
        let mut device_names = Vec::new();
        for device in devices {
            match device.name() {
                Ok(name) => device_names.push(name),
                Err(e) => {
                    // Log or print a warning for devices whose names can't be fetched
                    eprintln!("[Audio] Warning: Could not get name for an input device: {}", e);
                    device_names.push("<Unnamed Device>".to_string()); // Add a placeholder
                }
            }
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
    pub fn new(device_name_opt: Option<&str>, step_ms: u32) -> anyhow::Result<Self> {
        let host = cpal::default_host();
        let device = match device_name_opt {
            Some(name) => {
                let mut devices = host.input_devices().map_err(|e| anyhow::anyhow!("Failed to list input devices: {}", e))?;
                devices.find(|d| d.name().map(|n| n == name).unwrap_or(false))
                    .ok_or_else(|| anyhow::anyhow!("Input device named '{}' not found", name))?
            }
            None => {
                host.default_input_device()
                    .ok_or_else(|| anyhow::anyhow!("No default input device available"))?
            }
        };

        let actual_device_name = device.name().unwrap_or("<Unknown>".to_string());
        let config = device
            .default_input_config()
            .map_err(|e| anyhow::anyhow!("No default input config for device '{}': {}", actual_device_name, e))?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        // buffer_size is primarily for the Vec::with_capacity in process_audio_stream_internal,
        // which is dynamically calculated there based on device_samples_per_step * audio_channels.
        // We can keep this field for informational purposes or if other parts might use it.
        // For now, let's calculate it based on the chosen device's sample rate and the step_ms for this AudioInput instance.
        let buffer_size = (sample_rate as f32 * (step_ms as f32 / 1000.0)) as usize * channels as usize;

        println!("[Audio] Using input device: {}", actual_device_name);
        println!("[Audio] Device input config: {:?}", config);

        if channels != 1 {
            // This warning is about the *source* device. Downmixing to mono happens later.
            eprintln!("[Audio][Warning] Source input device '{}' has {} channels. It will be downmixed to mono.", actual_device_name, channels);
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
        tx: Sender<Vec<f32>>,
        audio_channels: usize,
        device_samples_per_step: usize,
        mut resampler_opt: Option<FftFixedInOut<f32>>,
        err_fn_outer: impl Fn(StreamError) + Send + Sync + 'static + Clone,
        convert_sample: F,
        stop_processing_signal: Arc<AtomicBool>,
    ) -> Result<cpal::Stream, Error>
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

                    let final_chunk = if let Some(resampler) = resampler_opt.as_mut() {
                        match resampler.process(&[chunk], None) {
                            Ok(mut output_frames) => {
                                if output_frames.is_empty() {
                                    eprintln!("[Audio] Resampler processed but returned no frames. Skipping chunk.");
                                    Vec::new()
                                } else {
                                    output_frames.remove(0)
                                }
                            }
                            Err(e) => {
                                eprintln!("[Audio] Resample failed: {:?}. Skipping chunk.", e);
                                Vec::new()
                            }
                        }
                    } else {
                        chunk
                    };

                    if !final_chunk.is_empty() {
                        if tx.send(final_chunk).is_err() {
                            eprintln!("[Audio] Receiver dropped. Signalling to stop audio processing.");
                            callback_stop_signal.store(true, Ordering::Relaxed);
                            return;
                        }
                    }
                }
            },
            move |err| err_fn_outer(err),
            None, // Timeout
        ).map_err(Error::from)
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
    pub fn start_capture_16k(&self) -> Receiver<Vec<f32>> {
        let (tx_main, rx_main) = mpsc::channel();
        let host = cpal::default_host();
        // Re-fetch the device based on self.device_name to ensure we use the same one as in new()
        // This is a bit redundant but ensures consistency if host state could change or if new() didn't store the cpal::Device itself.
        // A better approach might be to store the cpal::Device in Self if its lifetime allows, or re-query by name.
        // For now, re-querying by name stored in self.device_name.
        let device = host.input_devices()
            .map_err(|e| anyhow::anyhow!("Failed to list input devices during start_capture: {}", e))
            .and_then(|mut devs|
                devs.find(|d| d.name().map(|n| n == self.device_name).unwrap_or(false))
                    .ok_or_else(|| anyhow::anyhow!("Audio device '{}' used during init not found at start_capture", self.device_name)))
            .expect("Critical: Audio device disappeared or name changed since AudioInput::new"); // Panicking here as state is inconsistent

        let input_config = device.default_input_config()
            .expect(&format!("No default input config for device '{}' at start_capture", self.device_name));

        let stream_sample_format = input_config.sample_format();
        let stream_config_cpal: StreamConfig = input_config.into();

        let device_actual_sample_rate = self.sample_rate; // This is from the config fetched in new() for the selected device
        let audio_channels = self.channels as usize;      // Same here
        // Use self.step_duration_ms for calculations
        let device_samples_per_step = (device_actual_sample_rate as f32 * (self.step_duration_ms as f32 / 1000.0)) as usize;
        let need_resample = device_actual_sample_rate != 16000;

        let stop_processing_signal_arc = Arc::new(AtomicBool::new(false));

        let resampler_option = if need_resample {
            Some(FftFixedInOut::<f32>::new(
                device_actual_sample_rate as usize,
                16000,
                device_samples_per_step,
                1, // Input to resampler is always 1 channel (mono)
            ).expect("Failed to create resampler"))
        } else {
            None
        };

        std::thread::spawn(move || {
            let err_fn_callback = |err: StreamError| eprintln!("[Audio] Stream error: {err}");
            let thread_stop_signal = stop_processing_signal_arc.clone();

            let stream_result = match stream_sample_format {
                SampleFormat::F32 => Self::process_audio_stream_internal::<f32, _>(
                    &device, &stream_config_cpal, tx_main.clone(), audio_channels, device_samples_per_step,
                    resampler_option,
                    err_fn_callback.clone(),
                    |sample_val| *sample_val,
                    thread_stop_signal.clone(),
                ),
                SampleFormat::I16 => Self::process_audio_stream_internal::<i16, _>(
                    &device, &stream_config_cpal, tx_main.clone(), audio_channels, device_samples_per_step,
                    resampler_option,
                    err_fn_callback.clone(),
                    |sample_val| *sample_val as f32 / 32768.0,
                    thread_stop_signal.clone(),
                ),
                SampleFormat::U16 => Self::process_audio_stream_internal::<u16, _>(
                    &device, &stream_config_cpal, tx_main.clone(), audio_channels, device_samples_per_step,
                    resampler_option,
                    err_fn_callback.clone(),
                    |sample_val| (*sample_val as f32 / u16::MAX as f32) * 2.0 - 1.0,
                    thread_stop_signal.clone(),
                ),
                unsupported_format => {
                    eprintln!("[Audio] Unsupported sample format: {:?}", unsupported_format);
                    Err(Error::msg(format!("Unsupported sample format: {:?}", unsupported_format)))
                }
            };

            match stream_result {
                Ok(stream) => {
                    if let Err(e) = stream.play() {
                        eprintln!("[Audio] Failed to play stream: {:?}", e);
                    } else {
                        std::thread::park(); // Keep thread alive while stream is playing
                    }
                }
                Err(e) => {
                    eprintln!("[Audio] Failed to build audio stream: {:?}", e);
                }
            }
        });

        rx_main
    }
}