use std::sync::mpsc::{self, Receiver};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use rubato::{FftFixedInOut, Resampler};

/// Struct to encapsulate audio input device and config
pub struct AudioInput {
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub buffer_size: usize,
}

impl AudioInput {
    /// Create a new AudioInput from the default input device
    pub fn new(step_ms: u32) -> Self {
        let host = cpal::default_host();
        let device = host.default_input_device().expect("No input device available");
        let device_name = device.name().unwrap_or("<Unknown>".to_string());
        let config = device.default_input_config().expect("No default input config");
        let sample_rate = config.sample_rate().0;
        let channels = config.channels();
        let buffer_size = (sample_rate as f32 * (step_ms as f32 / 1000.0)) as usize;
        println!("[Audio] Using input device: {}", device_name);
        println!("[Audio] Default input config: {:?}", config);
        if channels != 1 {
            eprintln!("[Audio][Warning] Default input device has {} channels, but code expects 1 channel (mono).", channels);
        }
        Self {
            device_name,
            sample_rate,
            channels,
            buffer_size,
        }
    }

    /// Start audio capture, resample to 16kHz mono. Returns a receiver of 16kHz mono f32 buffers.
    pub fn start_capture_16k(&self, step_ms: u32) -> Receiver<Vec<f32>> {
        let (tx, rx) = mpsc::channel();
        let host = cpal::default_host();
        let device = host.default_input_device().expect("No input device available");
        let config = device.default_input_config().expect("No default input config");
        let sample_format = config.sample_format();
        let config: StreamConfig = config.into();
        let device_sample_rate = self.sample_rate;
        let channels = self.channels as usize;
        let device_samples_per_step = (device_sample_rate as f32 * (step_ms as f32 / 1000.0)) as usize;
        let need_resample = device_sample_rate != 16000;
        std::thread::spawn(move || {
            let err_fn = |err| eprintln!("Stream error: {err}");
            let mut buffer = Vec::with_capacity(device_samples_per_step * channels);
            let resampler = if need_resample {
                Some(FftFixedInOut::<f32>::new(
                    device_sample_rate as usize,
                    16000,
                    device_samples_per_step,
                    1,
                ).expect("Failed to create resampler"))
            } else {
                None
            };
            let stream = match sample_format {
                SampleFormat::F32 => {
                    let tx = tx.clone();
                    let channels = channels;
                    let mut resampler = resampler;
                    device.build_input_stream(
                        &config,
                        move |data: &[f32], _| {
                            buffer.extend_from_slice(data);
                            while buffer.len() >= device_samples_per_step * channels {
                                let mut chunk: Vec<f32> = buffer.drain(..device_samples_per_step * channels).collect();
                                // Downmix to mono if needed
                                if channels > 1 {
                                    chunk = chunk
                                        .chunks(channels)
                                        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                                        .collect();
                                }
                                // Resample if needed
                                let chunk = if let Some(resampler) = resampler.as_mut() {
                                    let input = vec![chunk];
                                    let output = resampler.process(&input, None).expect("Resample failed");
                                    output.into_iter().next().unwrap()
                                } else {
                                    chunk
                                };
                                let _ = tx.send(chunk);
                            }
                        },
                        err_fn,
                        None,
                    )
                }
                SampleFormat::I16 => {
                    let tx = tx.clone();
                    let channels = channels;
                    let mut resampler = resampler;
                    device.build_input_stream(
                        &config,
                        move |data: &[i16], _| {
                            buffer.extend(data.iter().map(|&s| s as f32 / i16::MAX as f32));
                            while buffer.len() >= device_samples_per_step * channels {
                                let mut chunk: Vec<f32> = buffer.drain(..device_samples_per_step * channels).collect();
                                if channels > 1 {
                                    chunk = chunk
                                        .chunks(channels)
                                        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                                        .collect();
                                }
                                let chunk = if let Some(resampler) = resampler.as_mut() {
                                    let input = vec![chunk];
                                    let output = resampler.process(&input, None).expect("Resample failed");
                                    output.into_iter().next().unwrap()
                                } else {
                                    chunk
                                };
                                let _ = tx.send(chunk);
                            }
                        },
                        err_fn,
                        None,
                    )
                }
                SampleFormat::U16 => {
                    let tx = tx.clone();
                    let channels = channels;
                    let mut resampler = resampler;
                    device.build_input_stream(
                        &config,
                        move |data: &[u16], _| {
                            buffer.extend(data.iter().map(|&s| s as f32 / u16::MAX as f32 - 0.5));
                            while buffer.len() >= device_samples_per_step * channels {
                                let mut chunk: Vec<f32> = buffer.drain(..device_samples_per_step * channels).collect();
                                if channels > 1 {
                                    chunk = chunk
                                        .chunks(channels)
                                        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                                        .collect();
                                }
                                let chunk = if let Some(resampler) = resampler.as_mut() {
                                    let input = vec![chunk];
                                    let output = resampler.process(&input, None).expect("Resample failed");
                                    output.into_iter().next().unwrap()
                                } else {
                                    chunk
                                };
                                let _ = tx.send(chunk);
                            }
                        },
                        err_fn,
                        None,
                    )
                }
                _ => panic!("Unsupported sample format"),
            }.expect("Failed to build input stream");
            stream.play().expect("Failed to play stream");
            std::thread::park();
        });
        rx
    }
}

/// Starts microphone capture and returns a receiver for audio buffers.
pub fn start_audio_capture(step_ms: u32) -> Receiver<Vec<f32>> {
    let (tx, rx) = mpsc::channel();
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device available");
    println!("[Audio] Using input device: {}", device.name().unwrap_or("<Unknown>".to_string()));
    let config = device.default_input_config().expect("No default input config");
    println!("[Audio] Default input config: {:?}", config);
    let sample_rate = 16000;
    let channels = 1;
    if config.channels() != channels {
        eprintln!("[Audio][Warning] Default input device has {} channels, but code expects {} channel(s).", config.channels(), channels);
    }
    if config.sample_rate().0 != sample_rate {
        eprintln!("[Audio][Warning] Default input device sample rate is {}, but code expects {}.", config.sample_rate().0, sample_rate);
    }
    let samples_per_step = (sample_rate as f32 * (step_ms as f32 / 1000.0)) as usize;
    let err_fn = |err| eprintln!("Stream error: {err}");

    let tx_clone = tx.clone();
    let sample_format = config.sample_format();
    println!("[Audio] Sample format: {:?}", sample_format);
    let config: StreamConfig = config.into();
    println!("[Audio] StreamConfig used: {:?}", config);
    std::thread::spawn(move || {
        let mut buffer = Vec::with_capacity(samples_per_step);
        let stream = match sample_format {
            SampleFormat::F32 => device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    for &sample in data {
                        buffer.push(sample);
                        if buffer.len() >= samples_per_step {
                            let chunk = buffer.split_off(0);
                            if tx_clone.send(chunk).is_err() {
                                return;
                            }
                        }
                    }
                },
                err_fn,
                None,
            ),
            SampleFormat::I16 => device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    for &sample in data {
                        buffer.push(sample as f32 / i16::MAX as f32);
                        if buffer.len() >= samples_per_step {
                            let chunk = buffer.split_off(0);
                            if tx_clone.send(chunk).is_err() {
                                return;
                            }
                        }
                    }
                },
                err_fn,
                None,
            ),
            SampleFormat::U16 => device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    for &sample in data {
                        buffer.push(sample as f32 / u16::MAX as f32 - 0.5);
                        if buffer.len() >= samples_per_step {
                            let chunk = buffer.split_off(0);
                            if tx_clone.send(chunk).is_err() {
                                return;
                            }
                        }
                    }
                },
                err_fn,
                None,
            ),
            _ => panic!("Unsupported sample format"),
        }.expect("Failed to build input stream");
        stream.play().expect("Failed to play stream");
        std::thread::park();
    });
    rx
}