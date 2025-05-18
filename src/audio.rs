use std::sync::mpsc::{self, Receiver};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};

/// Starts microphone capture and returns a receiver for audio buffers.
pub fn start_audio_capture(step_ms: u32) -> Receiver<Vec<f32>> {
    let (tx, rx) = mpsc::channel();
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device available");
    let config = device.default_input_config().expect("No default input config");
    let sample_rate = 16000;
    let channels = 1;
    let samples_per_step = (sample_rate as f32 * (step_ms as f32 / 1000.0)) as usize;
    let err_fn = |err| eprintln!("Stream error: {err}");

    let tx_clone = tx.clone();
    let sample_format = config.sample_format();
    let config: StreamConfig = config.into();
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