use std::borrow::Cow;
use hound::{WavWriter, WavSpec, SampleFormat};
use crate::error::WhisperStreamError; // Assuming WhisperStreamError is accessible via crate::error

/// Pads an audio segment with silence if it's shorter than `min_samples`.
///
/// # Arguments
/// * `audio_segment`: The input audio segment.
/// * `min_samples`: The minimum number of samples the output segment should have.
///
/// # Returns
/// A `Cow<[f32]>` which is either a borrowed slice of the original audio
/// if no padding was needed, or an owned, padded `Vec<f32>`.
pub fn pad_audio_if_needed(audio_segment: &[f32], min_samples: usize) -> Cow<'_, [f32]> {
    if audio_segment.len() < min_samples {
        let mut padded_segment = audio_segment.to_vec();
        let padding_needed = min_samples - padded_segment.len();
        padded_segment.extend(std::iter::repeat(0.0f32).take(padding_needed));
        Cow::Owned(padded_segment)
    } else {
        Cow::Borrowed(audio_segment)
    }
}

/// Handles recording audio chunks to a WAV file.
pub struct WavAudioRecorder {
    writer: Option<WavWriter<std::io::BufWriter<std::fs::File>>>,
    path: String,
    is_recording_active: bool,
}

impl WavAudioRecorder {
    /// Creates a new `WavAudioRecorder`.
    ///
    /// # Arguments
    /// * `path_opt`: Optional path to save the WAV file. If `None`, recording is disabled.
    pub fn new(path_opt: Option<&str>) -> Result<Self, WhisperStreamError> {
        match path_opt {
            Some(p) => {
                let spec = WavSpec {
                    channels: 1,        // Whisper processes mono audio
                    sample_rate: 16000, // Whisper processes 16kHz audio
                    bits_per_sample: 16,
                    sample_format: SampleFormat::Int,
                };
                let writer = WavWriter::create(p, spec)
                    .map_err(|e| WhisperStreamError::Hound { source: e })?;
                Ok(Self {
                    writer: Some(writer),
                    path: p.to_string(),
                    is_recording_active: true,
                })
            }
            None => Ok(Self {
                writer: None,
                path: String::new(),
                is_recording_active: false,
            }),
        }
    }

    /// Writes an audio chunk to the WAV file if recording is active.
    ///
    /// # Arguments
    /// * `audio_chunk`: A slice of `f32` audio samples (expected to be mono, 16kHz).
    ///                  Samples should be in the range -1.0 to 1.0.
    pub fn write_audio_chunk(&mut self, audio_chunk: &[f32]) -> Result<(), WhisperStreamError> {
        if let Some(writer) = self.writer.as_mut() {
            for &sample_f32 in audio_chunk {
                // Convert f32 sample (range -1.0 to 1.0) to i16
                let sample_i16 = (sample_f32 * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                if let Err(e) = writer.write_sample(sample_i16) {
                    return Err(WhisperStreamError::Hound { source: e });
                }
            }
        }
        Ok(())
    }

    /// Finalizes the WAV file. Must be called to complete the recording.
    /// Returns a system message indicating the result.
    pub fn finalize(mut self) -> Result<Option<String>, WhisperStreamError> {
        if let Some(writer) = self.writer.take() {
            // `take()` invalidates self.writer, ensuring finalize is called once.
            writer.finalize().map_err(|e| WhisperStreamError::Hound { source: e })?;
            Ok(Some(format!("[Recording] Finished saving audio to {}", self.path)))
        } else if self.is_recording_active && self.path.is_empty() {
            // This case should ideally not be hit if construction logic is correct
            Ok(Some("[Recording] Recording was intended but path was empty; nothing saved.".to_string()))
        }
         else if !self.is_recording_active {
            Ok(None) // No recording was active, so no message needed or an empty one.
        }
        else {
            // Should not happen if writer was Some initially and path was set.
             Ok(Some("[Recording] No active recording to finalize or already finalized.".to_string()))
        }
    }

    pub fn is_recording(&self) -> bool {
        self.is_recording_active
    }
}