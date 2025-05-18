use whisper_stream_rs::stream::TranscriptionConfig;

fn main() {
    whisper_rs::install_logging_hooks();
    let config = TranscriptionConfig {
        record_to_wav: Some("test_recording.wav".to_string()),
        ..Default::default()
    };
    let rx = whisper_stream_rs::start_transcription_stream(config);
    println!("[Start speaking]");
    for event in rx {
        match event {
            whisper_stream_rs::TranscriptionStreamEvent::Transcript { text, is_final } => {
                if is_final {
                    println!("[FINAL] {}", text);
                } else {
                    println!("{}", text);
                }
            }
            whisper_stream_rs::TranscriptionStreamEvent::SystemMessage(msg) => {
                println!("[SYSTEM] {}", msg);
            }
            whisper_stream_rs::TranscriptionStreamEvent::Error(err) => {
                eprintln!("[ERROR] {}", err);
            }
        }
    }
}