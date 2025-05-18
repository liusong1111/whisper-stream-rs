fn main() {
    whisper_rs::install_logging_hooks();
    let rx = whisper_stream_rs::start_transcription_stream();
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