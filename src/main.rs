fn main() {
    let rx = whisper_stream_rs::start_transcription_stream();
    println!("[Start speaking]");
    for transcript in rx {
        println!("{}", transcript);
    }
}