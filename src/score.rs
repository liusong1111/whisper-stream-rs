pub fn calculate_score(text: &str) -> i32 {
    let patterns = [
        "[ Silence ]",
        "[silence]",
        "[BLANK",
        "[BLANK_AUDIO]",
        "[ [ [ [",
        "[ [ [",
        "[ [",
        "[ ",
        "(buzzer)",
        "(buzzing)",
    ];
    for pattern in patterns.iter() {
        if text.trim_start().starts_with(pattern) {
            return 10;
        }
    }
    0
}