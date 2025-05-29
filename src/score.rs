/// Returns true if the text matches any low-quality output pattern (e.g., silence, blanks, bracketed/parenthesized).
///
/// # Arguments
/// * `text`: The input string to check.
///
/// # Returns
/// `true` if the text is considered low quality, `false` otherwise.
pub fn is_low_quality_output(text: &str) -> bool {
    let low_quality_patterns = [
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

    // Prepare text for pattern matching (case-insensitive, trim leading whitespace)
    let text_start_lower = text.trim_start().to_lowercase();

    // Check for any low-quality pattern
    if low_quality_patterns
        .iter()
        .any(|p| text_start_lower.starts_with(&p.to_lowercase()))
    {
        return true;
    }

    // Check if the text is fully enclosed in [] or ()
    let trimmed_text = text.trim();
    if trimmed_text.len() >= 2 &&
        ((trimmed_text.starts_with('[') && trimmed_text.ends_with(']')) ||
         (trimmed_text.starts_with('(') && trimmed_text.ends_with(')')))
    {
        return true;
    }

    false
}