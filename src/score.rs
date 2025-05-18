/// Calculates a score for a text based on specific patterns.
///
/// - Returns `10` for high-priority patterns (e.g., "[silence]", "(buzzer)") (case-insensitive start match).
/// - Returns `5` if the text is enclosed in `[]` or `()` (e.g., "[some text]") and no high-priority pattern matched.
/// - Returns `0` otherwise.
///
/// # Arguments
/// * `text`: The input string to score.
///
/// # Returns
/// The score: `10`, `5`, or `0`.
pub fn calculate_score(text: &str) -> i32 {
    let high_priority_patterns = [
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

    // Prepare text for pattern matching (score 10)
    // Trim leading whitespace and convert to lowercase for case-insensitive `starts_with`
    let text_start_lower = text.trim_start().to_lowercase();

    // Check for high-priority patterns
    if high_priority_patterns
        .iter()
        .any(|p| text_start_lower.starts_with(&p.to_lowercase()))
    {
        return 10; // Highest priority score
    }

    // Score 5 for text fully enclosed in [] or ()
    let trimmed_text = text.trim();

    // Check if the text is enclosed by matching brackets or parentheses
    if trimmed_text.len() >= 2 && // Must have at least two chars to be an enclosure
        ((trimmed_text.starts_with('[') && trimmed_text.ends_with(']')) ||
         (trimmed_text.starts_with('(') && trimmed_text.ends_with(')')))
    {
        return 5; // Score for bracketed/parenthesized text
    }

    0 // Default score if no conditions met
}