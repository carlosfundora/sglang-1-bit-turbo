use regex::Regex;
use std::sync::OnceLock;

static CORRECT_PAREN_RE: OnceLock<Regex> = OnceLock::new();
static CORRECT_BARE_RE: OnceLock<Regex> = OnceLock::new();
static ANSWER_COLON_RE: OnceLock<Regex> = OnceLock::new();
static ANSWER_IS_RE: OnceLock<Regex> = OnceLock::new();

pub fn extract_longbench_v2_answer(response: &str) -> Option<String> {
    let response = response.replace('*', "");

    let re = CORRECT_PAREN_RE
        .get_or_init(|| Regex::new(r"(?i)The correct answer is \(([A-D])\)").expect("valid regex"));
    if let Some(caps) = re.captures(&response) {
        return Some(caps.get(1)?.as_str().to_uppercase());
    }

    let re = CORRECT_BARE_RE
        .get_or_init(|| Regex::new(r"(?i)The correct answer is ([A-D])").expect("valid regex"));
    if let Some(caps) = re.captures(&response) {
        return Some(caps.get(1)?.as_str().to_uppercase());
    }

    let re = ANSWER_COLON_RE
        .get_or_init(|| Regex::new(r"(?i)Answer\s*:\s*([A-D])").expect("valid regex"));
    if let Some(caps) = re.captures(&response) {
        return Some(caps.get(1)?.as_str().to_uppercase());
    }

    let re = ANSWER_IS_RE
        .get_or_init(|| Regex::new(r"(?i)answer\s+is\s*\(?([A-D])\)?").expect("valid regex"));
    if let Some(caps) = re.captures(&response) {
        return Some(caps.get(1)?.as_str().to_uppercase());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::extract_longbench_v2_answer;

    #[test]
    fn extracts_official_longbench_answers() {
        assert_eq!(
            extract_longbench_v2_answer("The correct answer is (B)").as_deref(),
            Some("B")
        );
        assert_eq!(
            extract_longbench_v2_answer("The correct answer is c").as_deref(),
            Some("C")
        );
        assert_eq!(
            extract_longbench_v2_answer("Answer: D").as_deref(),
            Some("D")
        );
        assert_eq!(
            extract_longbench_v2_answer("answer is a").as_deref(),
            Some("A")
        );
    }

    #[test]
    fn strips_markdown_and_returns_none_when_missing() {
        assert_eq!(
            extract_longbench_v2_answer("**The correct answer is (a)**").as_deref(),
            Some("A")
        );
        assert_eq!(extract_longbench_v2_answer("No final choice here"), None);
    }
}
