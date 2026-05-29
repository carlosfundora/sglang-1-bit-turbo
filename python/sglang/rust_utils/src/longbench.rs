use regex::Regex;
use std::sync::OnceLock;

static RE_LONGBENCH_1: OnceLock<Regex> = OnceLock::new();
static RE_LONGBENCH_2: OnceLock<Regex> = OnceLock::new();
static RE_LONGBENCH_3: OnceLock<Regex> = OnceLock::new();
static RE_LONGBENCH_4: OnceLock<Regex> = OnceLock::new();

pub fn extract_longbench_v2_answer(response: &str) -> Option<String> {
    let response = response.replace("*", "");

    let re1 = RE_LONGBENCH_1.get_or_init(|| Regex::new(r"(?i)The correct answer is \(([A-D])\)").unwrap());
    if let Some(caps) = re1.captures(&response) {
        return Some(caps.get(1).unwrap().as_str().to_uppercase());
    }

    let re2 = RE_LONGBENCH_2.get_or_init(|| Regex::new(r"(?i)The correct answer is ([A-D])").unwrap());
    if let Some(caps) = re2.captures(&response) {
        return Some(caps.get(1).unwrap().as_str().to_uppercase());
    }

    let re3 = RE_LONGBENCH_3.get_or_init(|| Regex::new(r"(?i)Answer\s*:\s*([A-D])").unwrap());
    if let Some(caps) = re3.captures(&response) {
        return Some(caps.get(1).unwrap().as_str().to_uppercase());
    }

    let re4 = RE_LONGBENCH_4.get_or_init(|| Regex::new(r"(?i)answer\s+is\s*\(?([A-D])\)?").unwrap());
    if let Some(caps) = re4.captures(&response) {
        return Some(caps.get(1).unwrap().as_str().to_uppercase());
    }

    None
}
