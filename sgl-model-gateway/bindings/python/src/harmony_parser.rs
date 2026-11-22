use pyo3::prelude::*;
use regex::Regex;
use std::sync::OnceLock;

// Existing prefix_hold
#[pyfunction]
fn prefix_hold(text: &str, tokens: Vec<String>) -> (String, String) {
    if text.is_empty() {
        return (String::new(), String::new());
    }

    let mut char_indices: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    char_indices.push(text.len());

    let text_char_len = char_indices.len() - 1;
    let mut max_hold_chars = 0;

    for tok in tokens {
        if tok.is_empty() {
            continue;
        }

        let tok_char_len = tok.chars().count();
        let l = std::cmp::min(tok_char_len - 1, text_char_len);

        for k in (1..=l).rev() {
            let start_byte = char_indices[text_char_len - k];
            let suffix = &text[start_byte..];
            if tok.starts_with(suffix) {
                max_hold_chars = std::cmp::max(max_hold_chars, k);
                break;
            }
        }
    }

    if max_hold_chars == 0 {
        (text.to_string(), String::new())
    } else {
        let split_byte = char_indices[text_char_len - max_hold_chars];
        let (emit, hold) = text.split_at(split_byte);
        (emit.to_string(), hold.to_string())
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Event {
    #[pyo3(get, set)]
    pub event_type: String,
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub raw_text: Option<String>,
}

#[pymethods]
impl Event {
    #[new]
    #[pyo3(signature = (event_type, content, raw_text=None))]
    fn new(event_type: String, content: String, raw_text: Option<String>) -> Self {
        Event {
            event_type,
            content,
            raw_text,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct Token {
    #[pyo3(get, set)]
    pub token_type: String,
    #[pyo3(get, set)]
    pub start: usize,
    #[pyo3(get, set)]
    pub end: usize,
}

#[pymethods]
impl Token {
    #[new]
    fn new(token_type: String, start: usize, end: usize) -> Self {
        Token {
            token_type,
            start,
            end,
        }
    }
}

fn iter_tokens(text: &str, start_pos: usize) -> Vec<Token> {
    let tokens_def = [
        ("<|start|>", "START"),
        ("<|channel|>", "CHANNEL"),
        ("<|message|>", "MESSAGE"),
        ("<|constrain|>", "CONSTRAIN"),
        ("<|end|>", "END"),
        ("<|call|>", "CALL"),
        ("<|return|>", "RETURN"),
    ];

    let mut result = Vec::new();
    let mut pos = start_pos;
    let len = text.len();

    while pos < len {
        if let Some(rel_marker_pos) = text[pos..].find("<|") {
            let marker_pos = pos + rel_marker_pos;

            if marker_pos > pos {
                result.push(Token::new("TEXT".to_string(), pos, marker_pos));
            }

            let mut found_token = false;
            for (literal, token_type) in &tokens_def {
                if text[marker_pos..].starts_with(literal) {
                    result.push(Token::new(token_type.to_string(), marker_pos, marker_pos + literal.len()));
                    pos = marker_pos + literal.len();
                    found_token = true;
                    break;
                }
            }

            if !found_token {
                let tail = &text[marker_pos..];
                let mut is_partial = false;
                for (literal, _) in &tokens_def {
                    if literal.starts_with(tail) {
                        is_partial = true;
                        break;
                    }
                }

                if is_partial {
                    result.push(Token::new("TEXT".to_string(), marker_pos, len));
                    pos = len;
                    break;
                } else {
                    // Not a known token, advance by 2 (the `<|`) and continue looking
                    result.push(Token::new("TEXT".to_string(), marker_pos, marker_pos + 2));
                    pos = marker_pos + 2;
                }
            }
        } else {
            result.push(Token::new("TEXT".to_string(), pos, len));
            pos = len;
            break;
        }
    }

    // Merge consecutive TEXT tokens
    let mut merged: Vec<Token> = Vec::new();
    for token in result {
        if let Some(last) = merged.last_mut() {
            if last.token_type == "TEXT" && token.token_type == "TEXT" && last.end == token.start {
                last.end = token.end;
                continue;
            }
        }
        merged.push(token);
    }

    merged
}

#[pyclass]
pub struct CanonicalStrategy {
    guard_tokens: Vec<String>,
}

#[pymethods]
impl CanonicalStrategy {
    #[new]
    fn new() -> Self {
        CanonicalStrategy {
            guard_tokens: vec![
                "<|start|>".to_string(),
                "<|channel|>".to_string(),
                "<|message|>".to_string(),
                "<|constrain|>".to_string(),
                "<|end|>".to_string(),
                "<|call|>".to_string(),
                "<|return|>".to_string(),
            ],
        }
    }

    fn parse(&self, text: &str) -> PyResult<(Vec<Event>, String)> {
        let mut events = Vec::new();
        let tokens = iter_tokens(text, 0);

        if tokens.is_empty() {
            return Ok((events, String::new()));
        }

        let mut pos = 0;
        while pos < tokens.len() {
            let token = &tokens[pos];

            if token.token_type == "TEXT" {
                if pos == tokens.len() - 1 {
                    let content = &text[token.start..token.end];
                    let (emit, hold) = prefix_hold(content, self.guard_tokens.clone());
                    if !emit.is_empty() {
                        events.push(Event::new("normal".to_string(), emit, None));
                    }
                    return Ok((events, hold));
                } else {
                    if Self::is_commentary_filler_between_blocks(text, &tokens, pos) {
                        pos += 1;
                    } else {
                        let content = &text[token.start..token.end];
                        if !Self::is_standalone_structural_token(content) {
                            events.push(Event::new("normal".to_string(), content.to_string(), None));
                        }
                        pos += 1;
                    }
                }
            } else if token.token_type == "START" || token.token_type == "CHANNEL" {
                if let Some((event_opt, new_pos)) = self.parse_block(text, &tokens, pos) {
                    if let Some(event) = event_opt {
                        events.push(event);
                    }
                    pos = new_pos;
                } else {
                    if let Some((event, remaining_text)) = self.parse_partial_analysis(text, &tokens, pos) {
                        events.push(event);
                        return Ok((events, remaining_text));
                    }
                    let remaining_start = tokens[pos].start;
                    return Ok((events, text[remaining_start..].to_string()));
                }
            } else {
                if Self::is_commentary_filler_between_blocks(text, &tokens, pos) {
                    pos += 1;
                } else {
                    let content = &text[token.start..token.end];
                    if !Self::is_standalone_structural_token(content) {
                        events.push(Event::new("normal".to_string(), content.to_string(), None));
                    }
                    pos += 1;
                }
            }
        }

        Ok((events, String::new()))
    }
}

impl CanonicalStrategy {
    fn parse_partial_analysis(&self, text: &str, tokens: &[Token], start_pos: usize) -> Option<(Event, String)> {
        let mut pos = start_pos;
        if pos < tokens.len() && tokens[pos].token_type == "START" {
            pos += 1;
        }

        let mut channel_pos = None;
        let mut message_pos = None;

        for i in pos..tokens.len() {
            if tokens[i].token_type == "CHANNEL" && channel_pos.is_none() {
                channel_pos = Some(i);
            } else if tokens[i].token_type == "MESSAGE" {
                message_pos = Some(i);
                break;
            }
        }

        if channel_pos.is_none() || message_pos.is_none() {
            return None;
        }

        let c_pos = channel_pos.unwrap();
        let m_pos = message_pos.unwrap();

        let channel_start = if c_pos + 1 < tokens.len() {
            tokens[c_pos + 1].start
        } else {
            tokens[c_pos].end
        };
        let channel_end = tokens[m_pos].start;
        let channel_header = &text[channel_start..channel_end];

        let channel_type = Self::extract_channel_type(channel_header);
        if channel_type.as_deref() != Some("analysis") {
            return None;
        }

        let content_start = tokens[m_pos].end;
        let content = &text[content_start..];

        if content.is_empty() {
            let remain = text[tokens[start_pos].start..].to_string();
            Some((Event::new("reasoning".to_string(), "".to_string(), None), remain))
        } else {
            let (emit, hold) = prefix_hold(content, self.guard_tokens.clone());
            let remain = text[tokens[start_pos].start..tokens[m_pos].end].to_string() + &hold;
            Some((Event::new("reasoning".to_string(), emit, None), remain))
        }
    }

    fn extract_channel_type(header_text: &str) -> Option<String> {
        let header_clean = header_text.trim().to_lowercase();
        if header_clean.starts_with("analysis") {
            Some("analysis".to_string())
        } else if header_clean.starts_with("commentary") {
            Some("commentary".to_string())
        } else if header_clean.starts_with("final") {
            Some("final".to_string())
        } else {
            None
        }
    }

    fn parse_block(&self, text: &str, tokens: &[Token], start_pos: usize) -> Option<(Option<Event>, usize)> {
        let mut pos = start_pos;
        if pos < tokens.len() && tokens[pos].token_type == "START" {
            pos += 1;
        }

        let mut channel_pos = None;
        let mut message_pos = None;

        for i in pos..tokens.len() {
            if tokens[i].token_type == "CHANNEL" && channel_pos.is_none() {
                channel_pos = Some(i);
            } else if tokens[i].token_type == "MESSAGE" {
                message_pos = Some(i);
                break;
            }
        }

        if message_pos.is_none() {
            return None;
        }

        let m_pos = message_pos.unwrap();

        if channel_pos.is_none() {
            let content_start = tokens[m_pos].end;
            let mut end_token_pos = None;
            for i in (m_pos + 1)..tokens.len() {
                if tokens[i].token_type == "END" || tokens[i].token_type == "CALL" || tokens[i].token_type == "RETURN" {
                    end_token_pos = Some(i);
                    break;
                }
            }
            if end_token_pos.is_none() {
                return None;
            }
            let e_pos = end_token_pos.unwrap();
            let content = &text[content_start..tokens[e_pos].start];
            return Some((Some(Event::new("normal".to_string(), content.to_string(), None)), e_pos + 1));
        }

        let c_pos = channel_pos.unwrap();
        pos = c_pos + 1;

        let channel_end = tokens[m_pos].start;
        let channel_header = &text[tokens[pos].start..channel_end];
        let channel_type = Self::extract_channel_type(channel_header);

        let content_start = tokens[m_pos].end;
        let mut end_token_pos = None;
        for i in (m_pos + 1)..tokens.len() {
            if tokens[i].token_type == "END" || tokens[i].token_type == "CALL" || tokens[i].token_type == "RETURN" {
                end_token_pos = Some(i);
                break;
            }
        }

        if end_token_pos.is_none() {
            return None;
        }

        let e_pos = end_token_pos.unwrap();
        let end_token = &tokens[e_pos];
        let content = &text[content_start..end_token.start];

        if let Some(ct) = channel_type {
            if ct == "analysis" {
                Some((Some(Event::new("reasoning".to_string(), content.to_string(), None)), e_pos + 1))
            } else if ct == "commentary" {
                if end_token.token_type == "CALL" {
                    let raw_text = &text[tokens[start_pos].start..end_token.end];
                    Some((Some(Event::new("tool_call".to_string(), content.trim().to_string(), Some(raw_text.to_string()))), e_pos + 1))
                } else {
                    Some((Some(Event::new("normal".to_string(), content.to_string(), None)), e_pos + 1))
                }
            } else if ct == "final" {
                let mut final_content = content.to_string();
                if end_token.token_type == "RETURN" && e_pos + 1 < tokens.len() {
                    let next_token = &tokens[e_pos + 1];
                    if next_token.token_type == "TEXT" {
                        final_content.push_str(&text[next_token.start..next_token.end]);
                        return Some((Some(Event::new("normal".to_string(), final_content, None)), e_pos + 2));
                    }
                }
                Some((Some(Event::new("normal".to_string(), final_content, None)), e_pos + 1))
            } else {
                Some((None, e_pos + 1))
            }
        } else {
            Some((None, e_pos + 1))
        }
    }

    fn is_commentary_filler_between_blocks(text: &str, tokens: &[Token], pos: usize) -> bool {
        let current_token = &tokens[pos];
        let current_text = text[current_token.start..current_token.end].trim().to_lowercase();

        if pos > 0 && pos + 1 < tokens.len() {
            let prev_token = &tokens[pos - 1];
            let next_token = &tokens[pos + 1];

            if prev_token.token_type == "CALL" && next_token.token_type == "CHANNEL" && current_text == "commentary" {
                return true;
            }
        }

        if pos > 0 {
            let prev_token = &tokens[pos - 1];
            if prev_token.token_type == "CALL" {
                if current_token.token_type == "MESSAGE" {
                    return true;
                }
                if current_token.token_type == "TEXT" && current_text == "commentary" {
                    return true;
                }
            }
        }

        false
    }

    fn is_standalone_structural_token(content: &str) -> bool {
        let content_stripped = content.trim();
        let structural_tokens = [
            "<|start|>",
            "<|channel|>",
            "<|message|>",
            "<|constrain|>",
            "<|end|>",
            "<|call|>",
            "<|return|>",
        ];
        structural_tokens.contains(&content_stripped)
    }
}

#[pyclass]
pub struct TextStrategy {
    buffer_context: String,
}

static PATTERN_ANALYSIS_THEN_FINAL: OnceLock<Regex> = OnceLock::new();
static PATTERN_FINAL_ONLY: OnceLock<Regex> = OnceLock::new();
static PATTERN_ANALYSIS_ONLY: OnceLock<Regex> = OnceLock::new();

#[pymethods]
impl TextStrategy {
    #[new]
    fn new() -> Self {
        PATTERN_ANALYSIS_THEN_FINAL.get_or_init(|| Regex::new(r"(?is)^\s*(?:assistant)?\s*(analysis|commentary)(.*?)\s*assistantfinal\s*(.*)\s*$").unwrap());
        PATTERN_FINAL_ONLY.get_or_init(|| Regex::new(r"(?is)^\s*assistantfinal\s*(.*)\s*$").unwrap());
        PATTERN_ANALYSIS_ONLY.get_or_init(|| Regex::new(r"(?is)^\s*(?:assistant)?\s*(analysis|commentary)(.*)\s*$").unwrap());

        TextStrategy {
            buffer_context: String::new(),
        }
    }

    fn set_buffer_context(&mut self, buffer: String) {
        self.buffer_context = buffer;
    }

    fn parse(&self, text: &str) -> PyResult<(Vec<Event>, String)> {
        let mut events = Vec::new();

        if let Some(caps) = PATTERN_ANALYSIS_THEN_FINAL.get().unwrap().captures(text) {
            let channel = caps.get(1).unwrap().as_str().to_lowercase();
            let reasoning = caps.get(2).unwrap().as_str();
            let final_val = caps.get(3).unwrap().as_str();

            if channel == "analysis" && !reasoning.trim().is_empty() {
                events.push(Event::new("reasoning".to_string(), reasoning.trim().to_string(), None));
            } else if channel == "commentary" && !reasoning.trim().is_empty() {
                events.push(Event::new("normal".to_string(), reasoning.trim().to_string(), None));
            }

            if !final_val.trim().is_empty() {
                events.push(Event::new("normal".to_string(), final_val.trim().to_string(), None));
            }
            return Ok((events, String::new()));
        }

        let re_check = Regex::new(r"(?i)(?:^|\s)(?:assistant)?\s*(analysis|commentary)").unwrap();
        if re_check.is_match(text) {
            let low = text.to_lowercase();
            if low.contains("assistantfin") && !low.contains("assistantfinal") {
                return Ok((events, text.to_string()));
            }
        }

        if let Some(caps) = PATTERN_FINAL_ONLY.get().unwrap().captures(text) {
            let final_val = caps.get(1).unwrap().as_str();
            if !final_val.trim().is_empty() {
                events.push(Event::new("normal".to_string(), final_val.trim().to_string(), None));
            }
            return Ok((events, String::new()));
        }

        if let Some(caps) = PATTERN_ANALYSIS_ONLY.get().unwrap().captures(text) {
            let channel = caps.get(1).unwrap().as_str().to_lowercase();
            let content = caps.get(2).unwrap().as_str();
            let (emit, hold) = prefix_hold(content, vec!["assistantfinal".to_string()]);

            let start_idx = caps.get(2).unwrap().start();
            let prefix_text = &text[..start_idx];

            if channel == "analysis" && !emit.is_empty() {
                events.push(Event::new("reasoning".to_string(), emit, None));
                if !hold.is_empty() {
                    return Ok((events, prefix_text.to_string() + &hold));
                } else {
                    return Ok((events, channel));
                }
            } else if channel == "commentary" && !emit.is_empty() {
                let content_out = if !hold.is_empty() { emit } else { emit.trim().to_string() };
                events.push(Event::new("normal".to_string(), content_out, None));
                if !hold.is_empty() {
                    return Ok((events, prefix_text.to_string() + &hold));
                } else {
                    return Ok((events, String::new()));
                }
            }

            return Ok((events, prefix_text.to_string() + &hold));
        }

        let (emit, hold) = prefix_hold(text, vec!["analysis".to_string(), "commentary".to_string(), "assistantfinal".to_string()]);
        if !emit.is_empty() {
            events.push(Event::new("normal".to_string(), emit, None));
        }

        Ok((events, hold))
    }
}

#[pyclass]
pub struct HarmonyParser {
    strategy_type: Option<u8>, // 1: Canonical, 2: Text
    canonical_strategy: CanonicalStrategy,
    text_strategy: TextStrategy,
    #[pyo3(get)]
    pub buffer: String,
    should_filter_commentary: bool,
    partial_commentary: String,
}

#[pymethods]
impl HarmonyParser {
    #[new]
    fn new() -> Self {
        HarmonyParser {
            strategy_type: None,
            canonical_strategy: CanonicalStrategy::new(),
            text_strategy: TextStrategy::new(),
            buffer: String::new(),
            should_filter_commentary: false,
            partial_commentary: String::new(),
        }
    }

    fn parse(&mut self, chunk: &str) -> PyResult<Vec<Event>> {
        self.buffer.push_str(chunk);

        if self.strategy_type.is_none() {
            if self.buffer.contains("<|channel|>") || self.buffer.contains("<|start|>") {
                self.strategy_type = Some(1);
            } else {
                let re = Regex::new(r"(?i)(?:^|\s)(?:assistant)?\s*(analysis|commentary|assistantfinal)").unwrap();
                if re.is_match(&self.buffer) {
                    self.strategy_type = Some(2);
                } else {
                    return Ok(Vec::new());
                }
            }
        }

        let (events, remaining) = match self.strategy_type {
            Some(1) => {
                self.canonical_strategy.parse(&self.buffer)?
            },
            Some(2) => {
                self.text_strategy.set_buffer_context(self.buffer.clone());
                self.text_strategy.parse(&self.buffer)?
            },
            _ => (Vec::new(), self.buffer.clone()),
        };

        let buffer_has_call_token = self.buffer.trim_end().ends_with("<|call|>");
        self.buffer = remaining;

        let mut filtered_events = Vec::new();

        for event in events {
            let mut should_filter = false;

            if event.event_type == "normal" {
                if self.should_filter_commentary || !self.partial_commentary.is_empty() {
                    let potential_commentary = format!("{}{}", self.partial_commentary, event.content.trim().to_lowercase());

                    if potential_commentary == "commentary" {
                        should_filter = true;
                        self.partial_commentary = String::new();
                        self.should_filter_commentary = false;
                    } else if "commentary".starts_with(&potential_commentary) {
                        should_filter = true;
                        self.partial_commentary = potential_commentary;
                    } else {
                        self.partial_commentary = String::new();
                        self.should_filter_commentary = false;
                    }
                } else {
                    self.partial_commentary = String::new();
                }
            }

            if should_filter {
                continue;
            }

            if event.event_type == "tool_call" {
                self.should_filter_commentary = true;
                self.partial_commentary = String::new();
            } else if buffer_has_call_token {
                self.should_filter_commentary = true;
            }

            filtered_events.push(event);
        }

        Ok(filtered_events)
    }
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(prefix_hold, m)?)?;
    m.add_class::<Event>()?;
    m.add_class::<Token>()?;
    m.add_class::<CanonicalStrategy>()?;
    m.add_class::<TextStrategy>()?;
    m.add_class::<HarmonyParser>()?;
    Ok(())
}
