use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const DSML_TOKEN: &str = "｜DSML｜";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
const THINKING_END_TOKEN: &str = "<｜思绪▁结束｜>";

fn read_until_stop<'a>(
    index: usize,
    text: &'a str,
    stops: &[&'a str],
) -> (usize, &'a str, Option<&'a str>) {
    let mut min_pos = text.len();
    let mut matched_stop = None;

    for &s in stops {
        if let Some(pos) = text[index..].find(s) {
            let absolute_pos = index + pos;
            if absolute_pos < min_pos {
                min_pos = absolute_pos;
                matched_stop = Some(s);
            }
        }
    }

    if let Some(s) = matched_stop {
        let content = &text[index..min_pos];
        (min_pos + s.len(), content, Some(s))
    } else {
        let content = &text[index..];
        (text.len(), content, None)
    }
}

fn escape_json_string(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len() + 2);
    escaped.push('"');
    for c in s.chars() {
        match c {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            '\u{08}' => escaped.push_str("\\b"),
            '\u{0C}' => escaped.push_str("\\f"),
            _ => {
                if c.is_control() {
                    escaped.push_str(&format!("\\u{:04x}", c as u32));
                } else {
                    escaped.push(c);
                }
            }
        }
    }
    escaped.push('"');
    escaped
}

fn to_json(val: &str) -> String {
    escape_json_string(val)
}

fn decode_dsml_to_arguments(
    tool_name: &str,
    tool_args: &[(String, String, bool)],
) -> Result<(String, String), String> {
    let mut args_json = String::new();
    args_json.push('{');
    for (i, (key, value, is_str)) in tool_args.iter().enumerate() {
        if i > 0 {
            args_json.push_str(", ");
        }
        let encoded_val = if *is_str {
            to_json(value)
        } else {
            value.clone()
        };
        args_json.push_str(&format!("{}: {}", to_json(key), encoded_val));
    }
    args_json.push('}');

    Ok((tool_name.to_string(), args_json))
}

pub fn parse_tool_calls(
    mut index: usize,
    text: &str,
) -> Result<(usize, Option<String>, Vec<(String, String)>), String> {
    let mut tool_calls = Vec::new();
    let tool_calls_end_token = format!("</{}function_calls>", DSML_TOKEN);
    let invoke_start_token = format!("<{}invoke", DSML_TOKEN);
    let invoke_end_token = format!("</{}invoke>", DSML_TOKEN);
    let param_start_token = format!("<{}parameter", DSML_TOKEN);
    let param_end_token_python = format!("/{}parameter", DSML_TOKEN);

    let mut stop_token_str = None;

    while index < text.len() {
        let (new_index, content, stop_token) = read_until_stop(
            index,
            text,
            &[&invoke_start_token, &tool_calls_end_token],
        );
        index = new_index;

        if stop_token == Some(&tool_calls_end_token as &str) {
            stop_token_str = Some(tool_calls_end_token.clone());
            break;
        }

        if stop_token.is_none() {
            return Err("Missing special token".to_string());
        }

        if content != ">\n" && content != "\n" {
             return Err(format!("Tool call format error: invoke tag not followed by >\\n, got {:?}", content));
        }

        let (new_index, tool_name_content, _stop_token) = read_until_stop(
            index,
            text,
            &[&param_start_token, &invoke_end_token],
        );
        index = new_index;

        let tool_name = if let Some(start) = tool_name_content.find("name=\"") {
            let start = start + 6;
            if let Some(end) = tool_name_content[start..].find("\">\n") {
                &tool_name_content[start..start+end]
            } else {
                return Err("Tool name format error".to_string());
            }
        } else {
             return Err("Tool name format error".to_string());
        };

        let mut tool_args = Vec::new();
        let mut current_stop = _stop_token;

        while current_stop == Some(&param_start_token as &str) {
            let (new_index, param_content, _next_stop) = read_until_stop(
                index,
                text,
                &[&param_end_token_python],
            );
            index = new_index;

            let name_start_tag = " name=\"";
            let string_tag = "\" string=\"";

            let name_start = match param_content.find(name_start_tag) {
                Some(p) => p + name_start_tag.len(),
                None => return Err(format!("Parameter format error: name_start_tag not found in {:?}", param_content)),
            };
            let name_end = match param_content[name_start..].find(string_tag) {
                Some(p) => p + name_start,
                None => return Err(format!("Parameter format error: string_tag not found in {:?}", param_content)),
            };
            let param_name = &param_content[name_start..name_end];

            let string_start = name_end + string_tag.len();
            let string_end = match param_content[string_start..].find("\">") {
                Some(p) => p + string_start,
                None => return Err(format!("Parameter format error: end tag not found in {:?}", param_content)),
            };
            let is_str_str = &param_content[string_start..string_end];

            let value_start = string_end + 2;
            let value_end = param_content.len().saturating_sub(1);
            let param_value = if param_content.ends_with("<") {
                &param_content[value_start..value_end]
            } else {
                return Err(format!("Parameter format error: does not end with `<` in {:?}", param_content));
            };

            let is_str = match is_str_str {
                "true" => true,
                "false" => false,
                _ => return Err("Parameter format error: is_str mismatch".to_string()),
            };

            tool_args.push((param_name.to_string(), param_value.to_string(), is_str));

            let (new_index, content, next_stop2) = read_until_stop(
                index,
                text,
                &[&param_start_token, &invoke_end_token],
            );
            index = new_index;
            if content != ">\n" {
                return Err(format!("Parameter format error: next param tag mismatch, got {:?}", content));
            }
            current_stop = next_stop2;
        }

        let tool_call = decode_dsml_to_arguments(tool_name, &tool_args)?;
        tool_calls.push(tool_call);
    }

    Ok((index, stop_token_str, tool_calls))
}

#[pyfunction]
pub fn parse_message_from_completion_text<'py>(
    py: Python<'py>,
    text: &str,
    thinking_mode: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let is_thinking = thinking_mode == "thinking";
    let mut is_tool_calling = false;
    let mut index = 0;

    let tool_calls_start_token = format!("\n\n<{}function_calls", DSML_TOKEN);
    let tool_calls_start_token2 = format!("\n<{}function_calls", DSML_TOKEN);
    let tool_calls_start_token3 = format!("<{}function_calls", DSML_TOKEN);

    let mut reasoning_content = "";
    let mut summary_content = "";

    if is_thinking {
        let stops = [THINKING_END_TOKEN, tool_calls_start_token.as_str(), tool_calls_start_token2.as_str(), tool_calls_start_token3.as_str()];
        let (new_index, content_delta, stop_token) = read_until_stop(index, text, &stops);
        index = new_index;
        reasoning_content = content_delta;

        if stop_token != Some(THINKING_END_TOKEN) {
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid thinking format"));
        }
    }

    let stops = [EOS_TOKEN, tool_calls_start_token.as_str(), tool_calls_start_token2.as_str(), tool_calls_start_token3.as_str()];
    let (new_index, content_delta, stop_token) = read_until_stop(index, text, &stops);
    index = new_index;
    summary_content = content_delta;

    if stop_token == Some(tool_calls_start_token.as_str()) || stop_token == Some(tool_calls_start_token2.as_str()) || stop_token == Some(tool_calls_start_token3.as_str()) {
        is_tool_calling = true;
    } else if stop_token != Some(EOS_TOKEN) {
        return Err(pyo3::exceptions::PyValueError::new_err("Invalid summary format"));
    }

    let mut tool_calls = Vec::new();
    let mut final_stop_token_str: Option<String> = stop_token.map(|s| s.to_string());

    if is_tool_calling {
        match parse_tool_calls(index, text) {
            Ok((new_index, _stop, calls)) => {
                index = new_index;
                tool_calls = calls;
                let (next_index, tool_ends_text, next_stop) = read_until_stop(index, text, &[EOS_TOKEN]);
                index = next_index;
                if !tool_ends_text.is_empty() && tool_ends_text != "\n" {
                    return Err(pyo3::exceptions::PyValueError::new_err("Unexpected content after tool calls"));
                }
                final_stop_token_str = next_stop.map(|s| s.to_string());
            }
            Err(e) => return Err(pyo3::exceptions::PyValueError::new_err(e)),
        }
    }

    if index != text.len() || (final_stop_token_str.as_deref() != Some(EOS_TOKEN) && final_stop_token_str.is_some()) {
        return Err(pyo3::exceptions::PyValueError::new_err("Unexpected content at end"));
    }

    let bad_tokens = [
        "<｜begin▁of▁sentence｜>",
        EOS_TOKEN,
        "<｜思绪▁开始｜>",
        THINKING_END_TOKEN,
        DSML_TOKEN,
    ];

    for &sp_token in &bad_tokens {
        if summary_content.contains(sp_token) || reasoning_content.contains(sp_token) {
            return Err(pyo3::exceptions::PyValueError::new_err("Unexpected special token in content"));
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("role", "assistant")?;
    dict.set_item("content", summary_content)?;
    dict.set_item("reasoning_content", reasoning_content)?;

    let py_tool_calls = PyList::empty(py);
    for (name, arguments) in tool_calls {
        let tc_dict = PyDict::new(py);
        tc_dict.set_item("name", name)?;
        tc_dict.set_item("arguments", arguments)?;
        py_tool_calls.append(tc_dict)?;
    }
    dict.set_item("tool_calls", py_tool_calls)?;

    Ok(dict)
}
