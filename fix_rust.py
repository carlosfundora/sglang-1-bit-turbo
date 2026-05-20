import re

with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    content = f.read()

# I am completely deleting the lib.rs from git restore and injecting the user's implementation
import subprocess
subprocess.check_call(["git", "restore", "python/sglang/rust_utils/src/lib.rs"])
with open("python/sglang/rust_utils/src/lib.rs", "r") as f:
    orig = f.read()

rust_code = """
#[pyclass(eq, hash)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SeparatorStyle {
    AddColonSingle = 1,
    AddColonTwo = 2,
    AddColonSpaceSingle = 3,
    NoColonSingle = 4,
    NoColonTwo = 5,
    AddNewLineSingle = 6,
    Llama2 = 7,
    Llama3 = 8,
    Llama4 = 9,
    ChatGLM = 10,
    ChatML = 11,
    ChatIntern = 12,
    Dolly = 13,
    Rwkv = 14,
    Phoenix = 15,
    Robin = 16,
    FalconChat = 17,
    ChatGLM3 = 18,
    DeepseekChat = 19,
    MetaMath = 20,
    DeepSeekVL2 = 21,
    Qwen2VLEmbed = 22,
    Qwen2Audio = 23,
    Gemma3 = 24,
    MPT = 25,
    PaddleOCR = 26,
    Auto = 99,
}

#[pymethods]
impl SeparatorStyle {
    #[staticmethod]
    fn from_int(value: i32) -> PyResult<Self> {
        match value {
            1 => Ok(Self::AddColonSingle),
            2 => Ok(Self::AddColonTwo),
            3 => Ok(Self::AddColonSpaceSingle),
            4 => Ok(Self::NoColonSingle),
            5 => Ok(Self::NoColonTwo),
            6 => Ok(Self::AddNewLineSingle),
            7 => Ok(Self::Llama2),
            8 => Ok(Self::Llama3),
            9 => Ok(Self::Llama4),
            10 => Ok(Self::ChatGLM),
            11 => Ok(Self::ChatML),
            12 => Ok(Self::ChatIntern),
            13 => Ok(Self::Dolly),
            14 => Ok(Self::Rwkv),
            15 => Ok(Self::Phoenix),
            16 => Ok(Self::Robin),
            17 => Ok(Self::FalconChat),
            18 => Ok(Self::ChatGLM3),
            19 => Ok(Self::DeepseekChat),
            20 => Ok(Self::MetaMath),
            21 => Ok(Self::DeepSeekVL2),
            22 => Ok(Self::Qwen2VLEmbed),
            23 => Ok(Self::Qwen2Audio),
            24 => Ok(Self::Gemma3),
            25 => Ok(Self::MPT),
            26 => Ok(Self::PaddleOCR),
            99 => Ok(Self::Auto),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown SeparatorStyle value: {}", value
            ))),
        }
    }
}

#[pyclass]
pub struct ConversationRust {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub system: Option<String>,
    #[pyo3(get)]
    pub roles: (String, String),
    #[pyo3(get)]
    pub sep_style: SeparatorStyle,
    #[pyo3(get)]
    pub sep: String,
    #[pyo3(get)]
    pub sep2: Option<String>,
    #[pyo3(get)]
    pub messages: Vec<(String, String)>,
    #[pyo3(get)]
    pub image_token: Option<String>,
}

#[pymethods]
impl ConversationRust {
    #[new]
    #[pyo3(signature = (name, system, roles, sep_style_int, sep, sep2, messages, image_token))]
    fn new(
        name: String,
        system: Option<String>,
        roles: (String, String),
        sep_style_int: i32,
        sep: String,
        sep2: Option<String>,
        messages: Vec<(String, String)>,
        image_token: Option<String>,
    ) -> PyResult<Self> {
        let sep_style = SeparatorStyle::from_int(sep_style_int)?;
        Ok(ConversationRust {
            name,
            system,
            roles,
            sep_style,
            sep,
            sep2,
            messages,
            image_token,
        })
    }

    fn get_prompt_rust(&self) -> String {
        let mut prompt = String::new();

        // System prompt
        if let Some(sys) = &self.system {
            if !sys.is_empty() {
                match self.sep_style {
                    SeparatorStyle::Llama4 => {
                        prompt.push_str("<|header_start|>system<|header_end|>\n");
                        prompt.push_str(sys);
                        prompt.push_str("\n");
                    }
                    SeparatorStyle::ChatGLM | SeparatorStyle::ChatGLM3 => {
                        prompt.push_str(sys);
                        prompt.push_str("\n");
                    }
                    SeparatorStyle::ChatML => {
                        prompt.push_str(sys);
                        prompt.push_str(&self.sep);
                        prompt.push_str("\n");
                    }
                    SeparatorStyle::DeepSeekVL2 => {
                        prompt.push_str(sys);
                        prompt.push_str(&self.sep);
                    }
                    SeparatorStyle::Qwen2VLEmbed => {
                        prompt.push_str(sys);
                        prompt.push_str(&self.sep);
                    }
                    SeparatorStyle::Qwen2Audio => {
                        prompt.push_str(sys);
                        prompt.push_str(&self.sep);
                    }
                    SeparatorStyle::MetaMath => {
                        prompt.push_str(sys);
                        prompt.push_str(&self.sep);
                    }
                    SeparatorStyle::AddNewLineSingle => {
                        prompt.push_str(sys);
                        prompt.push_str(&self.sep);
                    }
                    SeparatorStyle::AddColonSingle | SeparatorStyle::AddColonSpaceSingle | SeparatorStyle::AddColonTwo | SeparatorStyle::Robin | SeparatorStyle::MPT => {
                        prompt.push_str(sys);
                        prompt.push_str(&self.sep);
                    }
                    _ => {
                        prompt.push_str(sys);
                    }
                }
            }
        } else if self.sep_style == SeparatorStyle::DeepSeekVL2 {
            prompt.push_str("<|bos|>");
        }

        // Messages
        for (i, (role, content)) in self.messages.iter().enumerate() {
            match self.sep_style {
                // === LLAMA FAMILY ===
                SeparatorStyle::Llama4 => {
                    let role_tag = if role.to_lowercase().contains("user") { "user" } else { "assistant" };
                    prompt.push_str(&format!("<|header_start|>{}<|header_end|>\n\n", role_tag));
                    if !content.is_empty() {
                        prompt.push_str(content.trim());
                        prompt.push_str("<|eot|>");
                    }
                }
                SeparatorStyle::Llama3 => {
                    prompt.push_str(&format!("<|start_header_id|>{}<|end_header_id|>\n\n", role));
                    if !content.is_empty() {
                        prompt.push_str(content.trim());
                        prompt.push_str("<|eot_id|>");
                    }
                }
                SeparatorStyle::Llama2 => {
                    if !content.is_empty() {
                        if i == 0 {
                            if self.system.is_some() && !self.system.as_ref().unwrap().is_empty() {
                                prompt.push_str(content);
                            } else {
                                prompt.push_str("[INST] ");
                                prompt.push_str(content);
                            }
                            prompt.push_str(" [/INST]");
                        } else if i % 2 == 0 {
                            prompt.push_str("<s>[INST] ");
                            prompt.push_str(content);
                            prompt.push_str(" [/INST]");
                        } else {
                            prompt.push_str(" ");
                            prompt.push_str(content);
                            prompt.push_str(" </s>");
                        }
                    } else if i % 2 == 0 {
                        if i == 0 && (self.system.is_none() || self.system.as_ref().unwrap().is_empty()) {
                            prompt.push_str("[INST] ");
                        } else {
                            prompt.push_str("<s>[INST] ");
                        }
                    } else {
                        prompt.push_str(" ");
                    }
                }

                // === CHATGLM FAMILY ===
                SeparatorStyle::ChatGLM => {
                    if role == "问：" || role.to_lowercase().contains("user") {
                        prompt.push_str(&format!("[Round {}]\n\n问：{}\n\n答：",
                            i / 2 + 1, content));
                    } else {
                        prompt.push_str(content);
                        if !content.is_empty() {
                            prompt.push_str("\n\n");
                        }
                    }
                }
                SeparatorStyle::ChatGLM3 => {
                    prompt.push_str(&format!("<|{}|>\n{}", role, content));
                }

                // === RWKV ===
                SeparatorStyle::Rwkv => {
                    let sep_to_use = if i % 2 == 0 { &self.sep } else { self.sep2.as_ref().unwrap_or(&"".to_string()) };
                    let c = content.replace("\r\n", "\n").replace("\n\n", "\n");
                    prompt.push_str(&format!("{}: {}{}", role, c, sep_to_use));
                }

                // === DOLLY ===
                SeparatorStyle::Dolly => {
                    prompt.push_str(&format!("{}:\n", role));
                    if !content.is_empty() {
                        prompt.push_str(content);
                        prompt.push_str("\n\n");
                        if i % 2 == 0 {
                            prompt.push_str(&self.sep);
                        } else {
                            if let Some(sep2) = &self.sep2 {
                                prompt.push_str(sep2);
                            }
                        }
                    }
                }

                SeparatorStyle::AddColonSingle => {
                    prompt.push_str(role);
                    prompt.push_str(":");
                    if !content.is_empty() {
                        prompt.push_str(" ");
                        prompt.push_str(content);
                        prompt.push_str(&self.sep);
                    }
                }

                SeparatorStyle::AddColonTwo => {
                    prompt.push_str(role);
                    prompt.push_str(":");
                    if !content.is_empty() {
                        prompt.push_str(" ");
                        prompt.push_str(content);
                        if i % 2 == 0 {
                            prompt.push_str(&self.sep);
                        } else {
                            if let Some(sep2) = &self.sep2 {
                                prompt.push_str(sep2);
                            }
                        }
                    }
                }

                SeparatorStyle::AddColonSpaceSingle => {
                    prompt.push_str(role);
                    prompt.push_str(":");
                    if !content.is_empty() {
                        prompt.push_str(" ");
                        prompt.push_str(content);
                        prompt.push_str(&self.sep);
                    } else {
                        prompt.push_str(" ");
                    }
                }

                SeparatorStyle::NoColonSingle => {
                    prompt.push_str(role);
                    if !content.is_empty() {
                        prompt.push_str(content);
                        prompt.push_str(&self.sep);
                    }
                }

                SeparatorStyle::NoColonTwo => {
                    prompt.push_str(role);
                    if !content.is_empty() {
                        prompt.push_str(content);
                        if i % 2 == 0 {
                            prompt.push_str(&self.sep);
                        } else {
                            if let Some(sep2) = &self.sep2 {
                                prompt.push_str(sep2);
                            }
                        }
                    }
                }

                SeparatorStyle::AddNewLineSingle => {
                    prompt.push_str(role);
                    prompt.push_str("\n");
                    if !content.is_empty() {
                        prompt.push_str(content);
                        prompt.push_str(&self.sep);
                    }
                }

                SeparatorStyle::ChatML => {
                    prompt.push_str(role);
                    prompt.push_str("\n");
                    if !content.is_empty() {
                        prompt.push_str(content);
                        prompt.push_str(&self.sep);
                        prompt.push_str("\n");
                    }
                }

                SeparatorStyle::ChatIntern => {
                    if i % 2 == 0 {
                        prompt.push_str(&format!("<|User|>:{}", content));
                        prompt.push_str(&self.sep);
                    } else {
                        prompt.push_str(&format!("<|Bot|>:{}", content));
                        if let Some(sep2) = &self.sep2 {
                            prompt.push_str(sep2);
                        }
                    }
                }

                SeparatorStyle::Phoenix => {
                    prompt.push_str(&format!("{}: <s>{}</s>", role, content));
                }

                SeparatorStyle::Robin => {
                    prompt.push_str(&format!("{}:\n{}", role, content));
                    prompt.push_str(&self.sep);
                }

                SeparatorStyle::FalconChat => {
                    if !content.is_empty() {
                        prompt.push_str(&format!("{}: {}{}", role, content, self.sep));
                    } else {
                        prompt.push_str(&format!("{}:", role));
                    }
                }

                SeparatorStyle::MetaMath => {
                    prompt.push_str(&format!("{}\n", role));
                    if !content.is_empty() {
                        prompt.push_str(content);
                        prompt.push_str(&self.sep);
                    }
                }

                SeparatorStyle::DeepseekChat => {
                    prompt.push_str(role);
                    prompt.push_str(":");
                    if !content.is_empty() {
                        prompt.push_str(" ");
                        prompt.push_str(content);
                        if i % 2 == 0 {
                            prompt.push_str(&self.sep);
                        } else {
                            if let Some(sep2) = &self.sep2 {
                                prompt.push_str(sep2);
                            }
                        }
                    }
                }

                SeparatorStyle::DeepSeekVL2 => {
                    prompt.push_str(role);
                    prompt.push_str(": ");
                    if !content.is_empty() {
                        if i % 2 == 0 {
                            let c = if let Some(it) = &self.image_token {
                                content.replace(&format!("{}\n", it), it)
                            } else {
                                content.to_string()
                            };
                            prompt.push_str(&c);
                            prompt.push_str(&self.sep);
                        } else {
                            prompt.push_str(content);
                            if let Some(sep2) = &self.sep2 {
                                prompt.push_str(sep2);
                            }
                        }
                    }
                }

                SeparatorStyle::Qwen2VLEmbed => {
                    prompt.push_str(role);
                    prompt.push_str("\n");
                    if !content.is_empty() {
                        prompt.push_str(content);
                        prompt.push_str(&self.sep);
                    }
                }

                SeparatorStyle::Qwen2Audio => {
                    prompt.push_str(role);
                    prompt.push_str("\n");
                    if !content.is_empty() {
                        prompt.push_str(content);
                        prompt.push_str(&self.sep);
                    }
                }

                SeparatorStyle::Gemma3 => {
                    prompt.push_str("<start_of_turn>");
                    prompt.push_str(role);
                    prompt.push_str("\n");
                    if !content.is_empty() {
                        prompt.push_str(content);
                        prompt.push_str("<end_of_turn>\n");
                    } else if i % 2 == 1 {
                        prompt.push_str("<end_of_turn>\n");
                    }
                }

                SeparatorStyle::MPT => {
                    prompt.push_str(role);
                    if !content.is_empty() {
                        prompt.push_str(content);
                        prompt.push_str(&self.sep);
                    }
                }

                SeparatorStyle::PaddleOCR => {
                    prompt.push_str(&format!("{}: ", role));
                    if role == &self.roles.0 {
                        let c = if let Some(it) = &self.image_token {
                            content.replace(&format!("{}\n", it), it)
                        } else {
                            content.to_string()
                        };
                        prompt.push_str(&c);
                        prompt.push_str("\n");
                    } else {
                        prompt.push_str(content);
                        prompt.push_str(&self.sep);
                    }
                }

                SeparatorStyle::Auto => {
                    // Python will fallback before calling us
                }
            }
        }

        // Final assistant prompt suffix
        match self.sep_style {
            SeparatorStyle::Llama4 => {
                prompt.push_str("<|header_start|>assistant<|header_end|>\n\n");
            }
            SeparatorStyle::Llama3 | SeparatorStyle::Llama2 => {
                prompt.push_str("[/INST] ");
            }
            _ => {}
        }

        // Image token replacement (unless handled specifically above)
        if self.sep_style != SeparatorStyle::DeepSeekVL2 && self.sep_style != SeparatorStyle::PaddleOCR {
            if let Some(img_token) = &self.image_token {
                prompt = prompt.replace("<image>", img_token);
            }
        }

        prompt
    }
}
"""

orig = orig.replace("m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;",
"m.add_function(wrap_pyfunction!(process_content_for_template_format, m)?)?;\n    m.add_class::<SeparatorStyle>()?;\n    m.add_class::<ConversationRust>()?;")

with open("python/sglang/rust_utils/src/lib.rs", "w") as f:
    f.write(orig + "\n\n" + rust_code)
