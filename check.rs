        9 => { // LLAMA4
            let mut ret = String::with_capacity(system_prompt.len() + messages.len() * 100);
            if !system_message.is_empty() {
                ret.push_str(&system_prompt);
            }
            for (i, msg) in messages.iter().enumerate() {
                let role = &msg[0];
                let message = &msg[1];
                ret.push_str("<|start_header_id|>");
                ret.push_str(role);
                ret.push_str("<|end_header_id|>\n\n");
