import re

with open("python/sglang/srt/entrypoints/openai/serving_chat.py", "r") as f:
    content = f.read()

# Replace import
content = content.replace("from jsonschema import Draft202012Validator, SchemaError\n", "from jsonschema import Draft202012Validator, SchemaError\n\ntry:\n    from sglang.sglang_rust_utils import is_valid_json_schema as _rust_is_valid_json_schema\nexcept ImportError:\n    _rust_is_valid_json_schema = None\n")

# Replace validation call
old_call = """            try:
                Draft202012Validator.check_schema(tool.function.parameters)
            except SchemaError as e:
                return f"Tool {i} function has invalid 'parameters' schema: {str(e)}\""""

new_call = """            if _rust_is_valid_json_schema is not None:
                import json
                try:
                    schema_str = json.dumps(tool.function.parameters)
                    _rust_is_valid_json_schema(schema_str)
                except ValueError as e:
                    return f"Tool {i} function has invalid 'parameters' schema: {str(e)}"
            else:
                try:
                    Draft202012Validator.check_schema(tool.function.parameters)
                except SchemaError as e:
                    return f"Tool {i} function has invalid 'parameters' schema: {str(e)}\""""

content = content.replace(old_call, new_call)

with open("python/sglang/srt/entrypoints/openai/serving_chat.py", "w") as f:
    f.write(content)
