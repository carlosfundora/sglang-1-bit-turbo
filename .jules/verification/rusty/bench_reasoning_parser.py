import time
import json
import sys
import importlib.util

# Use importlib to bypass loading __init__ and heavy modules
spec = importlib.util.spec_from_file_location("harmony_parser", "python/sglang/srt/parser/harmony_parser.py")
harmony_parser = importlib.util.module_from_spec(spec)
sys.modules['sglang.srt.parser.harmony_parser'] = harmony_parser
spec.loader.exec_module(harmony_parser)

sys.modules['sglang.srt.entrypoints.openai.protocol'] = type('protocol', (), {'ChatCompletionRequest': type('ChatCompletionRequest', (), {})})

spec = importlib.util.spec_from_file_location("reasoning_parser", "python/sglang/srt/parser/reasoning_parser.py")
reasoning_parser = importlib.util.module_from_spec(spec)
sys.modules['sglang.srt.parser.reasoning_parser'] = reasoning_parser
spec.loader.exec_module(reasoning_parser)

def run_benchmark():
    parser = reasoning_parser.ReasoningParser("deepseek-r1", stream_reasoning=True)

    text = "<think>\n"
    text += "This is a long reasoning process " * 200
    text += "\n</think>\n"
    text += "This is the final answer " * 200

    chunk_size = 8
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    start_time = time.time()
    iterations = 1000

    for _ in range(iterations):
        parser = reasoning_parser.ReasoningParser("deepseek-r1", stream_reasoning=True)
        for chunk in chunks:
            parser.parse_stream_chunk(chunk)

    end_time = time.time()

    duration_ms = (end_time - start_time) * 1000

    result = {
        "candidate": "python/sglang/srt/parser/reasoning_parser.py",
        "implementation": "before",
        "command": "python3 .jules/verification/rusty/bench_reasoning_parser.py",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iterations": iterations,
        "input_description": f"Streaming text of length {len(text)} in chunks of size {chunk_size}",
        "duration_ms": duration_ms,
        "throughput": f"{iterations * len(text) / (end_time - start_time):.2f} chars/sec"
    }

    with open(".jules/verification/rusty/before-benchmark.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_benchmark()
