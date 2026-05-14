import sys, importlib.util, time, json

spec = importlib.util.spec_from_file_location("reasoning_parser", "python/sglang/srt/parser/reasoning_parser.py")
reasoning_parser = importlib.util.module_from_spec(spec)
sys.modules["sglang.srt.entrypoints.openai.protocol"] = type("MockOpenAIProtocol", (), {"ChatCompletionRequest": type("ChatCompletionRequest", (), {})})()
sys.modules["sglang.srt.parser.harmony_parser"] = type("MockHarmonyParser", (), {"HarmonyParser": type("HarmonyParser", (), {})})()

spec.loader.exec_module(reasoning_parser)

def run_benchmark():
    detector = reasoning_parser.DeepSeekR1Detector(stream_reasoning=True)
    text_chunks = [
        "First ", "chunk ", "<think>", " I ", "am ", "thinking ", "now. ",
        "Still ", "thinking. ", "</think>", " The ", "final ", "answer ", "is ", "42."
    ]
    # Multiply to simulate high workload
    chunks = text_chunks * 10000

    start_time = time.perf_counter()
    for chunk in chunks:
        res = detector.parse_streaming_increment(chunk)
    end_time = time.perf_counter()

    duration_ms = (end_time - start_time) * 1000

    result = {
      "candidate": "python/sglang/srt/parser/reasoning_parser.py",
      "implementation": "before",
      "command": "python3 test_parse_reasoning_benchmark.py",
      "timestamp": "2023-10-27T00:00:00Z", # Placeholder
      "iterations": len(chunks),
      "input_description": "Streaming chunks simulating DeepSeek-R1 output with <think> tags",
      "duration_ms": duration_ms
    }
    with open("before-benchmark.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Benchmark finished. Time: {duration_ms:.2f} ms")

run_benchmark()
