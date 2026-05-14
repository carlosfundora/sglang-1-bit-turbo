import sys, importlib.util

spec = importlib.util.spec_from_file_location("reasoning_parser", "python/sglang/srt/parser/reasoning_parser.py")
reasoning_parser = importlib.util.module_from_spec(spec)
sys.modules["sglang.srt.entrypoints.openai.protocol"] = type("MockOpenAIProtocol", (), {"ChatCompletionRequest": type("ChatCompletionRequest", (), {})})()
sys.modules["sglang.srt.parser.harmony_parser"] = type("MockHarmonyParser", (), {"HarmonyParser": type("HarmonyParser", (), {})})()

spec.loader.exec_module(reasoning_parser)

detector = reasoning_parser.BaseReasoningFormatDetector("<think>", "</think>")
print("BaseReasoningFormatDetector created")

detector = reasoning_parser.DeepSeekR1Detector()
res = detector.detect_and_parse("hello <think> thinking </think> answer")
print("Reasoning:", res.reasoning_text)
print("Normal:", res.normal_text)
