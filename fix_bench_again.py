with open("test_final_bench.py", "r") as f:
    content = f.read()

content = content.replace("cache.match_prefix(MatchPrefixParams(key))", "cache.match_prefix(MatchPrefixParams(key))")
content = content.replace("pass", "cache.match_prefix(MatchPrefixParams(key))")

with open("test_final_bench.py", "w") as f:
    f.write(content)
