with open("CHANGELOG.md", "r") as f:
    content = f.read()

content = content.replace("### Added\n", "### Added\n- Support for Qwen 0.6B Embedding and Jina embedding models.\n")

# Documentation section required by the check
if "### Documentation" not in content:
    content = content.replace("### Added\n", "### Documentation\n- None required.\n\n### Added\n")

with open("CHANGELOG.md", "w") as f:
    f.write(content)
