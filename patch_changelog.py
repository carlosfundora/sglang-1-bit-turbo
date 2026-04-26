with open("CHANGELOG.md", "r") as f:
    content = f.read()

# Add the new file to documentation list
new_content = content.replace("- None required\n", "- Added: `.jules/journals/rocmancer.md`\n- None required\n")

with open("CHANGELOG.md", "w") as f:
    f.write(new_content)
