"""
Content extraction utilities for PR comments and reviews.

Ported from extract_coderabbit_prompts.py with improvements for structured data extraction.
"""
from __future__ import annotations

import re
from typing import Any


class ExtractedContent:
    """Represents extracted content from a comment."""

    def __init__(self, content_type: str, content: str, metadata: dict[str, Any] | None = None):
        self.content_type = content_type
        self.content = content
        self.metadata = metadata or {}


def parse_file_reference_from_body(body: str) -> str | None:
    """Parse file and line references from comment body text."""
    if not body:
        return None

    # Pattern 1: HTML summary tags "<summary>crawler_mcp/path/file.py (2)</summary>"
    html_summary_pattern = r"<summary>([^/\s]+/[^/\s]+\.py)"
    match = re.search(html_summary_pattern, body)
    if match:
        # Look for line numbers in backticks nearby
        line_backtick_pattern = r"`(\d+(?:-\d+)?)`:"
        line_match = re.search(line_backtick_pattern, body)
        if line_match:
            return f"{match.group(1)}:{line_match.group(1)}"
        return match.group(1)

    # Pattern 2: "In crawler_mcp/path/file.py around lines X-Y" or "around line X"
    file_in_text_pattern = (
        r"In ([^\s]+\.py) around lines? (\d+(?:-\d+)?|\d+ to \d+)"
    )
    match = re.search(file_in_text_pattern, body)
    if match:
        return f"{match.group(1)}:{match.group(2)}"

    # Pattern 3: "In file.py around lines X-Y" (just filename without path)
    file_simple_pattern = r"In ([^/\s]+\.py) around lines? (\d+(?:-\d+)?|\d+ to \d+)"
    match = re.search(file_simple_pattern, body)
    if match:
        return f"{match.group(1)}:{match.group(2)}"

    # Pattern 4: Line number in backticks "`445-447`:" at start or after newline
    line_backtick_pattern = r"`(\d+(?:-\d+)?)`:"
    match = re.search(line_backtick_pattern, body)
    if match:
        return f"line:{match.group(1)}"

    # Pattern 5: Line number prefix at start of body "414-447:" or "414:"
    line_prefix_pattern = r"^(\d+(?:-\d+)?):"
    match = re.match(line_prefix_pattern, body.strip())
    if match:
        return f"line:{match.group(1)}"

    # Pattern 6: Just file path mention "In crawler_mcp/path/file.py" without line numbers
    file_only_pattern = r"In ([^/\s]+/[^/\s]+\.py)"
    match = re.search(file_only_pattern, body)
    if match:
        return match.group(1)

    return None


def extract_ai_prompts(body: str, user_login: str) -> list[ExtractedContent]:
    """Extract AI prompts from CodeRabbitAI bot comments."""
    if "🤖 Prompt for AI Agents" not in body or user_login != "coderabbitai[bot]":
        return []

    extracted = []

    # Extract all details blocks that contain the prompt header
    details_pattern = r"<details>\s*<summary>🤖 Prompt for AI Agents</summary>\s*(.*?)\s*</details>"
    details_matches = re.findall(details_pattern, body, re.DOTALL)

    for details_content in details_matches:
        # Extract content between triple backticks
        code_pattern = r"```\s*(.*?)\s*```"
        code_matches = re.findall(code_pattern, details_content, re.DOTALL)

        for code_content in code_matches:
            cleaned_prompt = code_content.strip()
            if cleaned_prompt:
                extracted.append(ExtractedContent(
                    content_type="AI_PROMPT",
                    content=cleaned_prompt,
                    metadata={"source": "coderabbitai[bot]"}
                ))

    return extracted


def extract_committable_suggestions(body: str, user_login: str) -> list[ExtractedContent]:
    """Extract committable suggestions from any user."""
    if "📝 Committable suggestion" not in body:
        return []

    extracted = []

    # Extract the suggestion content from the suggestion block
    suggestion_pattern = r"```suggestion\s*(.*?)\s*```"
    suggestion_matches = re.findall(suggestion_pattern, body, re.DOTALL)

    for suggestion_content in suggestion_matches:
        cleaned_suggestion = suggestion_content.strip()
        if cleaned_suggestion:
            extracted.append(ExtractedContent(
                content_type="COMMITTABLE_SUGGESTION",
                content=cleaned_suggestion,
                metadata={"source": user_login}
            ))

    return extracted


def extract_copilot_suggestions(body: str, user_login: str) -> list[ExtractedContent]:
    """Extract Copilot suggestions and reviews."""
    extracted = []

    # Check for Copilot suggestion blocks
    if user_login in ["Copilot", "copilot-pull-request-reviewer[bot]"] and "```suggestion" in body:
        suggestion_pattern = r"```suggestion\s*(.*?)\s*```"
        suggestion_matches = re.findall(suggestion_pattern, body, re.DOTALL)

        for suggestion_content in suggestion_matches:
            cleaned_suggestion = suggestion_content.strip()
            if cleaned_suggestion:
                extracted.append(ExtractedContent(
                    content_type="COPILOT_SUGGESTION",
                    content=cleaned_suggestion,
                    metadata={"source": user_login}
                ))

    # Extract Copilot review overviews (without suggestions)
    elif (user_login == "copilot-pull-request-reviewer[bot]"
          and body and "```suggestion" not in body):
        # Extract meaningful review content (skip generic footers)
        if len(body.strip()) > 50 and not body.strip().startswith("---"):
            # Clean up the review content
            cleaned_review = body.strip()
            # Remove the footer tip section
            if "**Tip:** Customize your code reviews" in cleaned_review:
                cleaned_review = cleaned_review.split("---")[0].strip()

            if cleaned_review:
                extracted.append(ExtractedContent(
                    content_type="COPILOT_REVIEW",
                    content=cleaned_review,
                    metadata={"source": user_login}
                ))

    return extracted


def extract_code_blocks(body: str, user_login: str) -> list[ExtractedContent]:
    """Extract various code blocks from comments."""
    # Only extract if we haven't already processed this comment for AI prompts or suggestions
    if ("🤖 Prompt for AI Agents" in body or
        "📝 Committable suggestion" in body or
        (user_login in ["Copilot", "copilot-pull-request-reviewer[bot]"] and "```suggestion" in body)):
        return []

    extracted = []

    # Look for code blocks with various languages
    code_block_patterns = [
        (r"```diff\s*(.*?)\s*```", "DIFF"),
        (r"```patch\s*(.*?)\s*```", "PATCH"),
        (r"```python\s*(.*?)\s*```", "PYTHON"),
        (r"```javascript\s*(.*?)\s*```", "JAVASCRIPT"),
        (r"```typescript\s*(.*?)\s*```", "TYPESCRIPT"),
        (r"```json\s*(.*?)\s*```", "JSON"),
        (r"```yaml\s*(.*?)\s*```", "YAML"),
        (r"```sql\s*(.*?)\s*```", "SQL"),
        (r"```shell\s*(.*?)\s*```", "SHELL"),
        (r"```bash\s*(.*?)\s*```", "BASH"),
    ]

    for pattern, lang_type in code_block_patterns:
        matches = re.findall(pattern, body, re.DOTALL)
        for match in matches:
            cleaned_code = match.strip()
            if cleaned_code and len(cleaned_code) > 20:  # Only meaningful code blocks
                extracted.append(ExtractedContent(
                    content_type=f"{lang_type}_BLOCK",
                    content=cleaned_code,
                    metadata={"source": user_login, "language": lang_type.lower()}
                ))

    return extracted


def extract_all_content(body: str, user_login: str) -> list[ExtractedContent]:
    """Extract all relevant content from a comment body."""
    if not body or not body.strip():
        return []

    extracted = []

    # Extract in order of priority
    extracted.extend(extract_ai_prompts(body, user_login))
    extracted.extend(extract_committable_suggestions(body, user_login))
    extracted.extend(extract_copilot_suggestions(body, user_login))
    extracted.extend(extract_code_blocks(body, user_login))

    return extracted


def is_content_relevant(body: str, user_login: str) -> bool:
    """Check if comment contains relevant content worth extracting."""
    if not body:
        return False

    # Check for bot authors with typical patterns
    bot_patterns = ["coderabbitai[bot]", "copilot-pull-request-reviewer[bot]", "Copilot"]
    if any(bot in user_login for bot in bot_patterns):
        return True

    # Check for specific content markers
    content_markers = [
        "```",  # Any code block
        "📝 Committable suggestion",
        "🤖 Prompt for AI Agents",
    ]

    return any(marker in body for marker in content_markers)


def get_content_summary(extracted_content: list[ExtractedContent]) -> str:
    """Get a summary of extracted content for logging."""
    if not extracted_content:
        return "No content extracted"

    type_counts = {}
    for content in extracted_content:
        type_counts[content.content_type] = type_counts.get(content.content_type, 0) + 1

    parts = [f"{count} {content_type}" for content_type, count in type_counts.items()]
    return f"Extracted: {', '.join(parts)}"
