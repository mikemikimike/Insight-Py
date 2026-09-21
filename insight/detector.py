import google.generativeai as genai
import os
import re
import sys
from rich.console import Console
from rich.panel import Panel

console = Console()

AI_SCORE_LINE_PATTERN = re.compile(
    r"^[ \t]*AI_SCORE[ \t]*:[ \t]*(?P<value>[^\r\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)
VALID_SCORE_PATTERN = re.compile(r"^(10|[1-9])(?:[ \t]*/[ \t]*10)?$")
LABELED_SCORE_PATTERN = re.compile(
    r"^[ \t]*(?:Score|Probability)[ \t]*:[ \t]*"
    r"(?P<score>10|[1-9])(?:[ \t]*/[ \t]*10)?(?:[ \t]+.*)?$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_ai_score(text: str) -> int:
    """Extract a labeled AI score, falling back to the neutral default."""
    ai_score_lines = list(AI_SCORE_LINE_PATTERN.finditer(text))
    if ai_score_lines:
        value = ai_score_lines[-1].group("value").strip()
        match = VALID_SCORE_PATTERN.fullmatch(value)
        return int(match.group(1)) if match else 5

    match = LABELED_SCORE_PATTERN.search(text)
    return int(match.group("score")) if match else 5


def show_error(message):
    console.print(
        Panel.fit(
            f"[bold red]ERROR[/bold red]\n\n{message}",
            border_style="red",
            title="System Message",
            title_align="left",
        )
    )


# Check if we're running in a test environment
is_testing = (
    "PYTEST_CURRENT_TEST" in os.environ
    or "pytest" in sys.modules
    or any("pytest" in arg for arg in sys.argv)
)

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key and not is_testing:
    error_message = """Error: Missing Google API Key / Application Issue
It looks like the application encountered an issue, possibly due to a missing Google API key.
To fix a missing API key, set your key using one of the following methods:
   • On macOS/Linux:
       export GOOGLE_API_KEY='your_key_here'
   • On Windows (PowerShell or CMD):
       setx GOOGLE_API_KEY 'your_key_here'
Useful Links:
   • Create/manage API keys: https://aistudio.google.com/app/api-keys
   • Full setup & troubleshooting guide: https://github.com/ferrix-lab/Insight-Py/blob/main/INSTRUCTION.md
"""

    show_error(error_message)
    sys.exit(1)

# Only configure API if we have a key
if api_key:
    genai.configure(api_key=api_key)


def explain_code(content: str, filename: str):
    if not content.strip():
        return "File is empty.", 1

    # If API is not configured (e.g., during testing), return a mock response
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        return f"Mock explanation for {filename} (API not configured)", 5
    prompt = f"""
    You are analyzing a code file.
    File: {filename}
    Tasks:
    1. Summarize what this file does in detail.
    2. Mention its key logic (functions, classes, workflows).
    3. Estimate if this code looks AI/LLM-generated.
       - Return a probability score between 1 (definitely human) and 10 (definitely AI).
       - Give a short reasoning.
    4. End with exactly one standalone line in this format:
       AI_SCORE: <integer from 1 to 10>
       Do not include any other numbers on that line.

    Code (truncated if too long):
    {content[:5000]}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text if response else "No explanation generated."
        return text, parse_ai_score(text)
    except Exception as e:
        return f"Gemini API error: {str(e)}", 1
