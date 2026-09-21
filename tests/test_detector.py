#!/usr/bin/env python3

import pytest

from insight import detector


@pytest.mark.parametrize(
    ("response_text", "expected_score"),
    [
        ("Summary\nAI_SCORE: 7", 7),
        ("Score: 2/10 because the code reads as human-written.", 2),
        ("Probability: 10", 10),
        ("UTF-8, Python 3.10, 10 functions, and Step 1", 5),
        ("AI_SCORE: 11", 5),
        ("Score: 2\nAI_SCORE: 7", 7),
        ("AI_SCORE: 11\nScore: 2", 5),
        ("AI_SCORE:\n7", 5),
    ],
)
def test_explain_code_parses_only_labeled_scores(mocker, response_text, expected_score):
    model = mocker.patch.object(detector.genai, "GenerativeModel").return_value
    model.generate_content.return_value.text = response_text

    explanation, score = detector.explain_code("print('hello')", "sample.py")

    assert explanation == response_text
    assert score == expected_score


def test_explain_code_requests_dedicated_score_line(mocker):
    model = mocker.patch.object(detector.genai, "GenerativeModel").return_value
    model.generate_content.return_value.text = "Summary\nAI_SCORE: 5"

    detector.explain_code("print('hello')", "sample.py")

    prompt = model.generate_content.call_args.args[0]
    assert "AI_SCORE: <integer from 1 to 10>" in prompt
