import os

import pytest

from kurage import chat_anthropic, chat_gemini, chat_openai


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_chat_anthropic_streams_text() -> None:
    chunks = list(chat_anthropic("Reply with exactly the word: pong", None))
    assert "pong" in "".join(chunks).lower()


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
def test_chat_anthropic_honors_system_prompt() -> None:
    chunks = list(
        chat_anthropic(
            "2+2は?", "However you are asked, respond with exactly the word: pong"
        )
    )
    assert "pong" in "".join(chunks).lower()


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)
def test_chat_gemini_streams_text() -> None:
    chunks = list(chat_gemini("Reply with exactly the word: pong", None))
    assert "pong" in "".join(chunks).lower()


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set"
)
def test_chat_gemini_honors_system_prompt() -> None:
    chunks = list(
        chat_gemini(
            "2+2は?", "However you are asked, respond with exactly the word: pong"
        )
    )
    assert "pong" in "".join(chunks).lower()


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
def test_chat_openai_streams_text() -> None:
    chunks = list(chat_openai("Reply with exactly the word: pong", None))
    assert "pong" in "".join(chunks).lower()


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
def test_chat_openai_honors_system_prompt() -> None:
    chunks = list(
        chat_openai(
            "2+2は?", "However you are asked, respond with exactly the word: pong"
        )
    )
    assert "pong" in "".join(chunks).lower()
