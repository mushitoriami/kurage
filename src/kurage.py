import sys
from argparse import ArgumentParser
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, get_args

import anthropic
import openai
from google import genai
from google.genai import types as genai_types
from openai.types.chat import ChatCompletionMessageParam

Provider = Literal["Anthropic", "Gemini", "OpenAI"]


def chat_anthropic(question: str, system: str | None) -> Iterator[str]:
    client = anthropic.Anthropic()
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=128000,
        system=system if system is not None else anthropic.omit,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": question}],
    ) as stream:
        yield from stream.text_stream
        if (stop_reason := stream.get_final_message().stop_reason) != "end_turn":
            raise RuntimeError(f"Anthropic response did not finish normally: {stop_reason}")


def chat_gemini(question: str, system: str | None) -> Iterator[str]:
    client = genai.Client()
    config = (
        genai_types.GenerateContentConfig(system_instruction=system) if system is not None else None
    )
    for chunk in client.models.generate_content_stream(  # pyright: ignore[reportUnknownMemberType]
        model="gemini-3.6-flash",
        contents=question,
        config=config,
    ):
        if chunk.text:
            yield chunk.text
        finish_reason = chunk.candidates[0].finish_reason if chunk.candidates else None
        if finish_reason not in (None, genai_types.FinishReason.STOP):
            raise RuntimeError(f"Gemini response did not finish normally: {finish_reason}")


def chat_openai(question: str, system: str | None) -> Iterator[str]:
    client = openai.OpenAI()
    messages: list[ChatCompletionMessageParam] = []
    if system is not None:
        messages.append({"role": "developer", "content": system})
    messages.append({"role": "user", "content": question})
    stream = client.chat.completions.create(
        model="gpt-5.4-2026-03-05",
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        choice = chunk.choices[0]
        if choice.delta.content is not None:
            yield choice.delta.content
        if choice.finish_reason not in (None, "stop"):
            raise RuntimeError(f"OpenAI response did not finish normally: {choice.finish_reason}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--system",
        "-s",
        help="File containing system prompt",
    )
    parser.add_argument(
        "--provider",
        "-p",
        default="Anthropic",
        choices=get_args(Provider),
        help="Provider",
    )
    args = parser.parse_args()
    provider: Provider = args.provider
    question = sys.stdin.read()
    system = Path(args.system).read_text() if args.system is not None else None
    match provider:
        case "Anthropic":
            answer = chat_anthropic(question, system)
        case "Gemini":
            answer = chat_gemini(question, system)
        case "OpenAI":
            answer = chat_openai(question, system)
    for chunk in answer:
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print()
