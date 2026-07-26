import sys
from argparse import ArgumentParser
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, get_args

import anthropic
import openai
from openai.types.chat import ChatCompletionMessageParam

Provider = Literal["Anthropic", "OpenAI"]


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
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content


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
        case "OpenAI":
            answer = chat_openai(question, system)
    for chunk in answer:
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print()
