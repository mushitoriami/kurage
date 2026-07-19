import sys
from argparse import ArgumentParser
from pathlib import Path

import anthropic
import openai


def chat_anthropic(question: str, system: str) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=system,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": question}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    raise NotImplementedError


def chat_openai(question: str, system: str) -> str:
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-5.4-2026-03-05",
        messages=[{"role": "user", "content": question}],
    )
    text = response.choices[0].message.content
    if text is None:
        raise NotImplementedError
    return text


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
        choices=("Anthropic", "OpenAI"),
        help="Provider",
    )
    args = parser.parse_args()
    question = sys.stdin.read()
    system = Path(args.system).read_text() if args.system is not None else ""
    if args.provider == "Anthropic":
        answer = chat_anthropic(question, system)
    elif args.provider == "OpenAI":
        answer = chat_openai(question, system)
    else:
        raise ValueError
    print(answer)
