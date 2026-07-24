import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, get_args

import anthropic
import openai
from openai.types.chat import ChatCompletionMessageParam

Provider = Literal["Anthropic", "OpenAI"]


def enable_debug_logging() -> None:
    logging.basicConfig(
        format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for name in ("anthropic", "openai", "httpx"):
        logging.getLogger(name).setLevel(logging.DEBUG)


def chat_anthropic(question: str, system: str | None) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=system if system is not None else anthropic.omit,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": question}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    raise RuntimeError("Anthropic response contained no text block")


def chat_openai(question: str, system: str | None) -> str:
    client = openai.OpenAI()
    messages: list[ChatCompletionMessageParam] = []
    if system is not None:
        messages.append({"role": "developer", "content": system})
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(
        model="gpt-5.4-2026-03-05",
        messages=messages,
    )
    text = response.choices[0].message.content
    if text is None:
        raise RuntimeError("OpenAI response contained no text")
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
        choices=get_args(Provider),
        help="Provider",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Log HTTP request/response details to stderr",
    )
    args = parser.parse_args()
    if args.debug:
        enable_debug_logging()
    provider: Provider = args.provider
    question = sys.stdin.read()
    system = Path(args.system).read_text() if args.system is not None else None
    match provider:
        case "Anthropic":
            answer = chat_anthropic(question, system)
        case "OpenAI":
            answer = chat_openai(question, system)
    print(answer)
