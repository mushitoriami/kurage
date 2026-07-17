import sys
from argparse import ArgumentParser
from pathlib import Path
from textwrap import indent
from typing import TextIO

import anthropic
import openai

type Messages = list[dict[str, str]]


def chat_anthropic(question: str, system: str):
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


def chat_openai(question: str, _system: str):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-5.4-2026-03-05",
        messages=[{"role": "user", "content": question}],
    )
    text = response.choices[0].message.content
    if text is None:
        raise NotImplementedError
    return text


def loads_conversation(string: str):
    messages: Messages = []
    for line in string.splitlines(keepends=True):
        if line.startswith("user:"):
            messages.append({"role": "user", "content": line[len("user:") :].lstrip()})
        elif line.startswith("assistant:"):
            messages.append(
                {"role": "assistant", "content": line[len("assistant:") :].lstrip()}
            )
        elif line.startswith("  "):
            messages[-1]["content"] += line[len("  ") :]
        elif line.strip() == "":
            messages[-1]["content"] += "\n"
        else:
            raise ValueError
    return messages


def load_conversation(fp: TextIO = sys.stdin):
    return loads_conversation(fp.read())


def dumps_conversation(messages: Messages):
    string = ""
    for message in messages:
        role, content = message["role"], message["content"].rstrip()
        if "\n" not in content:
            string += f"{role}: {content}\n"
        else:
            string += f"{role}: \n" + indent(content, "  ") + "\n"
    return string


def dump_conversation(messages: Messages, fp: TextIO = sys.stdout):
    fp.write(dumps_conversation(messages))


def main():
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
