import sys
from argparse import ArgumentParser
from pathlib import Path
from textwrap import indent

import anthropic
import openai


def chat_anthropic(messages, system):
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=system,
        thinking={"type": "adaptive"},
        messages=messages,
    )
    for block in response.content:
        if block.type == "text":
            messages.append({"role": "assistant", "content": block.text})
    return messages


def chat_openai(messages, _system):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-5.4-2026-03-05", messages=messages
    )
    messages.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )
    return messages


def load_conversation(fp=sys.stdin):
    messages = []
    for line in fp:
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


def dump_conversation(messages, fp=sys.stdout):
    for message in messages:
        role, content = message["role"], message["content"].rstrip()
        if "\n" not in content:
            print(f"{role}: {content}", file=fp)
        else:
            print(f"{role}: ", file=fp)
            print(indent(content, "  "), file=fp)


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
    messages = load_conversation()
    system = Path(args.system).read_text() if args.system is not None else ""
    if messages:
        if args.provider == "Anthropic":
            messages = chat_anthropic(messages, system)
        elif args.provider == "OpenAI":
            messages = chat_openai(messages, system)
        else:
            raise ValueError
        dump_conversation(messages)
    print("user: ")
