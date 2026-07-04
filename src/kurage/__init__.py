import sys
from argparse import ArgumentParser
from pathlib import Path
from textwrap import indent

import anthropic


def chat(messages, system):
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


def read_context():
    messages = []
    for line in sys.stdin:
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


def write_context(messages):
    for message in messages:
        role, content = message["role"], message["content"].rstrip()
        if "\n" not in content:
            print(f"{role}: {content}")
        else:
            print(f"{role}: ")
            print(indent(content, "  "))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--system",
        "-s",
        help="File containing system prompt",
    )
    args = parser.parse_args()
    messages = read_context()
    system = Path(args.system).read_text() if args.system is not None else ""
    if messages:
        messages = chat(messages, system)
        write_context(messages)
    print("user: ")
