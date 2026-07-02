import sys
from argparse import ArgumentParser
from pathlib import Path
from textwrap import indent

import anthropic


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
    parser.add_argument("--max-tokens", "-m", default=8192, help="Max tokens")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    messages = read_context()
    system = Path(args.system).read_text() if args.system is not None else ""
    if messages:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=int(args.max_tokens),
            system=system,
            thinking={"type": "adaptive"},
            messages=messages,
        )
        for block in response.content:
            if block.type == "text":
                messages.append({"role": "assistant", "content": block.text})
        write_context(messages)
    print("user: ")
