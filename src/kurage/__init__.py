from argparse import ArgumentParser
from textwrap import indent

import anthropic
from dotenv import load_dotenv
from prompt_toolkit import prompt

load_dotenv()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--thinking",
        "-t",
        action="store_true",
        default=False,
        help="Enable extended thinking",
    )

    args = parser.parse_args()

    client = anthropic.Anthropic()
    messages = []
    while True:
        print("\nUser:\n")
        user_input = prompt("  | ", multiline=True, prompt_continuation="  | ")
        messages.append({"role": "user", "content": user_input})
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 4096}
            if args.thinking
            else {"type": "disabled"},
            messages=messages,
        )
        for block in response.content:
            if block.type == "thinking":
                print("\nAssistant [Thinking]:\n")
                print(indent(block.thinking, "  | ", predicate=(lambda x: True)))
            if block.type == "text":
                print("\nAssistant:\n")
                print(indent(block.text, "  | ", predicate=(lambda x: True)))
        messages.append({"role": "assistant", "content": response.content})
