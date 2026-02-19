from argparse import ArgumentParser
from pathlib import Path
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
    parser.add_argument(
        "--system",
        "-s",
        help="File containing system prompt",
    )
    parser.add_argument("--max-tokens", "-m", default=2048, help="Max tokens")
    parser.add_argument(
        "--budget-tokens",
        "-b",
        default=1024,
        help="Budget tokens (for extended thinking)",
    )
    args = parser.parse_args()

    client = anthropic.Anthropic()
    messages = []
    system_prompt = Path(args.system).read_text() if args.system is not None else ""
    while True:
        print("\nUser:\n")
        user_input = prompt("  | ", multiline=True, prompt_continuation="  | ")
        messages.append({"role": "user", "content": user_input})
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=int(args.max_tokens),
            system=system_prompt,
            thinking={"type": "enabled", "budget_tokens": int(args.budget_tokens)}
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
