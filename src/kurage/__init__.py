import json
from argparse import ArgumentParser
from itertools import cycle
from pathlib import Path
from textwrap import indent

import anthropic
from dotenv import load_dotenv
from prompt_toolkit import prompt

load_dotenv()


def construct_context(texts):
    return [
        {"role": role, "content": text}
        for role, text in zip(cycle(["user", "assistant"]), texts)
    ]


def construct_system_and_messages(texts, system_prompt, enable_objective):
    if enable_objective:
        context = json.dumps(construct_context(texts))
        system = json.dumps(system_prompt)
        instruction = f"""
The following is a conversation history between a user and Claude.

{context}

Generate an appropriate response from Claude that continues this conversation. Output only the generated response and nothing else.

Note that Claude is given the following system prompt:

{system}
"""
        return "", [{"role": "user", "content": instruction}]
    else:
        return system_prompt, construct_context(texts)


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
        "--objective",
        "-o",
        action="store_true",
        default=False,
        help="Enable objective mode",
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
    texts = []
    system_prompt = Path(args.system).read_text() if args.system is not None else ""
    while True:
        print("\nUser:\n")
        user_input = prompt("  | ", multiline=True, prompt_continuation="  | ")
        texts.append(user_input)
        system, messages = construct_system_and_messages(
            texts, system_prompt, args.objective
        )
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=int(args.max_tokens),
            system=system,
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
                texts.append(block.text)
