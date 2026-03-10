import json
from argparse import ArgumentParser
from itertools import cycle
from pathlib import Path
from textwrap import dedent, indent

import anthropic
from dotenv import load_dotenv
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings

kb = KeyBindings()


@kb.add("escape", "[", "1", "3", ";", "2", "u")
def _(event):
    event.current_buffer.insert_text("\n")


@kb.add("enter")
def _(event):
    event.current_buffer.validate_and_handle()


load_dotenv()


def construct_context(roles, texts):
    return [{"role": role, "content": text} for role, text in zip(cycle(roles), texts)]


def construct_system_and_messages(texts, system_prompt, character_setting):
    if character_setting is not None:
        context = json.dumps(construct_context(["Q", "P"], texts))
        setting = json.dumps(character_setting)
        instruction = dedent(f"""
            The following is a conversation history between two fictional characters.

            {context}

            Generate P's next utterance that continues this conversation.
            Output only the generated utterance and nothing else.

            The character settings for P and Q are as follows:

            {setting}
            """)
        return "", [{"role": "user", "content": instruction}]
    else:
        return system_prompt, construct_context(["user", "assistant"], texts)


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
        "--character",
        "-c",
        help="File containing character setting (enable conversation mode)",
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
    character_setting = (
        Path(args.character).read_text() if args.character is not None else None
    )
    while True:
        print("\nUser:\n")
        user_input = prompt(
            "  | ", key_bindings=kb, multiline=True, prompt_continuation="  | "
        )
        texts.append(user_input)
        system, messages = construct_system_and_messages(
            texts, system_prompt, character_setting
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
