from textwrap import indent

import anthropic
from dotenv import load_dotenv
from prompt_toolkit import prompt

load_dotenv()


def main():
    client = anthropic.Anthropic()
    messages = []
    while True:
        print("\nUser:\n")
        user_input = prompt("  | ", multiline=True, prompt_continuation="  | ")
        messages.append({"role": "user", "content": user_input})
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929", max_tokens=1024, messages=messages
        )
        assistant_message = response.content[0].text
        print("\nAssistant:\n")
        print(indent(assistant_message, "  | ", predicate=(lambda x: True)))
        messages.append({"role": "assistant", "content": assistant_message})
