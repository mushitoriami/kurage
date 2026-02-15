from dotenv import load_dotenv
import anthropic

load_dotenv()


def main():
    client = anthropic.Anthropic()
    messages = []
    while True:
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929", max_tokens=1024, messages=messages
        )
        assistant_message = response.content[0].text
        print(f"\nAssistant: {assistant_message}\n")
        messages.append({"role": "assistant", "content": assistant_message})
