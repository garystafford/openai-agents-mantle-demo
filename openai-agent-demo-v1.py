import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
    api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
)


def ask(prompt: str) -> str:
    response = client.responses.create(
        model="openai.gpt-5.5",
        instructions="You answer questions clearly and concisely.",
        input=prompt,
    )
    return response.output_text


if __name__ == "__main__":
    print(ask("How does a rainbow form?"))
