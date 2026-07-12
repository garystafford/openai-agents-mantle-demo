import asyncio
import os

from agents import (
    Agent,
    ModelSettings,
    Runner,
    set_default_openai_client,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

load_dotenv()

# Endpoint isn't openai.com, so the SDK's built-in tracing upload has no key.
client = AsyncOpenAI(
    base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
    api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
)
set_default_openai_client(client, use_for_tracing=False)
set_tracing_disabled(True)


agent = Agent(
    name="Q&A Agent",
    model="openai.gpt-5.5",
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="medium", summary="auto"),
    ),
    instructions="You answer questions clearly and concisely.",
)


async def main() -> None:
    result = await Runner.run(agent, "How does a rainbow form?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
