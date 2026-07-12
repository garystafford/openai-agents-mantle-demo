import asyncio
import os
from datetime import datetime
from typing import Any

from agents import (
    Agent,
    ModelSettings,
    Runner,
    set_default_openai_client,
    set_trace_processors,
)
from agents.tracing import Span, Trace, TracingProcessor
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.shared import Reasoning

load_dotenv()

client = AsyncOpenAI(
    base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
    api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
)
set_default_openai_client(client, use_for_tracing=False)


class ConsoleTracingProcessor(TracingProcessor):
    """Prints trace and span events to the terminal.

    Replaces the SDK's default BackendSpanExporter (which uploads to
    openai.com) — useful when the model endpoint isn't OpenAI's.
    """

    def on_trace_start(self, trace: Trace) -> None:
        print(f"[trace start] {trace.name}  id={trace.trace_id}")

    def on_trace_end(self, trace: Trace) -> None:
        print(f"[trace end]   {trace.name}")

    def on_span_start(self, span: Span[Any]) -> None:
        kind = type(span.span_data).__name__.removesuffix("SpanData").lower()
        print(f"  [span start] {kind}")

    def on_span_end(self, span: Span[Any]) -> None:
        payload = span.export() or {}
        data = payload.get("span_data", {}) or {}
        kind = data.pop("type", "span")
        duration_ms = _duration_ms(payload.get("started_at"), payload.get("ended_at"))
        summary = ", ".join(f"{k}={_short(v)}" for k, v in data.items() if v)
        line = f"  [span end]   {kind}  {duration_ms:>6.1f} ms"
        if summary:
            line += f"  | {summary}"
        print(line)
        if payload.get("error"):
            print(f"    ! error: {payload['error']}")

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


def _duration_ms(started_at: str | None, ended_at: str | None) -> float:
    if not started_at or not ended_at:
        return 0.0
    start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    return (end - start).total_seconds() * 1000


def _short(value: Any, limit: int = 80) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


set_trace_processors([ConsoleTracingProcessor()])


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
    print(f"\n{result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
