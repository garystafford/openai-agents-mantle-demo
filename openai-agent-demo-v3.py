import asyncio
import os
from typing import Literal

from agents import (
    Agent,
    Runner,
    TracingProcessor,
    add_trace_processor,
    set_default_openai_client,
    set_trace_processors,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry import trace as trace_api
from phoenix.otel import register
from pydantic import BaseModel

load_dotenv()

client = AsyncOpenAI(
    base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
    api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
)
set_default_openai_client(client, use_for_tracing=False)

PHOENIX_PROJECT_NAME = "openai-agents-mantle-demo"
DEMO_SCRIPT = "v3"


class DemoScriptSpanProcessor(TracingProcessor):
    def __init__(self, script: str):
        self.script = script

    def on_trace_start(self, trace) -> None:
        pass

    def on_trace_end(self, trace) -> None:
        pass

    def on_span_start(self, span) -> None:
        current_span = trace_api.get_current_span()
        current_span.set_attribute("demo.script", self.script)

    def on_span_end(self, span) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


# Keep SDK tracing enabled — OpenAIAgentsInstrumentor listens on that pipeline
# and re-emits spans to Phoenix. Clear the default processors so the SDK
# doesn't also try to upload traces to openai.com in parallel.
set_trace_processors([])
tracer_provider = register(
    project_name=PHOENIX_PROJECT_NAME,
    endpoint="http://localhost:6006/v1/traces",
    batch=True,  # BatchSpanProcessor instead of the default SimpleSpanProcessor
    verbose=False,  # suppress the startup OpenTelemetry details banner
)
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
add_trace_processor(DemoScriptSpanProcessor(DEMO_SCRIPT))


class Issue(BaseModel):
    claim: str
    problem: str


class FactCheck(BaseModel):
    verdict: Literal["supported", "disputed", "unsupported"]
    confidence: float
    issues: list[Issue]
    notes: str


fact_checker = Agent(
    name="Fact checker",
    model="openai.gpt-5.5",
    instructions=(
        "You verify factual claims. Given one or more claims, judge each "
        "against well-established facts. Return a single FactCheck: "
        "verdict='supported' only if every claim is accurate; 'disputed' if "
        "any claim is contested by mainstream sources; 'unsupported' if any "
        "claim is factually wrong. Populate issues[] with each problematic "
        "claim and a short explanation."
    ),
    output_type=FactCheck,
)


qa_agent = Agent(
    name="Q&A Agent",
    model="openai.gpt-5.4",
    instructions=(
        "You answer questions clearly and concisely. Before finalizing an "
        "answer, call the fact_check tool with the specific factual claims "
        "in your draft (dates, names, causes, outcomes). If the verdict is "
        "not 'supported', revise your answer to fix the flagged issues, "
        "then answer. If 'supported', answer as drafted."
    ),
    tools=[
        fact_checker.as_tool(
            tool_name="fact_check",
            tool_description=(
                "Verify factual claims. Pass the specific claims from your "
                "draft answer as a single string."
            ),
        )
    ],
)


def _print_stream_event(event) -> None:
    if event.type == "agent_updated_stream_event":
        print(f"[agent] → {event.new_agent.name}")
    elif event.type == "run_item_stream_event":
        item = event.item
        if item.type == "tool_call_item":
            raw = item.raw_item
            name = getattr(raw, "name", None) or getattr(raw, "type", "tool")
            print(f"[tool call] {name}")
        elif item.type == "tool_call_output_item":
            preview = str(item.output).strip().replace("\n", " ")[:120]
            print(f"[tool output] {preview}")


async def main(prompt: str) -> None:
    try:
        streamed = Runner.run_streamed(qa_agent, prompt)
        async for event in streamed.stream_events():
            _print_stream_event(event)
        print(f"\n{streamed.final_output}")
    finally:
        tracer_provider.shutdown()  # flush any pending spans before exit


if __name__ == "__main__":
    asyncio.run(main("How does a rainbow form?"))
