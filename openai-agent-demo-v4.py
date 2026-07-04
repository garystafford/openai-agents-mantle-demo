import asyncio
import os
import re
from datetime import date
from typing import Literal

import httpx
from agents import (
    Agent,
    ModelSettings,
    Runner,
    TracingProcessor,
    add_trace_processor,
    function_tool,
    set_default_openai_client,
    set_trace_processors,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry import trace as trace_api
from phoenix.otel import register
from pydantic import BaseModel
from tavily import TavilyClient

load_dotenv()

client = AsyncOpenAI(
    base_url="https://bedrock-mantle.us-east-1.api.aws/openai/v1",
    api_key=os.environ["AWS_BEARER_TOKEN_BEDROCK"],
)
set_default_openai_client(client, use_for_tracing=False)

MAX_TURNS = 25
PHOENIX_PROJECT_NAME = "openai-agents-mantle-demo"
DEMO_SCRIPT = "v4"


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


tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@function_tool
async def web_search(query: str) -> str:
    """Search the web for authoritative sources on a factual claim.

    Returns up to 3 results as plain text (title, url, snippet). Use for
    dates, numbers, names, causal claims, anything contested or post-training.
    """
    try:
        response = await asyncio.to_thread(
            tavily_client.search,
            query=query,
            max_results=3,
            search_depth="basic",
            include_answer=False,
        )
    except Exception as e:
        return f"Search failed: {e}"

    results = response.get("results", [])
    if not results:
        return "No results."
    return "\n\n".join(
        f"{r.get('title', '(no title)')}\n{r.get('url', '')}\n{r.get('content', '')}"
        for r in results
    )


@function_tool
async def read_page(url: str) -> str:
    """Fetch and return the readable text of a web page.

    Use when a search snippet is promising but you need the full context
    (e.g. the snippet claims X but you want to see what the source
    actually says). Returns up to ~8000 chars of extracted text.
    """
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 (fact-checker-demo)"},
        ) as http:
            resp = await http.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        return f"Fetch failed: {e}"

    # Strip scripts/styles/tags for a rough text view. Good enough for
    # the model to read; not trying to be a real HTML parser.
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:8000] if text else "No readable text extracted."


@function_tool
def current_date() -> str:
    """Return today's date in ISO format (YYYY-MM-DD).

    Use to reason about how recent a claim is relative to now, or to
    check whether an event described as 'current' has since ended.
    """
    return date.today().isoformat()


class Issue(BaseModel):
    claim: str
    problem: str


class FactCheck(BaseModel):
    verdict: Literal["supported", "disputed", "unsupported"]
    confidence: float
    issues: list[Issue]
    notes: str


class QAAnswer(BaseModel):
    answer: str
    sources_used: list[str]


fact_checker = Agent(
    name="Fact checker",
    instructions=(
        "You verify factual claims. Given one or more claims, judge each "
        "against well-established facts. For any claim you cannot verify "
        "with high confidence from your own knowledge — dates, numbers, "
        "names, causal claims, anything contested or post-training — call "
        "the web_search tool with a specific query. When a search snippet "
        "looks decisive but you want to confirm the source actually says "
        "what the snippet implies, call read_page on the url. Call "
        "current_date when a claim depends on how recent something is "
        "('currently', 'ongoing', 'former', ages, tenures). "
        "Tool budget: at most one web_search per distinct claim and at "
        "most one read_page follow-up per search. If evidence is still "
        "ambiguous after that, return verdict='disputed' with what you "
        "found rather than searching again. Judge based on returned "
        "sources, not general impression. Return a single FactCheck: "
        "verdict='supported' only if every claim is accurate; 'disputed' "
        "if any claim is contested by mainstream sources; 'unsupported' "
        "if any claim is factually wrong. Populate issues[] with each "
        "problematic claim and a short explanation. When a verdict rests "
        "on search results, cite the sources in notes."
    ),
    model="openai.gpt-5.5",
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium")),
    tools=[web_search, read_page, current_date],
    output_type=FactCheck,
)


# Custom fact_check wrapper instead of fact_checker.as_tool() so the
# inner Runner gets a raised max_turns — with reasoning + web_search +
# read_page the fact_checker routinely exceeds the SDK default of 10.
@function_tool(
    name_override="fact_check",
    description_override=(
        "Verify factual claims. Pass the specific claims from "
        "your draft answer as a single string."
    ),
)
async def fact_check(claims: str) -> FactCheck:
    result = await Runner.run(fact_checker, claims, max_turns=MAX_TURNS)
    return result.final_output


qa_agent = Agent(
    name="Q&A Agent",
    instructions=(
        "You answer questions clearly and concisely. Before drafting, "
        "call current_date if the question involves anything time-"
        "sensitive (current officeholders, ages, tenures, 'recent', "
        "'ongoing', 'former') so you don't draft claims based on stale "
        "training data. Then draft an answer and call the fact_check "
        "tool with the specific factual claims (dates, names, causes, "
        "outcomes). Call fact_check at most twice: once on the initial "
        "draft, and, if the verdict is not 'supported', once more on a "
        "revised draft that addresses the flagged issues. If the second "
        "verdict is still not 'supported', return the best answer you "
        "have and note the remaining uncertainty in it — do not call "
        "fact_check a third time. Set sources_used to the urls the "
        "fact_check cited in its notes."
    ),
    model="openai.gpt-5.4",
    tools=[current_date, fact_check],
    output_type=QAAnswer,
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


def _print_answer(answer: QAAnswer) -> None:
    print("\n--- final answer ---")
    print(answer.answer)
    if answer.sources_used:
        print("\nsources:")
        for src in answer.sources_used:
            print(f"  - {src}")


async def main(prompt: str) -> None:
    try:
        streamed = Runner.run_streamed(qa_agent, prompt, max_turns=MAX_TURNS)
        async for event in streamed.stream_events():
            _print_stream_event(event)
        _print_answer(streamed.final_output)
    finally:
        tracer_provider.shutdown()  # flush any pending spans before exit


if __name__ == "__main__":
    asyncio.run(main("Are UFOs real?"))
