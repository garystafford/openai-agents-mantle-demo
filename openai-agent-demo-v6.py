import asyncio
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import httpx
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    ModelSettings,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TracingProcessor,
    add_trace_processor,
    function_tool,
    handoff,
    input_guardrail,
    output_guardrail,
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
SEARCH_BUDGET = 5
PHOENIX_PROJECT_NAME = "openai-agents-mantle-demo"
DEMO_SCRIPT = "v6"


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


@dataclass
class RunContext:
    """Per-run state threaded through every tool and guardrail.

    The Agents SDK passes this to any tool that declares
    RunContextWrapper[RunContext] as its first parameter. Because the
    fact_check wrapper forwards the parent's context to the inner
    Runner.run, the same counter is shared across the whole run tree —
    a search budget that spans both agents, not per-agent.
    """

    tavily_calls: int = 0
    tavily_queries: list[str] = field(default_factory=list)


@function_tool
async def web_search(ctx: RunContextWrapper[RunContext], query: str) -> str:
    """Search the web for authoritative sources on a factual claim.

    Returns up to 3 results as plain text (title, url, snippet). Use for
    dates, numbers, names, causal claims, anything contested or post-training.
    """
    if ctx.context.tavily_calls >= SEARCH_BUDGET:
        return (
            f"Search budget exhausted ({SEARCH_BUDGET} calls used). "
            "Judge with existing evidence."
        )
    ctx.context.tavily_calls += 1
    ctx.context.tavily_queries.append(query)

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
    confidence: float


class InputCheck(BaseModel):
    is_valid: bool
    reason: str


class OutputCheck(BaseModel):
    is_valid: bool
    reason: str


@input_guardrail
async def prompt_length_guardrail(
    ctx: RunContextWrapper[RunContext], agent: Agent, prompt: str | list
) -> GuardrailFunctionOutput:
    """Trip if the user prompt is trivially short.

    Rule-based, not LLM-based, so the mechanism is obvious in the trace.
    Real deployments would gate on PII, prompt injection, off-topic
    queries — often via a small classifier agent.
    """
    text = prompt if isinstance(prompt, str) else str(prompt)
    is_valid = len(text.strip()) >= 10
    return GuardrailFunctionOutput(
        output_info=InputCheck(
            is_valid=is_valid,
            reason="ok" if is_valid else "prompt is too short to answer meaningfully",
        ),
        tripwire_triggered=not is_valid,
    )


@output_guardrail
async def answer_quality_guardrail(
    ctx: RunContextWrapper[RunContext], agent: Agent, answer: QAAnswer
) -> GuardrailFunctionOutput:
    """Trip if the final answer looks under-supported.

    Reruns qa_agent when confidence is low or when a non-trivial answer
    cites no sources. Pairs with the input guardrail to bracket the run
    with sanity checks at both ends.
    """
    low_confidence = answer.confidence < 0.5
    needs_sources = len(answer.answer) > 100 and not answer.sources_used
    is_valid = not (low_confidence or needs_sources)
    reason = "ok"
    if low_confidence:
        reason = f"confidence {answer.confidence} below 0.5"
    elif needs_sources:
        reason = "substantive answer cites no sources"
    return GuardrailFunctionOutput(
        output_info=OutputCheck(is_valid=is_valid, reason=reason),
        tripwire_triggered=not is_valid,
    )


fact_checker = Agent[RunContext](
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
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="medium", summary="auto"),
    ),
    tools=[web_search, read_page, current_date],
    output_type=FactCheck,
)


def fact_check_failed(ctx: RunContextWrapper[RunContext], error: Exception) -> str:
    """Turn a crash in the nested fact-check run into an instruction the
    Q&A agent can act on, instead of letting it propagate and kill the run.

    The inner Runner can raise (e.g. MaxTurnsExceeded, a model error). By
    default that surfaces as a generic error string; here we tell the agent
    explicitly to proceed unverified and flag the uncertainty — so a broken
    tool degrades the answer rather than crashing the program.
    """
    return (
        f"fact_check could not complete ({type(error).__name__}). Proceed "
        "with your best answer from your own knowledge, lower your "
        "confidence, and note that the claims could not be independently "
        "verified. Do not call fact_check again."
    )


# Custom fact_check wrapper instead of fact_checker.as_tool() so the
# inner Runner gets a raised max_turns AND inherits the parent run's
# RunContext — that's how the search budget spans both agents.
# failure_error_function keeps a crash in the nested run from killing the
# whole run: the exception is converted to a message the agent routes around.
@function_tool(
    name_override="fact_check",
    description_override=(
        "Verify factual claims. Pass the specific claims from "
        "your draft answer as a single string."
    ),
    failure_error_function=fact_check_failed,
)
async def fact_check(ctx: RunContextWrapper[RunContext], claims: str) -> FactCheck:
    result = await Runner.run(
        fact_checker, claims, context=ctx.context, max_turns=MAX_TURNS
    )
    return result.final_output


qa_agent = Agent[RunContext](
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
        "fact_check cited; set confidence to reflect how well-supported "
        "the final answer is."
    ),
    model="openai.gpt-5.4",
    tools=[current_date, fact_check],
    output_guardrails=[answer_quality_guardrail],
    output_type=QAAnswer,
)


# A second specialist with no fact-checking apparatus. Creative requests
# (write a poem, brainstorm names) have no truth value, so routing them
# through the web-search + fact_check loop would be wasted work — there's
# nothing to verify. This is why triage HANDS OFF rather than calls a tool:
# it transfers control and steps out; the specialist owns the response.
creative_agent = Agent[RunContext](
    name="Creative Agent",
    instructions=(
        "You handle open-ended creative requests — poems, stories, "
        "brainstorms, rewrites. Respond directly and imaginatively. Do "
        "not fact-check; these tasks have no factual claims to verify."
    ),
    model="openai.gpt-5.4",
)


triage_agent = Agent[RunContext](
    name="Triage Agent",
    instructions=(
        "You route each request to the right specialist and do not answer "
        "it yourself. Hand off to the Q&A Agent for questions with a "
        "factual answer (dates, names, causes, how things work, whether a "
        "claim is true). Hand off to the Creative Agent for open-ended "
        "generative tasks with no factual answer (write a poem or story, "
        "brainstorm ideas, rewrite this text). When a request has both, "
        "prefer the Q&A Agent. Always hand off; never answer directly."
    ),
    model="openai.gpt-5.4",
    # tool_name_override keeps the generated transfer tool names valid —
    # the SDK derives them from agent names, and "Q&A Agent" has chars
    # ("&", space) that aren't allowed in function-call tool names.
    handoffs=[
        handoff(qa_agent, tool_name_override="transfer_to_qa_agent"),
        handoff(creative_agent, tool_name_override="transfer_to_creative_agent"),
    ],
    input_guardrails=[prompt_length_guardrail],
)


def _print_stream_event(event) -> None:
    if event.type == "agent_updated_stream_event":
        # Fires on the initial agent and again after each handoff, so the
        # routing decision is visible: Triage Agent → Q&A Agent (or Creative).
        print(f"[agent] → {event.new_agent.name}")
    elif event.type == "run_item_stream_event":
        item = event.item
        if item.type == "handoff_output_item":
            print(f"[handoff] {item.source_agent.name} → {item.target_agent.name}")
        elif item.type == "tool_call_item":
            raw = item.raw_item
            name = getattr(raw, "name", None) or getattr(raw, "type", "tool")
            print(f"[tool call] {name}")
        elif item.type == "tool_call_output_item":
            preview = str(item.output).strip().replace("\n", " ")[:120]
            print(f"[tool output] {preview}")


def _print_answer(output, ctx: RunContext) -> None:
    print("\n--- final answer ---")
    # The answering agent depends on how triage routed: the Q&A Agent
    # returns a structured QAAnswer, the Creative Agent returns plain text.
    if isinstance(output, QAAnswer):
        print(output.answer)
        print(f"\nconfidence: {output.confidence}")
        if output.sources_used:
            print("sources:")
            for src in output.sources_used:
                print(f"  - {src}")
    else:
        print(output)
    print(f"\ntavily calls: {ctx.tavily_calls}/{SEARCH_BUDGET}")


async def _run_once(prompt: str, ctx: RunContext):
    streamed = Runner.run_streamed(
        triage_agent, prompt, context=ctx, max_turns=MAX_TURNS
    )
    async for event in streamed.stream_events():
        _print_stream_event(event)
    return streamed.final_output


async def main(prompt: str) -> None:
    ctx = RunContext()
    try:
        try:
            output = await _run_once(prompt, ctx)
        except OutputGuardrailTripwireTriggered as e:
            print(
                f"[guardrail] output rejected: "
                f"{e.guardrail_result.output.output_info.reason} — retrying"
            )
            try:
                output = await _run_once(prompt, ctx)
            except OutputGuardrailTripwireTriggered as e2:
                # Retry rejected too — surface the best answer we produced
                # rather than crashing. The rejected output rides along on
                # the exception, so we can still show it, flagged.
                print(
                    f"[guardrail] output rejected again: "
                    f"{e2.guardrail_result.output.output_info.reason} — "
                    "returning last answer with caveat"
                )
                output = e2.guardrail_result.agent_output
        _print_answer(output, ctx)
    except InputGuardrailTripwireTriggered as e:
        print(
            f"[guardrail] input blocked: {e.guardrail_result.output.output_info.reason}"
        )
    finally:
        tracer_provider.shutdown()


if __name__ == "__main__":
    # "Are UFOs real?" routes to the Q&A Agent (factual → fact-checked).
    # "Write a haiku about autumn." routes to the Creative Agent (no facts).
    # A prompt under 10 chars (e.g. "why?") trips the input guardrail.
    asyncio.run(main("Are UFOs real?"))
