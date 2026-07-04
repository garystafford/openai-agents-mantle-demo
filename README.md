# OpenAI Agents SDK on Amazon Bedrock Mantle

A question-answering agent built with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) against [Amazon Bedrock Mantle](https://aws.amazon.com/bedrock/)'s OpenAI-compatible endpoint, developed incrementally across six versions.

## What's in the repo

| Script                                                             | Adds                                                                                                 |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| [openai-agent-demo-v1.py](openai-agent-demo-v1.py)                 | Bare `OpenAI` client pointed at the Mantle base URL                                                  |
| [openai-agent-demo-v2.py](openai-agent-demo-v2.py)                 | `Agent` + `Runner` from the Agents SDK (async)                                                       |
| [openai-agent-demo-v2-tracing.py](openai-agent-demo-v2-tracing.py) | Variant of v2 with a custom `TracingProcessor` that prints spans to the terminal (no Phoenix/Docker) |
| [openai-agent-demo-v3.py](openai-agent-demo-v3.py)                 | Agent-as-tool fact checker, Pydantic structured output, Arize Phoenix tracing                        |
| [openai-agent-demo-v4.py](openai-agent-demo-v4.py)                 | Real tools (Tavily web search, page fetch, current date), reasoning effort                           |
| [openai-agent-demo-v5.py](openai-agent-demo-v5.py)                 | Input/output guardrails, typed `RunContext`, hard search budget                                      |
| [openai-agent-demo-v6.py](openai-agent-demo-v6.py)                 | Handoffs, triage router, graceful tool-failure handling                                              |

The capabilities are cumulative: each version retains everything before it and adds one row. The staircase of checkboxes shows where each feature enters, and the final row shows where the system becomes an agent.

| Capability                              | v1  | v2  | v3  | v4  | v5  | v6  |
| --------------------------------------- | :-: | :-: | :-: | :-: | :-: | :-: |
| Agents                                  |  0  |  1  |  2  |  2  |  2  |  4  |
| Lines of code                           | 24  | 32  | 107 | 236 | 361 | 432 |
| Agents SDK framework (`Agent`/`Runner`) |     | ✅  | ✅  | ✅  | ✅  | ✅  |
| Multiple agents                         |     |     | ✅  | ✅  | ✅  | ✅  |
| Structured output (Pydantic)            |     |     | ✅  | ✅  | ✅  | ✅  |
| Streaming                               |     |     | ✅  | ✅  | ✅  | ✅  |
| Tracing (Phoenix)                       |     |     | ✅  | ✅  | ✅  | ✅  |
| Tools (web search, fetch, date)         |     |     |     | ✅  | ✅  | ✅  |
| Reasoning effort                        |     |     |     | ✅  | ✅  | ✅  |
| Guardrails (input/output)               |     |     |     |     | ✅  | ✅  |
| Typed run context + search budget       |     |     |     |     | ✅  | ✅  |
| Handoffs + triage router                |     |     |     |     |     | ✅  |
| Tool failure handling                   |     |     |     |     |     | ✅  |
| **Agent** (model-authored control flow) | ❌  | ❌  | ❌  | ✅  | ✅  | ✅  |

## Prerequisites

- Python 3.11+ (tested against 3.13)
- An [AWS Bedrock Amazon Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys-how.html)
- A [Tavily](https://tavily.com) API key (required from v4 onward)
- Docker, to run [Arize Phoenix](https://phoenix.arize.com) locally for tracing (v3+)

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Environment variables

Create a `.env` file at the repo root:

```text
AWS_BEARER_TOKEN_BEDROCK=your-bedrock-api-key
TAVILY_API_KEY=your-tavily-key
```

The demo scripts load it automatically via `python-dotenv`. To export the values into your current shell as well:

```bash
set -a && source .env && set +a
```

## Optional: start Phoenix for tracing (v3+)

```bash
docker run -p 6006:6006 arizephoenix/phoenix:latest
```

Traces are sent to `http://localhost:6006/v1/traces`; open `http://localhost:6006` in a browser to view them.

## Run a demo

```bash
python openai-agent-demo-v1.py
# ...through v6
```

Each script is a standalone entry point — start at v1 and work up.
