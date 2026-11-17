"""ReAct agent (LangGraph) that answers FAERS adverse-event questions via openFDA.

Model: GPT-4o through OpenRouter's OpenAI-compatible API.
Requires OPEN_ROUTER_KEY in the environment (or .env).

Usage:
    uv run python -m test_agent.fears_agent "What are the most reported adverse events for ibuprofen?"
"""

import os
import sys

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from test_agent.services.query_faers import query_faers

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o"

SYSTEM_PROMPT = """You answer questions about drug adverse events using the FDA \
Adverse Event Reporting System (FAERS) via the `faers_top_adverse_events` tool.

Rules:
- Always call the tool before answering; never answer from memory.
- Pass the drug's generic name as `drug` when you know it. If the user gives a brand
  name, pass it as `brand_name` (and the generic as `drug` if you know it).
- Report the reaction terms with their report counts, in descending order.
- If the tool returns no results, say FAERS has no matching reports; do not guess.
- Note briefly that FAERS counts are spontaneous reports and do not establish
  causation or incidence rates."""


@tool
def faers_top_adverse_events(
    drug: str | None = None,
    brand_name: str | None = None,
    limit: int = 10,
) -> list[dict] | str:
    """Get the most frequently reported adverse reactions for a drug from FDA FAERS.

    Args:
        drug: Generic drug name, e.g. "ibuprofen". A brand name also works.
        brand_name: Optional brand name, e.g. "Advil".
        limit: Number of reaction terms to return (1-100, default 10).

    Returns a list of {"term": MedDRA preferred term, "count": number of reports},
    sorted by count descending. An empty list means no matching reports.
    """
    try:
        return query_faers(drug=drug, brand_name=brand_name, limit=max(1, min(limit, 100)))
    except (ValueError, httpx.HTTPError) as e:
        # Return the error to the model so it can retry or explain, instead of crashing the run.
        return f"FAERS query failed: {e}"


def build_agent(model: str = DEFAULT_MODEL, temperature: float = 0.0):
    load_dotenv()
    api_key = os.environ.get("OPEN_ROUTER_KEY")
    if not api_key:
        raise RuntimeError("OPEN_ROUTER_KEY is not set (add it to .env)")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
    )
    return create_react_agent(llm, tools=[faers_top_adverse_events], prompt=SYSTEM_PROMPT)


def ask(question: str, agent=None) -> str:
    agent = agent or build_agent()
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the most reported adverse events for ibuprofen?"
    print(ask(q))
