"""
Ranking Agent — core LangGraph logic.

Responsibilities:
- Query CSRankings CSV (for CS subfields)
- Web search QS/US News for non-CS fields
- Filter by country, top-N, scholarship coverage
- Handle multi-field intersection queries (e.g. top-20 CS AND Civil)
"""

import json
from typing import Any

import httpx
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from shared.config import config
from shared.logging import logger
from shared.a2a_helpers import web_search
from shared.utils import get_llm, message_preprocessor

# ── Constants ─────────────────────────────────────────────────────────────────

CSRANKINGS_CSV_URL = (
    "https://raw.githubusercontent.com/emeryberger/CSrankings/"
    "gh-pages/csrankings.csv"
)

# CS subfield aliases that map to CSRankings column names
CS_FIELD_MAP = {
    "cs": None,              # all CS → use entire CSV
    "software engineering": "softeng",
    "machine learning": "mlmining",
    "artificial intelligence": "ai",
    "computer vision": "vision",
    "nlp": "nlp",
    "natural language processing": "nlp",
    "systems": "sys",
    "networks": "networks",
    "security": "sec",
    "databases": "db",
    "programming languages": "plan",
    "hci": "hci",
    "robotics": "robotics",
    "computer architecture": "arch",
    "theory": "theory",
}

FULLY_FUNDED_KEYWORDS = [
    "full funding", "fully funded", "fellowship", "stipend",
    "tuition waiver", "research assistantship", "RA funding",
]

SYSTEM_PROMPT = """You are a university ranking research assistant. 
Your job is to find accurate, up-to-date university rankings.

When given a ranking query:
1. For CS subfields: use the csrankings_lookup tool
2. For other fields (Civil, Environmental, etc.): use the web_search_rankings tool
3. For scholarship info: use the web_search_scholarships tool
4. Always return results as a valid JSON list

Be concise. Return only the data, no prose explanations.
For ranking queries, return JSON like:
[{"name": "MIT", "rank": 1, "field": "cs", "country": "US", "website": "https://mit.edu"}]
"""


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def csrankings_lookup(field: str, country: str, top_n: int = 10) -> str:
    """
    Look up university rankings from CSRankings data for a CS subfield.
    Args:
        field: CS subfield (e.g. 'software engineering', 'ai', 'systems')
        country: Country filter (e.g. 'US', 'UK', 'Canada')
        top_n: How many top universities to return
    """
    try:
        logger.info(f"Fetching CSRankings CSV for field={field}")
        df = pd.read_csv(CSRANKINGS_CSV_URL)

        # Map field name to CSV column
        column = CS_FIELD_MAP.get(field.lower())

        if column and column in df.columns:
            # Filter to specific subfield (non-zero score)
            df = df[df[column] > 0]

        # Filter by country if column exists
        if "countryabbrv" in df.columns:
            df = df[df["countryabbrv"].str.upper() == country.upper()]
        elif "region" in df.columns:
            df = df[df["region"].str.contains(country, case=False, na=False)]

        # Aggregate by institution
        if "institution" in df.columns:
            top = (
                df.groupby("institution")
                .size()
                .reset_index(name="faculty_count")
                .sort_values("faculty_count", ascending=False)
                .head(top_n)
            )
            results = [
                {
                    "name": row["institution"],
                    "rank": i + 1,
                    "field": field,
                    "country": country,
                    "faculty_count": int(row["faculty_count"]),
                }
                for i, row in top.iterrows()
            ]
            return json.dumps(results)

        return json.dumps({"error": "Could not parse CSRankings data"})

    except Exception as e:
        logger.error(f"CSRankings lookup failed: {e}")
        return json.dumps({"error": str(e)})


@tool
def web_search_rankings(query: str) -> str:
    """
    Search the web for university rankings when CSRankings doesn't cover the field.
    Args:
        query: Search query, e.g. 'top 20 civil engineering universities US QS ranking 2024'
    """
    return web_search(query)  # delegate to shared Tavily search helper


@tool
def web_search_scholarships(university_name: str, field: str) -> str:
    """
    Search for PhD scholarship/funding information at a specific university.
    Args:
        university_name: e.g. 'MIT'
        field: e.g. 'computer science'
    """
    query = (
        f"{university_name} PhD {field} funding scholarship "
        "stipend tuition waiver international students fully funded"
    )
    return web_search(query)  # reuse Tavily search helper


# ── Agent factory ─────────────────────────────────────────────────────────────

def create_ranking_agent():
    """Returns a LangGraph ReAct agent with ranking tools."""
    llm = get_llm()
    tools = [csrankings_lookup, web_search_rankings, web_search_scholarships]
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


def run_ranking_agent(query: str, context_id: str | None = None) -> str:
    """
    Invoke the ranking agent with a natural language query.
    Returns the agent's text response.
    """
    agent = create_ranking_agent()
    messages = [HumanMessage(content=query)]

    logger.info(f"Ranking agent invoked | context_id={context_id} | query={query[:80]}\n\n")

    result = agent.invoke({"messages": messages})
    output = message_preprocessor(result["messages"][-1])
    logger.info(f"Ranking agent completed | output_length={len(output)}\n\n")
    return output
