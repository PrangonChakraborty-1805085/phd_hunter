"""
Professor Finder Agent — core LangGraph logic.

Responsibilities:
- Find faculty pages for a university + field
- Scrape professor names, emails, lab URLs, research areas
- Handle obfuscated emails gracefully (return dept page link as fallback)
- Return structured professor list
"""

import json
import re
from typing import Any

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from shared.a2a_helpers import web_search
from shared.config import config
from shared.logging import logger
from shared.utils import get_llm, message_preprocessor


SYSTEM_PROMPT = """You are a professor research assistant.
Your job is to find faculty members at universities who work in a specific research area.

Steps:
1. Use web_search_professors to find the department faculty page
2. Use fetch_page to read the faculty listing page
3. For each relevant professor, use web_search_professor_profile to find their personal page
4. Use fetch_page to get their email and research details

Return a JSON list of professors. Each professor object must have:
{
    "name": "Professor Name",
    "title": "Associate Professor",
    "university": "MIT",
    "department": "EECS",
    "email": "email@mit.edu or null if not found",
    "lab_url": "https://lab.mit.edu or null",
    "profile_url": "https://...",
    "research_areas": ["distributed systems", "cloud computing"],
    "recent_papers": ["Paper title 1", "Paper title 2"],
    "email_found": true
}

If email is obfuscated (e.g. name [at] domain [dot] com), reconstruct it.
If email truly cannot be found, set email to null and email_found to false.
Return ONLY valid JSON, no prose.
"""


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def web_search_professors(university: str, field: str, max_results: int = 3) -> str:
    """
    Search for the faculty directory page of a university department.
    Args:
        university: University name, e.g. 'MIT'
        field: Research field, e.g. 'software engineering'
        max_results: Number of search results to return
    """
    query = f"{university} {field} faculty directory professors research"
    try:
        web_search_result = web_search(query, max_results=max_results)
        return web_search_result
    except Exception as e:
        logger.error(f"web_search_professors failed: {e}")
        return json.dumps({"error": str(e)})


@tool
def fetch_page(url: str, max_length: int = 8000) -> str:
    """
    Fetches and parses a web page, returning clean text content.
    Args:
        url: Full URL to fetch
        max_length: Maximum characters to return (to stay within token limits)
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
            )
        }
        resp = httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "lxml")

        # Remove script, style, nav noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive whitespace
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        clean = "\n".join(lines)

        return clean[:max_length]

    except Exception as e:
        logger.error(f"fetch_page failed for {url}: {e}")
        return json.dumps({"error": str(e), "url": url})


@tool
def web_search_professor_profile(professor_name: str, university: str) -> str:
    """
    Search for a specific professor's personal/lab page.
    Args:
        professor_name: Full name, e.g. 'Tim Berners-Lee'
        university: University name, e.g. 'MIT'
    """
    query = f"{professor_name} {university} professor research lab homepage email"
    try:
        web_search_result = web_search(query, max_results=3)
        return web_search_result
    except Exception as e:
        logger.error(f"web_search_professor_profile failed: {e}")
        return json.dumps({"error": str(e)})


# ── Agent factory ─────────────────────────────────────────────────────────────

def create_professor_finder_agent():
    llm = get_llm()
    tools = [web_search_professors, fetch_page, web_search_professor_profile]
    return create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def run_professor_finder(query: str, context_id: str | None = None) -> str:
    """
    Run the professor finder agent.
    Query example: 'Find professors in software engineering at MIT, top 5'
    """
    agent = create_professor_finder_agent()
    messages = [HumanMessage(content=query)]

    logger.info(f"ProfessorFinder invoked | context_id={context_id} | query={query[:80]}")
    result = agent.invoke({"messages": messages})
    output = message_preprocessor(result["messages"][-1])
    logger.info(f"ProfessorFinder completed | output_length={len(output)}")
    return output
