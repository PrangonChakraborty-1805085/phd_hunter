"""
Research Matcher Agent — core LangGraph logic.

Responsibilities:
- Fetch professor's lab/profile page and recent publications
- Compare professor's research themes to student's stated interests
- Produce a 0.0–1.0 alignment score + 2–3 sentence summary of WHY they match
- Extract specific paper titles / project names the student can reference in email
"""

import json

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from shared.a2a_helpers import web_search
from shared.config import config
from shared.utils import get_llm, message_preprocessor
from shared.logging import logger


SYSTEM_PROMPT = """You are a research alignment analyst for PhD applications.

Given a professor's profile URL and a student's research interests, your job is:
1. Use fetch_professor_page to read the professor's lab/profile page
2. Use search_professor_papers to find their recent publications
3. Analyze how well the professor's work matches the student's interests
4. Return a structured JSON result

Your output MUST be valid JSON in exactly this format:
{
    "professor_name": "Dr. Jane Smith",
    "university": "MIT",
    "alignment_score": 0.85,
    "alignment_summary": "2-3 sentences explaining why this is a good match, 
                          mentioning specific papers or projects by name.",
    "matching_topics": ["topic1", "topic2", "topic3"],
    "professor_recent_work": "1-2 sentence summary of what the professor is 
                              currently working on, based on recent papers.",
    "suggested_paper_to_mention": "Title of their most relevant paper the 
                                    student should reference in the email"
}

Scoring guide:
- 0.9–1.0: Direct match, professor's primary focus IS the student's interest
- 0.7–0.89: Strong overlap, significant shared themes
- 0.5–0.69: Partial match, some overlap but different focus areas
- Below 0.5: Weak match, do not recommend

Return ONLY the JSON object, no prose before or after.
"""


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def fetch_professor_page(url: str) -> str:
    """
    Fetches a professor's lab or profile page and returns clean text.
    Args:
        url: Professor's profile or lab page URL
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
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Focus on content-rich sections
        content_tags = soup.find_all(
            ["p", "h1", "h2", "h3", "h4", "li", "article", "section"]
        )
        text_parts = []
        for tag in content_tags:
            t = tag.get_text(strip=True)
            if len(t) > 30:  # skip very short fragments
                text_parts.append(t)

        clean = "\n".join(text_parts)
        return clean[:6000]  # keep within Groq token budget

    except Exception as e:
        logger.error(f"fetch_professor_page failed for {url}: {e}")
        return json.dumps({"error": str(e), "url": url})


@tool
def search_professor_papers(professor_name: str, university: str) -> str:
    """
    Searches for a professor's recent publications via web search.
    Args:
        professor_name: e.g. 'John Doe'
        university: e.g. 'Stanford'
    """
    query = (
        f"{professor_name} {university} recent publications research papers "
        "2022 2023 2024 Google Scholar"
    )
    try:
        search_results = web_search(query, max_results=2)
        return json.dumps(search_results)
    except Exception as e:
        logger.error(f"search_professor_papers failed: {e}")
        return json.dumps({"error": str(e)})


# ── Agent factory ─────────────────────────────────────────────────────────────

def create_research_matcher_agent():
    llm = get_llm()
    tools = [fetch_professor_page, search_professor_papers]
    return create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def run_research_matcher(
    professor_name: str,
    university: str,
    profile_url: str,
    student_interests: list[str],
    context_id: str | None = None,
) -> str:
    """
    Match a professor's research to student interests.

    Returns JSON string with alignment_score, summary, matching_topics, etc.
    """
    interests_str = ", ".join(student_interests)
    query = (
        f"Professor: {professor_name} at {university}\n"
        f"Profile URL: {profile_url}\n"
        f"Student interests: {interests_str}\n\n"
        "Analyze alignment and return the JSON result."
    )

    agent = create_research_matcher_agent()
    logger.info(
        f"ResearchMatcher invoked | professor={professor_name} | "
        f"interests={interests_str[:60]}"
    )

    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    output = message_preprocessor(result["messages"][-1])
    logger.info(f"ResearchMatcher completed | output_length={len(output)}")
    return output
