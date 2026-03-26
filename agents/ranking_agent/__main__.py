"""
Entry point for the Ranking Agent A2A server.

Run with:
    python -m agents.ranking_agent
"""

from a2a.types import AgentSkill

from agents.ranking_agent.agent_executor import RankingAgentExecutor
from shared.a2a_helpers import build_a2a_server
from shared.config import config
from shared.logging import logger, setup_logger

setup_logger("ranking_agent", config.log_level)


SKILLS = [
    AgentSkill(
        id="university_rankings",
        name="University Rankings Lookup",
        description=(
            "Finds and ranks universities by field and country. "
            "Supports CS subfields (software engineering, AI, systems, NLP, etc.) "
            "using CSRankings data and other fields (Civil, Environmental, etc.) "
            "using web search. Can filter by top-N and minimum scholarship coverage. "
            "Can intersect rankings across multiple fields."
        ),
        tags=["rankings", "universities", "cs", "engineering", "scholarship"],
        examples=[
            "Top 20 CS universities in US",
            "Best software engineering programs in Canada",
            "Universities in top 20 for both CS and Civil Engineering in US",
            "Top 15 AI universities with full PhD funding in US",
        ],
    ),
]


if __name__ == "__main__":
    port = config.ranking_agent.port
    url = f"http://{config.agent_host}:{port}"

    logger.info("=" * 50)
    logger.info("Ranking Agent starting...")
    logger.info(f"AgentCard: {url}/.well-known/agent-card.json")
    logger.info("=" * 50)

    build_a2a_server(
        name="Ranking Agent",
        description=(
            "Finds university rankings by field (CS, Civil, Environmental, etc.) "
            "and country. Filters by scholarship/funding coverage. "
            "Can compare rankings across multiple fields simultaneously."
        ),
        url=url,
        version="1.0.0",
        skills=SKILLS,
        executor=RankingAgentExecutor(),
        host="0.0.0.0",
        port=port,
    )
