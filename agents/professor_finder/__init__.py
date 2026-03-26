"""Professor Finder Agent — finds professors at universities by research field."""

from a2a.types import AgentSkill

from agents.professor_finder.agent_executor import ProfessorFinderExecutor
from shared.a2a_helpers import build_a2a_app
from shared.config import config
from shared.logging import logger, setup_logger

setup_logger("professor_finder", config.log_level)

SKILLS = [
    AgentSkill(
        id="find_professors",
        name="Professor Discovery",
        description=(
            "Finds faculty members at a specific university in a given research field. "
            "Scrapes department pages to extract professor names, emails, lab URLs, "
            "research areas, and recent publications. Handles obfuscated emails. "
            "Returns a structured list of professors with profile links."
        ),
        tags=["professors", "faculty", "research", "email", "university"],
        examples=[
            "Find professors in software engineering at MIT",
            "List AI researchers at Stanford with their emails",
            "Find top 5 distributed systems faculty at CMU",
            "Get professors working on NLP at University of Washington",
        ],
    ),
]

def get_professor_finder_agent():
    port = config.professor_finder.port
    url = f"{config.agent_url}/professor_finder"

    logger.info("=" * 50)
    logger.info("Professor Finder Agent starting...")
    logger.info(f"AgentCard: {url}/.well-known/agent-card.json")
    logger.info("=" * 50)

    return build_a2a_app(
        name="Professor Finder Agent",
        description=(
            "Finds and profiles faculty members at universities by research field. "
            "Extracts emails, lab pages, research areas, and recent papers. "
            "Falls back to department page link when email is not publicly listed."
        ),
        url=url,
        version="1.0.0",
        skills=SKILLS,
        executor=ProfessorFinderExecutor(),
        host="0.0.0.0",
        port=port,
    )