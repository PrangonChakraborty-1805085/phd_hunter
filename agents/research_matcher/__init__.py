"""Research Matcher Agent — scores alignment between professor work and student interests."""

from a2a.types import AgentSkill

from agents.research_matcher.agent_executor import ResearchMatcherExecutor
from shared.a2a_helpers import build_a2a_app
from shared.config import config
from shared.logging import logger, setup_logger

setup_logger("research_matcher", config.log_level)

SKILLS = [
    AgentSkill(
        id="match_research_interests",
        name="Research Interest Alignment",
        description=(
            "Reads a professor's lab page and recent publications, then scores how "
            "well their research aligns with a student's stated interests (0.0–1.0). "
            "Returns a 2–3 sentence alignment summary, list of matching topics, "
            "and the most relevant paper to mention in a cold email. "
            "Useful before writing a PhD cold email to personalize the research section."
        ),
        tags=["research", "alignment", "professor", "publications", "matching"],
        examples=[
            "Match Dr. John Doe at MIT with interests in distributed systems and cloud",
            "Does Professor Jane Smith's work align with NLP and transformer research?",
            "Score alignment between Prof. Lee at Stanford and student interested in CV",
        ],
    ),
]

def get_research_matcher_agent():
    port = config.research_matcher.port
    url = f"{config.agent_url}/research_matcher"

    logger.info("=" * 50)
    logger.info("Research Matcher Agent starting...")
    logger.info(f"AgentCard: {url}/.well-known/agent-card.json")
    logger.info("=" * 50)

    return build_a2a_app(
        name="Research Matcher Agent",
        description=(
            "Analyzes a professor's research publications and lab page to score "
            "alignment with a student's research interests. Returns alignment score, "
            "summary, and the best paper to reference in a cold email."
        ),
        url=url,
        version="1.0.0",
        skills=SKILLS,
        executor=ResearchMatcherExecutor(),
        host="0.0.0.0",
        port=port,
    )