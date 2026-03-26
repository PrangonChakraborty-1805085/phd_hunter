"""Entry point for Email Composer A2A server.

Run with:
    python -m agents.email_composer
"""

from a2a.types import AgentSkill

from agents.email_composer.agent_executor import EmailComposerExecutor
from shared.a2a_helpers import build_a2a_server
from shared.config import config
from shared.logging import logger, setup_logger

setup_logger("email_composer", config.log_level)

SKILLS = [
    AgentSkill(
        id="compose_phd_email",
        name="PhD Cold Email Composer",
        description=(
            "Generates a professional, human-sounding cold email to a professor "
            "for PhD or Masters admission. Uses a fixed human-written template "
            "and fills in only the research-specific sections using real data "
            "from the professor's work. Avoids AI-sounding phrases. "
            "Returns subject line, email body with any remaining placeholders "
            "clearly marked for student review, and customization tips."
        ),
        tags=["email", "phd", "masters", "cold email", "application", "professor"],
        examples=[
            "Write a PhD email to Professor Smith at MIT about distributed systems",
            "Compose a cold email to Dr. Lee at Stanford for Fall 2025 admission",
            "Generate an email template for approaching a professor in NLP research",
        ],
    ),
]

if __name__ == "__main__":
    port = config.email_composer.port
    url = f"{config.agent_url}/email_composer"

    logger.info("=" * 50)
    logger.info("Email Composer Agent starting...")
    logger.info(f"AgentCard: {url}/.well-known/agent-card.json")
    logger.info("=" * 50)

    build_a2a_server(
        name="Email Composer Agent",
        description=(
            "Composes professional PhD/Masters cold email templates for contacting "
            "professors. Uses a human-written structural template filled with "
            "real professor research data. Returns ready-to-review email with "
            "placeholders for final student personalization."
        ),
        url=url,
        version="1.0.0",
        skills=SKILLS,
        executor=EmailComposerExecutor(),
        host="0.0.0.0",
        port=port,
    )
