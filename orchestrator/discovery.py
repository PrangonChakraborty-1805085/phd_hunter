"""
Agent discovery — runs ONCE at orchestrator startup.

Fetches /.well-known/agent-card.json from each agent URL,
stores AgentCards in memory, and builds the system prompt
string that describes all available agents to the LLM.
"""

import asyncio
from dataclasses import dataclass

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import AgentCard

from shared.config import config
from shared.logging import logger


@dataclass
class DiscoveredAgent:
    name: str
    url: str
    description: str
    skills_text: str     # formatted for LLM system prompt
    card: AgentCard


# Module-level registry — populated by discover_all_agents()
_registry: dict[str, DiscoveredAgent] = {}


async def discover_all_agents() -> dict[str, DiscoveredAgent]:
    """
    Fetch AgentCards from all agent URLs in config.
    Called ONCE at orchestrator startup.
    Returns dict keyed by a short agent id (e.g. 'ranking_agent').
    """
    global _registry

    agent_ids = [
        "ranking_agent",
        "professor_finder",
        "research_matcher",
        "email_composer",
    ]
    urls = config.all_agent_urls

    logger.info("Discovering agents...")
    discovered = {}

    async with httpx.AsyncClient(timeout=10) as http:
        for agent_id, url in zip(agent_ids, urls):
            try:
                resolver = A2ACardResolver(httpx_client=http, base_url=url, agent_card_path=f"/{agent_id}/.well-known/agent-card.json")
                card = await resolver.get_agent_card()

                skills_text = _format_skills(card)
                agent = DiscoveredAgent(
                    name=card.name,
                    url=card.url,
                    description=card.description or "",
                    skills_text=skills_text,
                    card=card,
                )
                discovered[agent_id] = agent
                logger.info(f"  ✓ {agent_id}: {card.name} @ {url}/{agent_id}")

            except Exception as e:
                logger.warning(f"  ✗ Could not reach {agent_id} @ {url}/{agent_id}: {e}")
                # Add a placeholder so orchestrator knows the agent exists
                # but is currently unreachable
                discovered[agent_id] = DiscoveredAgent(
                    name=agent_id,
                    url=url,
                    description="[UNAVAILABLE]",
                    skills_text="[UNAVAILABLE — agent not running]",
                    card=None,
                )

    _registry = discovered
    logger.info(f"Discovery complete. {len(discovered)} agents registered.")
    return discovered


def get_registry() -> dict[str, DiscoveredAgent]:
    """Return the cached agent registry."""
    return _registry


def build_agents_prompt_section(registry: dict[str, DiscoveredAgent]) -> str:
    """
    Serializes all AgentCards into a plain-text description
    that gets injected into the orchestrator's system prompt.
    """
    lines = ["You have access to these specialized agents:\n"]
    for agent_id, agent in registry.items():
        status = "" if agent.description != "[UNAVAILABLE]" else " [CURRENTLY UNAVAILABLE]"
        lines.append(f"## {agent_id}{status}")
        lines.append(f"Name: {agent.name}")
        lines.append(f"URL: {agent.url}")
        lines.append(f"Description: {agent.description}")
        lines.append(f"Skills:\n{agent.skills_text}")
        lines.append("")
    return "\n".join(lines)


def _format_skills(card: AgentCard) -> str:
    if not card or not card.skills:
        return "  (no skills listed)"
    lines = []
    for skill in card.skills:
        lines.append(f"  - {skill.name}: {skill.description}")
        if skill.examples:
            examples = " | ".join(skill.examples[:2])
            lines.append(f"    Examples: {examples}")
    return "\n".join(lines)
