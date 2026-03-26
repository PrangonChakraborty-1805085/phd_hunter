"""Central configuration — loaded once, imported everywhere."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AgentConfig:
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"{config.agent_url}"


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str
    groq_api_key: str
    tavily_api_key: str
    openrouter_url: str
    openrouter_key: str
    openrouter_model: str
    groq_model: str
    gemini_model: str
    gemini_api_key: str
    agent_host: str
    agent_url: str
    log_level: str

    # Individual agent configs
    ranking_agent: AgentConfig
    professor_finder: AgentConfig
    research_matcher: AgentConfig
    email_composer: AgentConfig

    @property
    def all_agent_urls(self) -> list[str]:
        """Used by orchestrator to discover all agents at startup."""
        return [
            self.ranking_agent.base_url,
            self.professor_finder.base_url,
            self.research_matcher.base_url,
            self.email_composer.base_url,
        ]


def load_config() -> AppConfig:
    """Load and validate config from environment variables."""
    llm_provider = os.getenv("LLM_PROVIDER", "openrouter")
    groq_key = os.getenv("GROQ_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    openrouter_url = os.getenv("OPENROUTER_URL", "")
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    if not groq_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com"
        )
    if not tavily_key:
        raise EnvironmentError(
            "TAVILY_API_KEY is not set. "
            "Get a free key at https://app.tavily.com"
        )
    
    if not openrouter_key or not openrouter_url or not openrouter_model:
        raise EnvironmentError(
            "OPENROUTER_URL, OPENROUTER_API_KEY, and OPENROUTER_MODEL must all be set. "
            "Get a free key at https://openrouter.ai"
        )
    if not llm_provider:
        raise EnvironmentError("LLM_PROVIDER is not set. Supported: openrouter, groq, ollama")

    host = os.getenv("AGENT_HOST", "localhost")
    agent_url = os.getenv("AGENT_URL", f"http://{host}:8000")

    return AppConfig(
        llm_provider=llm_provider,
        groq_api_key=groq_key,
        tavily_api_key=tavily_key,
        openrouter_url=openrouter_url,
        openrouter_key=openrouter_key,
        openrouter_model=openrouter_model,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        gemini_model=gemini_model,
        gemini_api_key=gemini_api_key,
        agent_host=host,
        agent_url=agent_url,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        ranking_agent=AgentConfig(
            host=host, port=int(os.getenv("RANKING_AGENT_PORT", "8000"))
        ),
        professor_finder=AgentConfig(
            host=host, port=int(os.getenv("PROFESSOR_FINDER_PORT", "8000"))
        ),
        research_matcher=AgentConfig(
            host=host, port=int(os.getenv("RESEARCH_MATCHER_PORT", "8000"))
        ),
        email_composer=AgentConfig(
            host=host, port=int(os.getenv("EMAIL_COMPOSER_PORT", "8000"))
        ),
    )


# Singleton — import this everywhere
config = load_config()
