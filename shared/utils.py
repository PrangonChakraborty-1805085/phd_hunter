from langchain_openrouter import ChatOpenRouter
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from shared.config import config
    
def get_llm(temperature: float = 0, max_tokens: int = 2048):
    """
    Returns a LangChain BaseChatModel
    """
    if config.llm_provider == "openrouter":
        return ChatOpenRouter(
            model=config.openrouter_model,
            api_key=config.openrouter_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif config.llm_provider == "groq":
        return ChatGroq(
            api_key=config.groq_api_key,
            model=config.groq_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout = None
        )
    elif config.llm_provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.gemini_model,
            google_api_key=config.gemini_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout = None
        )


def message_preprocessor(message: AIMessage | None) -> str:
    """Normalize LLM output into a single clean string.

    Accepts the raw LLM response object or content and returns normalized text.

    Returns:
        - "" when input is None or empty
        - stripped string when input is string
        - concat of text parts when input is list of dicts from Groq/other providers
    """
    if message is None:
        return ""

    content = getattr(message, "content", message)

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            return content["text"].strip()
        return str(content).strip()

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return " ".join(text_parts)

    return str(content).strip()