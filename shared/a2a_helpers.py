"""
Reusable helpers that wrap the a2a-sdk boilerplate.

Every agent's __main__.py uses `build_a2a_server()`.
The orchestrator uses `call_agent()`.
"""

import asyncio
import json
import uuid
from typing import Any

import httpx
import uvicorn
from a2a.client import A2ACardResolver, A2AClient
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MessageSendParams,
    SendMessageRequest,
    SendMessageSuccessResponse,
    GetTaskSuccessResponse,
    JSONRPCErrorResponse,
    Task,
    Message,
    GetTaskRequest,
)

from shared.logging import logger
from shared.config import config

# ── Server builder ────────────────────────────────────────────────────────────

def build_a2a_server(
    *,
    name: str,
    description: str,
    url: str,
    version: str,
    skills: list[AgentSkill],
    executor,                   # AgentExecutor subclass instance
    host: str = "0.0.0.0",
    port: int,
):
    """
    Builds and runs a complete A2A-compliant uvicorn server.
    
    Usage in each agent's __main__.py:
        build_a2a_server(name="Ranking Agent", ..., executor=RankingAgentExecutor())
    """
    agent_card = AgentCard(
        name=name,
        description=description,
        url=url,
        version=version,
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build()

    logger.info(f"Starting {name} on {host}:{port}")
    logger.info(f"AgentCard available at: http://{host}:{port}/.well-known/agent-card.json")

    uvicorn.run(app, host=host, port=port, log_level="warning")

    

def build_a2a_app(
    *,
    name: str,
    description: str,
    url: str,
    version: str,
    skills: list[AgentSkill],
    executor,                   # AgentExecutor subclass instance
    host: str = "0.0.0.0",
    port: int,
):
    """
    Builds and runs a complete A2A-compliant uvicorn server.
    
    Usage in each agent's __main__.py:
        build_a2a_server(name="Ranking Agent", ..., executor=RankingAgentExecutor())
    """
    agent_card = AgentCard(
        name=name,
        description=description,
        url=url,
        version=version,
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build()

    logger.info(f"Starting {name} on {url}")
    logger.info(f"AgentCard available at: {url}/.well-known/agent-card.json")
    return app


# ── Client helpers ────────────────────────────────────────────────────────────

async def wait_for_result(client: A2AClient, task: Task):
    while True:
        task_request = GetTaskRequest(id=uuid.uuid4().hex, params={"id": task.id, "metadata": task.metadata})
        task_response = await client.get_task(task_request)

        if isinstance(task_response.root, JSONRPCErrorResponse):
            raise Exception(f"Error fetching task result: {task_response.root.error}")

        _task = task_response.root.result
        current_task_status = _task.status.state.value if _task.status and _task.status.state else "unknown"
        logger.info(f"Polling task {_task.id} | current status: {current_task_status}")

        if current_task_status == "completed":
            return _task

        elif current_task_status == "failed":
            raise Exception("Task failed")

        await asyncio.sleep(1)

async def discover_agent(httpx_client: httpx.AsyncClient, base_url: str, agent_id: str) -> AgentCard:
    """
    Fetches /.well-known/agent-card.json from an agent's base URL.
    Called ONCE at orchestrator startup per agent.
    """
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url, agent_card_path=f"{agent_id}/.well-known/agent-card.json")
    return await resolver.get_agent_card()


async def call_agent(
    httpx_client: httpx.AsyncClient,
    agent_url: str,
    message_text: str,
    context_id: str | None = None,
    task_id: str | None = None,
) -> tuple[str, str, str]:
    """
    Send a message to an A2A agent and return (response_text, context_id, task_id).

    - On first call: context_id=None, task_id=None  → agent creates them
    - On follow-up:  pass returned context_id + task_id to continue conversation

    Returns:
        (response_text, context_id, task_id)
    """
    # Step 1: resolve AgentCard (fetches /.well-known/agent-card.json)
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=agent_url, agent_card_path="/.well-known/agent-card.json")
    agent_card = await resolver.get_agent_card()

    # Normalize agent_url to always have trailing slash
    if not agent_card.url.endswith("/"):
        agent_card.url = agent_card.url + "/"

    # Step 2: build client from the resolved card
    client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

    # Step 3: build the typed SendMessageRequest
    message_payload: dict[str, Any] = {
        "role": "user",
        "parts": [{"kind": "text", "text": message_text}],
        "messageId": uuid.uuid4().hex,
    }
    if context_id:
        message_payload["context_id"] = context_id

    send_params: dict[str, Any] = {"message": message_payload}
    if task_id:
        send_params["id"] = task_id

    request = SendMessageRequest(
        id=uuid.uuid4().hex,
        params=MessageSendParams(**send_params),
    )
    logger.info(f"request to send : {request}")
    response = await client.send_message(request)

    # response.root is either SendMessageSuccessResponse or JSONRPCErrorResponse
    if not isinstance(response.root, SendMessageSuccessResponse):
        error = getattr(response.root, "error", "unknown error")
        logger.error(f"A2A error response from {agent_url}: {error}")
        return f"[ERROR] {error}", context_id or "", task_id or ""
    
    result = response.root.result
    if isinstance(result, Task):
        # LangGraph agents always return a Task
        returned_task_id = result.id
        returned_context_id = result.context_id or context_id or uuid.uuid4().hex

        # Check for INPUT_REQUIRED state
        if result.status and result.status.state.value == "input-required":
            question = _extract_text(result.status.message) if result.status.message else "<no message>"
            return f"[INPUT_REQUIRED] {question}", returned_context_id, returned_task_id
        final_task = await wait_for_result(client, result)
        returned_task_id = final_task.id
        returned_context_id = final_task.context_id or returned_context_id
        final_response_text = _extract_text(final_task)
        logger.info(f"result from agent in Task : {final_response_text}")
    
    elif isinstance(result, Message):
        logger.info(f"result from agent in Message : {result}")
        # Simple agents return a Message directly
        returned_task_id = task_id or uuid.uuid4().hex
        returned_context_id = context_id or uuid.uuid4().hex
        final_response_text = _extract_text(result)

    logger.debug(
        f"Agent call → url={agent_url} | "
        f"final response = {final_response_text[:80]} | "
        f"context_id={returned_context_id} | "
        f"task_id={returned_task_id}"
    )
    return final_response_text, returned_context_id, returned_task_id


def _extract_text(obj: Task | Message) -> str:
    """Extracts question or response text from a Task or Message object."""
    if isinstance(obj, Task):
        if obj.artifacts:
            for artifact in obj.artifacts:
                if artifact.parts:
                    for part in artifact.parts:
                        # part is a Part object with a .root that is TextPart/DataPart/etc
                        root = getattr(part, "root", part)
                        text = getattr(root, "text", None)
                        if text:
                            return text
            
    elif isinstance(obj, Message):
        if obj.parts:
            for part in obj.parts:
                # part is a Part object with a .root that is TextPart/DataPart/etc
                root = getattr(part, "root", part)
                text = getattr(root, "text", None)
                if text:
                    return text
        return ""
    return ""

def web_search(query: str, max_results: int = 2) -> str:
    """
    Helper function to perform a web search using the Tavily API.
    Returns a JSON string of search results.
    """
    from tavily import TavilyClient
    try:
        client = TavilyClient(api_key=config.tavily_api_key)
        response = client.search(query, search_depth="advanced", max_results=max_results)
        return json.dumps(response.get("results", []))
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return json.dumps({"error": str(e)})
