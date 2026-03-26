"""
Orchestrator LangGraph — the brain of PhD Hunter.

Graph topology:
  START → orchestrator_node → [conditional edges] → agent nodes → orchestrator_node
                                                  ↘ END (when done)

The orchestrator_node runs in a loop. Each iteration:
1. LLM reads the full state (query + collected data so far)
2. LLM decides next action: call an agent, call agents in parallel, or respond
3. Router sends execution to the right branch
4. Agent results are merged back into state
5. Loop repeats until LLM outputs "respond_to_user"
"""

import asyncio
import json
import re
from typing import Annotated, Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send
from typing_extensions import TypedDict

from orchestrator.discovery import DiscoveredAgent, build_agents_prompt_section, get_registry
from shared.a2a_helpers import call_agent
from shared.utils import get_llm, message_preprocessor
from shared.logging import logger


# ── State ─────────────────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    # Core conversation
    messages: Annotated[list, add_messages]
    user_query: str

    # Data collected from agents (accumulated across loop iterations)
    ranking_data: str
    professor_data: str
    match_data: str
    email_data: str

    # Orchestrator decisions
    next_action: str          # e.g. "call_ranking_agent" or "respond_to_user"
    agents_called: list[str]  # tracking for graph visualization

    # A2A context tracking for multi-turn (keyed by agent_id)
    context_ids: dict[str, str]
    task_ids: dict[str, str]

    # Final answer
    final_response: str


# ── System prompt factory ─────────────────────────────────────────────────────

def _build_system_prompt(registry: dict[str, DiscoveredAgent]) -> str:
    agents_section = build_agents_prompt_section(registry)

    return f"""You are an intelligent orchestrator for "PhD Hunter" — an AI assistant
that helps students find PhD/Masters programs and approach professors.

{agents_section}

DECISION RULES:
- Analyze the user query and the data already collected (shown in context)
- Decide which agent(s) to call next based on what is still missing
- You can call agents SEQUENTIALLY (one at a time) or IN PARALLEL (when independent)
- Do NOT call an agent if you already have sufficient data from it for this query
- When you have everything needed, output "respond_to_user" with a complete answer

OUTPUT FORMAT — respond with ONLY one of these exact formats:

1. Simple greetings/asking/user clarification:
   ACTION: respond_to_user
   RESPONSE: <your message to the user, e.g. asking for clarification or saying hello
2. Single agent call:
   ACTION: call_ranking_agent
   MESSAGE: <exact message to send to the agent>

3. Parallel agent calls:
   ACTION: call_parallel
   AGENTS: ranking_agent,professor_finder
   MESSAGE_ranking_agent: <message for ranking agent>
   MESSAGE_professor_finder: <message for professor finder>

4. Done — answer the user:
   ACTION: respond_to_user
   RESPONSE: <your complete, well-formatted answer to the user>

CONCISENESS RULES (important for free-tier Groq rate limits):
- Keep agent messages short and specific — under 100 words each
- For ranking queries: specify field, country, and top-N clearly
- For professor queries: specify university, field, and max count
- For match queries: include professor name, URL, and student interests on separate lines
- For email queries: send a JSON object with all required fields

STUDENT PROFILE (use when composing emails or matching):
- If the user has not provided their profile, use placeholders like <Your Name>
- Ask for profile details only if writing an email and no profile was given
"""



# ── Orchestrator node (the loop brain) ────────────────────────────────────────

def orchestrator_node(state: OrchestratorState) -> dict:
    """
    Core decision node. Runs on every loop iteration.
    Reads full state, asks LLM what to do next.
    """
    logger.info("------------------ in orchestrator node -------------------\n")
    registry = get_registry()
    system_prompt = _build_system_prompt(registry)

    # Build context summary of what we've collected so far
    context_parts = []
    if state.get("ranking_data"):
        context_parts.append(f"RANKING DATA COLLECTED:\n{state['ranking_data'][:800]}")
    if state.get("professor_data"):
        context_parts.append(f"PROFESSOR DATA COLLECTED:\n{state['professor_data'][:800]}")
    if state.get("match_data"):
        context_parts.append(f"RESEARCH MATCH DATA:\n{state['match_data'][:600]}")
    if state.get("email_data"):
        context_parts.append(f"EMAIL DRAFT:\n{state['email_data'][:600]}")

    collected = "\n\n".join(context_parts) if context_parts else "Nothing collected yet."

    llm_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"USER QUERY: {state['user_query']}\n\n"
            f"COLLECTED SO FAR:\n{collected}\n\n"
            f"AGENTS ALREADY CALLED: {state.get('agents_called', [])}\n\n"
            "Decide the next action."
        )),
    ]

    llm = get_llm()
    response = llm.invoke(llm_messages)
    logger.info(f"aimessage from orchestrator {response}")
    decision = message_preprocessor(response)

    logger.info(f"Orchestrator decision:\n{decision}")

    return {
        "next_action": decision,
        "messages": [AIMessage(content=decision)],
    }


# ── Router ────────────────────────────────────────────────────────────────────

def router(state: OrchestratorState) -> str:
    """
    Reads the LLM's ACTION line and routes to the right node.
    """
    action = state.get("next_action", "")

    if "ACTION: call_parallel" in action:
        return "parallel_dispatcher"
    if "ACTION: call_ranking_agent" in action:
        return "invoke_ranking_agent"
    if "ACTION: call_professor_finder" in action:
        return "invoke_professor_finder"
    if "ACTION: call_research_matcher" in action:
        return "invoke_research_matcher"
    if "ACTION: call_email_composer" in action:
        return "invoke_email_composer"
    if "ACTION: respond_to_user" in action:
        return "build_response"

    # Safety: if LLM output is malformed, go to response
    logger.warning(f"Unrecognised action, defaulting to respond: {action[:100]}")
    return "build_response"


# ── Agent invocation nodes ─────────────────────────────────────────────────────

def _extract_message(decision: str, prefix: str = "MESSAGE:") -> str:
    """Pull the message body from the LLM's decision string."""
    for line in decision.splitlines():
        if line.strip().startswith(prefix):
            return line.split(prefix, 1)[1].strip()
    # Fallback: return everything after the ACTION line
    lines = decision.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("ACTION:") and i + 1 < len(lines):
            return "\n".join(lines[i + 1:]).strip()
    return decision


async def _do_call(agent_id: str, message: str, state: OrchestratorState) -> tuple[str, str, str]:
    """Fire one A2A call, honouring existing context_id/task_id for multi-turn."""
    registry = get_registry()
    # logger.info(f"agents registry is: {registry}")
    agent = registry.get(agent_id)
    if not agent or not agent.url:
        return f"[{agent_id} is unavailable]", "", ""

    context_id = state.get("context_ids", {}).get(agent_id)
    task_id = state.get("task_ids", {}).get(agent_id)

    async with httpx.AsyncClient(timeout=60) as http_client:
        response_text, new_ctx, new_task = await call_agent(
            http_client, agent.url, message, context_id, task_id
        )
    return response_text, new_ctx, new_task


def invoke_ranking_agent(state: OrchestratorState) -> dict:
    logger.info("-------------------- in invoke_ranking_agent node -------------------\n")
    message = _extract_message(state["next_action"])
    logger.info(f"Calling ranking_agent | msg={message[:80]}")
    result, ctx, tid = asyncio.run(
        _do_call("ranking_agent", message, state)
    )
    return {
        "ranking_data": result,
        "agents_called": state.get("agents_called", []) + ["ranking_agent"],
        "context_ids": {**state.get("context_ids", {}), "ranking_agent": ctx},
        "task_ids": {**state.get("task_ids", {}), "ranking_agent": tid},
    }


def invoke_professor_finder(state: OrchestratorState) -> dict:
    logger.info("-------------------- in invoke_professor_finder node -------------------\n")
    message = _extract_message(state["next_action"])
    logger.info(f"Calling professor_finder | msg={message[:80]}")
    result, ctx, tid = asyncio.run(
        _do_call("professor_finder", message, state)
    )
    return {
        "professor_data": result,
        "agents_called": state.get("agents_called", []) + ["professor_finder"],
        "context_ids": {**state.get("context_ids", {}), "professor_finder": ctx},
        "task_ids": {**state.get("task_ids", {}), "professor_finder": tid},
    }


def invoke_research_matcher(state: OrchestratorState) -> dict:
    logger.info("-------------------- in invoke_research_matcher node -------------------\n")
    message = _extract_message(state["next_action"])
    logger.info(f"Calling research_matcher | msg={message[:80]}")
    result, ctx, tid = asyncio.run(
        _do_call("research_matcher", message, state)
    )
    return {
        "match_data": result,
        "agents_called": state.get("agents_called", []) + ["research_matcher"],
        "context_ids": {**state.get("context_ids", {}), "research_matcher": ctx},
        "task_ids": {**state.get("task_ids", {}), "research_matcher": tid},
    }


def invoke_email_composer(state: OrchestratorState) -> dict:
    logger.info("-------------------- in invoke_email_composer node -------------------\n")
    message = _extract_message(state["next_action"])
    logger.info(f"Calling email_composer | msg={message[:80]}")
    result, ctx, tid = asyncio.run(
        _do_call("email_composer", message, state)
    )
    return {
        "email_data": result,
        "agents_called": state.get("agents_called", []) + ["email_composer"],
        "context_ids": {**state.get("context_ids", {}), "email_composer": ctx},
        "task_ids": {**state.get("task_ids", {}), "email_composer": tid},
    }


# ── Parallel dispatcher ────────────────────────────────────────────────────────

def parallel_dispatcher(state: OrchestratorState) -> Command:
    """
    Parses the LLM's parallel action and fans out to multiple agent nodes.
    Uses LangGraph's Send primitive for true concurrent execution.
    """
    logger.info("-------------------- in parallel_dispatcher node -------------------\n")
    decision = state["next_action"]

    # Parse AGENTS line
    agents_line = ""
    messages: dict[str, list[str]] = {}
    for line in decision.splitlines():
        line = line.strip()
        if line.startswith("AGENTS:"):
            agents_line = line.replace("AGENTS:", "").strip()
        elif line.startswith("MESSAGE_"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                agent_key = parts[0].replace("MESSAGE_", "").strip()
                messages.setdefault(agent_key, []).append(parts[1].strip())

    agent_ids = [a.strip() for a in agents_line.split(",") if a.strip()]
    logger.info(f"Parallel dispatch to: {agent_ids}")

    node_map = {
        "ranking_agent": "invoke_ranking_agent",
        "professor_finder": "invoke_professor_finder",
        "research_matcher": "invoke_research_matcher",
        "email_composer": "invoke_email_composer",
    }

    sends: list[Send] = []
    for index, agent_id in enumerate(agent_ids):
        node = node_map.get(agent_id)
        if not node:
            continue

        agent_messages = messages.get(agent_id, [])
        # if we have multiple agent entries and multiple messages, match by index,
        # else use the first message or fallback to generic query form.
        if index < len(agent_messages):
            msg = agent_messages[index]
        elif agent_messages:
            msg = agent_messages[-1]
        else:
            msg = f"Process query: {state['user_query']}"

        modified_state = {
            **state,
            "next_action": f"ACTION: call_{agent_id}\nMESSAGE: {msg}",
        }
        sends.append(Send(node, modified_state))

    return Command(goto=sends)


# ── Response builder ───────────────────────────────────────────────────────────

def build_response(state: OrchestratorState) -> dict:
    """
    Extracts the final response from the LLM's decision, or synthesizes
    one from all collected data if the LLM skipped to respond_to_user early.
    """
    logger.info("-------------------- in build_response node -------------------\n")
    decision = state.get("next_action", "")

    # Try to extract RESPONSE: block
    response_text = ""
    in_response = False
    for line in decision.splitlines():
        if line.strip().startswith("RESPONSE:"):
            response_text = line.split("RESPONSE:", 1)[1].strip()
            in_response = True
        elif in_response:
            response_text += "\n" + line

    # If no explicit response, synthesize from collected data
    if not response_text.strip():
        parts = []
        if state.get("email_data"):
            try:
                email = json.loads(state["email_data"])
                parts.append(
                    f"**Subject:** {email.get('subject', '')}\n\n"
                    f"{email.get('body', '')}\n\n"
                    f"*Note: {email.get('notes', '')}*"
                )
            except Exception:
                parts.append(state["email_data"])
        if state.get("match_data"):
            parts.append(f"**Research Match Analysis:**\n{state['match_data']}")
        if state.get("professor_data"):
            parts.append(f"**Professors Found:**\n{state['professor_data'][:1000]}")
        if state.get("ranking_data"):
            parts.append(f"**University Rankings:**\n{state['ranking_data'][:800]}")

        response_text = "\n\n---\n\n".join(parts) if parts else "Task complete."

    logger.info(f"Final response built | length={len(response_text)}")
    logger.info(f"final response is : {response_text}")
    return {
        "final_response": response_text.strip(),
        "messages": [AIMessage(content=response_text.strip())],
    }


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph():
    """Compile and return the orchestrator LangGraph."""
    g = StateGraph(OrchestratorState)

    # Nodes
    g.add_node("orchestrator_node", orchestrator_node)
    g.add_node("invoke_ranking_agent", invoke_ranking_agent)
    g.add_node("invoke_professor_finder", invoke_professor_finder)
    g.add_node("invoke_research_matcher", invoke_research_matcher)
    g.add_node("invoke_email_composer", invoke_email_composer)
    g.add_node("parallel_dispatcher", parallel_dispatcher)
    g.add_node("build_response", build_response)

    # Entry
    g.add_edge(START, "orchestrator_node")

    # Dynamic routing from orchestrator
    g.add_conditional_edges(
        "orchestrator_node",
        router,
        {
            "invoke_ranking_agent":    "invoke_ranking_agent",
            "invoke_professor_finder": "invoke_professor_finder",
            "invoke_research_matcher": "invoke_research_matcher",
            "invoke_email_composer":   "invoke_email_composer",
            "parallel_dispatcher":     "parallel_dispatcher",
            "build_response":          "build_response",
        },
    )

    # All agent nodes return to orchestrator for next decision
    for node in [
        "invoke_ranking_agent",
        "invoke_professor_finder",
        "invoke_research_matcher",
        "invoke_email_composer",
    ]:
        g.add_edge(node, "orchestrator_node")

    # Parallel dispatcher fans out via Send — results merge back to orchestrator
    # (LangGraph handles this automatically with Send)

    # Response is terminal
    g.add_edge("build_response", END)

    return g.compile()


# ── Public run function ────────────────────────────────────────────────────────

def run_orchestrator(user_query: str, conversation_history: list[dict] | None = None) -> dict:
    """
    Entry point called by Streamlit UI.

    Returns dict with:
        final_response: str
        agents_called: list[str]
        ranking_data, professor_data, match_data, email_data: str
    """
    graph = build_graph()

    initial_state: OrchestratorState = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "ranking_data": "",
        "professor_data": "",
        "match_data": "",
        "email_data": "",
        "next_action": "",
        "agents_called": [],
        "context_ids": {},
        "task_ids": {},
        "final_response": "",
    }

    logger.info(f"Orchestrator running | query={user_query[:80]}")
    final_state = graph.invoke(initial_state, {"recursion_limit": 15})
    logger.info(
        f"Orchestrator done | agents_called={final_state.get('agents_called')}"
    )
    return final_state
