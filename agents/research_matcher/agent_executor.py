"""Research Matcher A2A executor."""

import json

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TextPart
from a2a.utils import new_task

from agents.research_matcher.agent import run_research_matcher
from shared.logging import logger


class ResearchMatcherExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        try:
            user_text = self._get_text(context)
            logger.info(f"ResearchMatcher received: {user_text[:120]}")

            # Parse the structured query from orchestrator
            # Expected format:
            # "Professor: <name> at <university>\nProfile URL: <url>\nStudent interests: <list>"
            params = self._parse_query(user_text)

            response = run_research_matcher(
                professor_name=params.get("professor_name", "Unknown"),
                university=params.get("university", "Unknown"),
                profile_url=params.get("profile_url", ""),
                student_interests=params.get("student_interests", []),
                context_id=task.context_id,
            )

            await updater.add_artifact(
                parts=[TextPart(text=response)],
                name="match_result",
            )
            await updater.complete()

        except Exception as e:
            logger.error(f"ResearchMatcher error: {e}")
            await updater.failed(
                message=updater.new_agent_message(
                    parts=[TextPart(text=f"Error in Research Matcher: {str(e)}")]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError

    def _get_text(self, context: RequestContext) -> str:
        try:
            for part in context.message.parts:
                if hasattr(part, "root") and hasattr(part.root, "text"):
                    return part.root.text
                if hasattr(part, "text"):
                    return part.text
        except Exception:
            pass
        return str(context.message)

    def _parse_query(self, text: str) -> dict:
        """
        Parse the structured message from the orchestrator.
        Falls back gracefully if format is unexpected.
        """
        result = {
            "professor_name": "Unknown",
            "university": "Unknown",
            "profile_url": "",
            "student_interests": [],
        }
        try:
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("Professor:"):
                    # "Professor: John Doe at MIT"
                    val = line.replace("Professor:", "").strip()
                    if " at " in val:
                        parts = val.split(" at ", 1)
                        result["professor_name"] = parts[0].strip()
                        result["university"] = parts[1].strip()
                    else:
                        result["professor_name"] = val
                elif line.startswith("Profile URL:"):
                    result["profile_url"] = line.replace("Profile URL:", "").strip()
                elif line.startswith("Student interests:"):
                    interests_str = line.replace("Student interests:", "").strip()
                    result["student_interests"] = [
                        i.strip() for i in interests_str.split(",") if i.strip()
                    ]
        except Exception as e:
            logger.warning(f"_parse_query fallback: {e}")
        return result
