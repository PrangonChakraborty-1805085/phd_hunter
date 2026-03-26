"""
Ranking Agent Executor — bridges A2A protocol to LangGraph agent.

The SDK calls execute() for every incoming message.
We extract the user text, run the LangGraph agent, and enqueue the result.
"""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, TextPart
from a2a.utils import new_task

from agents.ranking_agent.agent import run_ranking_agent
from shared.logging import logger


class RankingAgentExecutor(AgentExecutor):
    """A2A executor for the Ranking Agent."""

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        # 1. Create or retrieve task
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        try:
            # 2. Extract user message text
            user_text = self._get_text(context)
            logger.info(f"RankingAgent received: {user_text[:100]}")

            # 3. Run the LangGraph agent
            response_text = run_ranking_agent(
                query=user_text,
                context_id=task.context_id,
            )

            # 4. Return completed result
            await updater.add_artifact(
                parts=[TextPart(text=response_text)],
                name="ranking_result",
            )
            await updater.complete()

        except Exception as e:
            logger.error(f"RankingAgent error: {e}")
            await updater.failed(
                message=updater.new_agent_message(
                    parts=[TextPart(text=f"Error in Ranking Agent: {str(e)}")]
                )
            )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise NotImplementedError("Ranking Agent does not support cancellation")

    def _get_text(self, context: RequestContext) -> str:
        """Extract plain text from the A2A message parts."""
        try:
            parts = context.message.parts
            for part in parts:
                if hasattr(part, "root") and hasattr(part.root, "text"):
                    return part.root.text
                if hasattr(part, "text"):
                    return part.text
        except Exception:
            pass
        return str(context.message)
