"""Professor Finder A2A executor."""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TextPart
from a2a.utils import new_task

from agents.professor_finder.agent import run_professor_finder
from shared.logging import logger


class ProfessorFinderExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        try:
            user_text = self._get_text(context)
            logger.info(f"ProfessorFinder received: {user_text[:100]}")

            response = run_professor_finder(
                query=user_text,
                context_id=task.context_id,
            )

            await updater.add_artifact(
                parts=[TextPart(text=response)],
                name="professor_list",
            )
            await updater.complete()

        except Exception as e:
            logger.error(f"ProfessorFinder error: {e}")
            await updater.failed(
                message=updater.new_agent_message(
                    parts=[TextPart(text=f"Error in Professor Finder: {str(e)}")]
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
