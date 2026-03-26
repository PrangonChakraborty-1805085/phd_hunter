"""Email Composer A2A executor."""

import json

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TextPart
from a2a.utils import new_task

from agents.email_composer.agent import run_email_composer
from shared.logging import logger


class EmailComposerExecutor(AgentExecutor):

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        try:
            user_text = self._get_text(context)
            logger.info(f"EmailComposer received message length={len(user_text)}")

            # Orchestrator sends a JSON payload with all required fields
            params = json.loads(user_text)

            result = run_email_composer(
                professor_name=params.get("professor_name", ""),
                professor_title=params.get("professor_title", "Professor"),
                university=params.get("university", ""),
                alignment_summary=params.get("alignment_summary", ""),
                matching_topics=params.get("matching_topics", []),
                professor_recent_work=params.get("professor_recent_work", ""),
                suggested_paper=params.get("suggested_paper", ""),
                student_name=params.get("student_name", "<Your Name>"),
                degree=params.get("degree", "<Your Degree>"),
                student_university=params.get("student_university", "<Your University>"),
                graduation_year=params.get("graduation_year", "<Year>"),
                cgpa=params.get("cgpa"),
                relevant_experience=params.get("relevant_experience"),
                target_semester=params.get("target_semester", "Fall 2025"),
                email_type=params.get("email_type", "PhD"),
                field=params.get("field", "Computer Science"),
                context_id=task.context_id,
            )

            await updater.add_artifact(
                parts=[TextPart(text=json.dumps(result))],
                name="email_draft",
            )
            await updater.complete()

        except json.JSONDecodeError as e:
            logger.error(f"EmailComposer: invalid JSON payload: {e}")
            await updater.failed(
                message=updater.new_agent_message(
                    parts=[TextPart(
                        text=f"EmailComposer expects a JSON payload. Error: {e}"
                    )]
                )
            )
        except Exception as e:
            logger.error(f"EmailComposer error: {e}")
            await updater.failed(
                message=updater.new_agent_message(
                    parts=[TextPart(text=f"Error in Email Composer: {str(e)}")]
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
