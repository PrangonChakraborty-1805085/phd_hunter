"""
Email Composer Agent — core LangGraph logic.

Strategy (critical for non-AI-sounding output):
- The skeleton email is a HUMAN-WRITTEN template in templates/email_phd.txt
- The LLM fills ONLY the research-specific sentences using real data from Research Matcher
- Everything else (greetings, structure, closing) comes from the fixed template
- Result: professional, standard email that reads human, not AI-generated
"""

import json
import re
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool


from shared.config import config
from shared.logging import logger
from shared.utils import get_llm, message_preprocessor


TEMPLATE_PATH = Path(__file__).parent.parent.parent / "templates" / "email_phd.txt"

# ── System prompt — very constrained, fills only the research sentence ────────

SYSTEM_PROMPT = """You are an expert at writing PhD cold emails.

You will receive:
1. A pre-written email template with {placeholder} fields
2. Professor data: name, university, recent work, alignment summary, matching topics
3. Student profile data: name, degree, university, year, interests

Your ONLY job is to fill in the placeholders. Rules:
- DO NOT rewrite the template structure
- DO NOT add new paragraphs
- DO NOT use AI phrases like "I am passionate about", "I am excited to", 
  "leveraging synergies", "cutting-edge", "state-of-the-art"
- Keep the {alignment_summary} to 2 plain sentences referencing the professor's real work
- Keep the {relevant_experience_block} to 2 sentences, plain and factual
- {matching_topics} should be a natural comma-separated phrase, e.g. 
  "fault-tolerant distributed systems and cloud resource scheduling"
- Return the COMPLETE filled email — subject line included
- No explanations before or after the email

The goal: an email that looks like a human PhD applicant wrote it after reading
the professor's actual papers. Simple, direct, specific.
"""


def _load_template() -> str:
    try:
        return TEMPLATE_PATH.read_text()
    except FileNotFoundError:
        logger.error(f"Email template not found at {TEMPLATE_PATH}")
        raise


def _build_fill_prompt(
    template: str,
    professor_name: str,
    professor_title: str,
    university: str,
    alignment_summary: str,
    matching_topics: list[str],
    professor_recent_work: str,
    suggested_paper: str,
    student_name: str,
    degree: str,
    student_university: str,
    graduation_year: str,
    cgpa: str | None,
    relevant_experience: str | None,
    target_semester: str,
    email_type: str,
    field: str,
) -> str:
    """Builds the prompt that asks the LLM to fill the template."""
    last_name = professor_name.split()[-1] if professor_name else "Professor"
    cgpa_line = f", CGPA {cgpa}/4.0" if cgpa else ""
    experience_block = (
        relevant_experience
        if relevant_experience
        else "<Describe your most relevant project or thesis work in 1-2 sentences>"
    )
    topics_str = ", ".join(matching_topics) if matching_topics else "<your research interests>"

    return f"""Fill in the following email template with the data provided.

TEMPLATE:
{template}

DATA TO USE:
- professor_title: {professor_title}
- professor_last_name: {last_name}
- professor_recent_work: {professor_recent_work}
- suggested_paper: {suggested_paper}
- alignment_summary: {alignment_summary}
- matching_topics: {topics_str}
- student_name: {student_name}
- degree: {degree}
- student_university: {student_university}
- graduation_year: {graduation_year}
- cgpa_line: {cgpa_line}
- relevant_experience_block: {experience_block}
- target_semester: {target_semester}
- email_type: {email_type}
- field: {field}
- student_email: <your.email@university.edu>

Return the complete filled email. Subject line first, then blank line, then body."""


def run_email_composer(
    professor_name: str,
    professor_title: str,
    university: str,
    alignment_summary: str,
    matching_topics: list[str],
    professor_recent_work: str,
    suggested_paper: str,
    student_name: str,
    degree: str,
    student_university: str,
    graduation_year: str,
    cgpa: str | None,
    relevant_experience: str | None,
    target_semester: str,
    email_type: str = "PhD",
    field: str = "Computer Science",
    context_id: str | None = None,
) -> dict:
    """
    Generate a filled email draft.
    Returns dict with 'subject', 'body', 'notes'.
    """
    template = _load_template()
    fill_prompt = _build_fill_prompt(
        template=template,
        professor_name=professor_name,
        professor_title=professor_title,
        university=university,
        alignment_summary=alignment_summary,
        matching_topics=matching_topics,
        professor_recent_work=professor_recent_work,
        suggested_paper=suggested_paper,
        student_name=student_name,
        degree=degree,
        student_university=student_university,
        graduation_year=graduation_year,
        cgpa=cgpa,
        relevant_experience=relevant_experience,
        target_semester=target_semester,
        email_type=email_type,
        field=field,
    )

    llm = get_llm()

    logger.info(
        f"EmailComposer invoked | professor={professor_name} | "
        f"context_id={context_id}"
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=fill_prompt),
    ])

    full_email = message_preprocessor(response)

    # Split subject from body
    lines = full_email.split("\n", 2)
    subject = ""
    body = full_email
    if lines and lines[0].lower().startswith("subject:"):
        subject = lines[0].replace("Subject:", "").replace("subject:", "").strip()
        body = "\n".join(lines[2:]).strip() if len(lines) > 2 else "\n".join(lines[1:]).strip()

    notes = (
        "Review and personalize: fill in <your.email@...>, "
        "verify the paper title is correct, and adjust the experience block "
        "to match your actual work. Keep the email under 300 words."
    )

    logger.info(f"EmailComposer completed | subject={subject[:60]}")
    return {"subject": subject, "body": body, "notes": notes}
