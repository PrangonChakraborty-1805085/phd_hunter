"""Shared data models used as inputs/outputs between agents."""

from typing import Optional
from pydantic import BaseModel, HttpUrl


# ── Ranking Agent I/O ─────────────────────────────────────────────────────────

class RankingRequest(BaseModel):
    fields: list[str]           # e.g. ["cs", "civil"]  
    country: str                # e.g. "US"
    top_n: int = 20
    min_scholarship_pct: int = 0  # 0 = no filter, 90 = 90%+ funded only


class University(BaseModel):
    name: str
    rank: int
    field: str
    country: str
    website: Optional[str] = None
    scholarship_info: Optional[str] = None
    fully_funded: Optional[bool] = None


class RankingResult(BaseModel):
    universities: list[University]
    source: str
    notes: Optional[str] = None


# ── Professor Finder I/O ──────────────────────────────────────────────────────

class ProfessorRequest(BaseModel):
    university_name: str
    university_website: Optional[str] = None
    research_field: str         # e.g. "software engineering"
    max_results: int = 10


class Professor(BaseModel):
    name: str
    title: Optional[str] = None
    university: str
    department: Optional[str] = None
    email: Optional[str] = None
    lab_url: Optional[str] = None
    profile_url: Optional[str] = None
    research_areas: list[str] = []
    recent_papers: list[str] = []   # paper titles
    email_found: bool = False


class ProfessorResult(BaseModel):
    professors: list[Professor]
    university: str
    field: str
    notes: Optional[str] = None


# ── Research Matcher I/O ──────────────────────────────────────────────────────

class MatchRequest(BaseModel):
    professor: Professor
    student_interests: list[str]    # e.g. ["distributed systems", "cloud"]


class MatchResult(BaseModel):
    professor_name: str
    university: str
    alignment_score: float          # 0.0 to 1.0
    alignment_summary: str          # 2-3 sentences why they match
    matching_topics: list[str]
    professor_recent_work: str      # 1-2 sentence summary of their work


# ── Email Composer I/O ────────────────────────────────────────────────────────

class StudentProfile(BaseModel):
    full_name: str
    degree: str                     # e.g. "BSc in Computer Science"
    university: str
    graduation_year: str            # e.g. "2024"
    cgpa: Optional[str] = None
    research_interests: list[str]
    relevant_experience: Optional[str] = None
    target_semester: str            # e.g. "Fall 2025"


class EmailRequest(BaseModel):
    professor: Professor
    match_result: MatchResult
    student_profile: StudentProfile
    email_type: str = "phd"         # "phd" or "masters"


class EmailDraft(BaseModel):
    subject: str
    body: str                       # contains <placeholders> for student review
    professor_name: str
    university: str
    notes: str                      # tips for customization


# ── Orchestrator State ────────────────────────────────────────────────────────

class OrchestratorState(BaseModel):
    """LangGraph state that flows through the orchestrator graph."""
    user_query: str
    conversation_history: list[dict] = []

    # Collected data from agents
    ranking_results: Optional[list[University]] = None
    professor_results: Optional[list[Professor]] = None
    match_results: Optional[list[MatchResult]] = None
    email_drafts: Optional[list[EmailDraft]] = None

    # Orchestrator decisions
    next_action: Optional[str] = None
    agents_called: list[str] = []
    final_response: Optional[str] = None
    error: Optional[str] = None
