from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.email_composer import get_email_composer_agent
from agents.professor_finder import get_professor_finder_agent
from agents.ranking_agent import get_ranking_agent
from agents.research_matcher import get_research_matcher_agent


app = FastAPI(
    title="PhD Hunter App API",
    description="Backend API for PhD Hunter agents and shared tools",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In prod: restrict to your Streamlit origin
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/email_composer", get_email_composer_agent())
app.mount("/professor_finder", get_professor_finder_agent())
app.mount("/ranking_agent", get_ranking_agent())
app.mount("/research_matcher", get_research_matcher_agent())

@app.get("/")
def root():
    return {"status": "running"}