<div align="center">

# 🎓 PhD Hunter

### AI-Powered University & Professor Discovery for Graduate Applications

*Stop sending 400 emails hoping for one reply. Let agents do the research.*

[![Python](https://img.shields.io/badge/Python-3.11.9-blue?logo=python&logoColor=white)](https://python.org)
[![A2A Protocol](https://img.shields.io/badge/Google-A2A_Protocol-4285F4?logo=google&logoColor=white)](https://github.com/google/a2a)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.6-green)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44.0-red?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📚 Table of Contents

- [🎓 PhD Hunter](#-phd-hunter)
    - [AI-Powered University \& Professor Discovery for Graduate Applications](#ai-powered-university--professor-discovery-for-graduate-applications)
  - [📚 Table of Contents](#-table-of-contents)
  - [📖 Inspiration](#-inspiration)
  - [✨ Features](#-features)
  - [🏗️ Architecture](#️-architecture)
    - [System Overview](#system-overview)
    - [Orchestrator LangGraph — Execution Graph](#orchestrator-langgraph--execution-graph)
  - [🔗 A2A Protocol — How It Works](#-a2a-protocol--how-it-works)
    - [Discovery (runs once at startup)](#discovery-runs-once-at-startup)
    - [Task execution (every agent call)](#task-execution-every-agent-call)
    - [Multi-turn (INPUT\_REQUIRED)](#multi-turn-input_required)
  - [🤖 LLM Providers](#-llm-providers)
  - [🛠️ Tech Stack](#️-tech-stack)
  - [📋 Prerequisites](#-prerequisites)
  - [⚙️ Environment Variables](#️-environment-variables)
  - [🚀 Installation \& Running](#-installation--running)
    - [1. Clone and set up](#1-clone-and-set-up)
    - [2. Start the backend (FastAPI)](#2-start-the-backend-fastapi)
    - [3. Start the UI (Streamlit)](#3-start-the-ui-streamlit)
    - [4. Verify agents are running](#4-verify-agents-are-running)
  - [💬 Example Queries](#-example-queries)
  - [📁 Project Structure](#-project-structure)
  - [✉️ Email Template Design](#️-email-template-design)
  - [🐛 Troubleshooting](#-troubleshooting)
  - [❓ FAQ](#-faq)
  - [🗺️ Roadmap](#️-roadmap)
  - [📄 License](#-license)

---

## 📖 Inspiration

Every year, thousands of students from South Asia, the Middle East, and Africa apply for PhD and Masters programs abroad. The process is brutal: find universities by field ranking, cross-reference scholarship availability, hunt down professors whose research actually matches yours, read their papers, figure out their email, and write a personalised cold email that doesn't sound like it was written by a robot.

Then repeat that 200 to 400 times to land one offer.

**PhD Hunter was built to change that.** It is a multi-agent AI system where a core orchestrator dynamically discovers and delegates to four specialised agents — each one responsible for a slice of the application pipeline. The agents communicate over Google's open **Agent-to-Agent (A2A) protocol**, making the system modular, observable, and extensible. You chat with it in plain English, it does the research, and you get ranked university lists, matched professor profiles, and ready-to-send email drafts — all in one session.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏫 **University Rankings** | CS subfields via CSRankings CSV. Other fields (Civil, Environmental, etc.) via live web search. Cross-field intersection queries supported. |
| 🔍 **Professor Discovery** | Scrapes department faculty pages, extracts emails, lab URLs, and recent paper titles. Graceful fallback when emails are obfuscated. |
| 🧠 **Research Alignment** | Reads a professor's actual lab page and publications, scores 0–1 alignment with your stated interests, and names the best paper to mention. |
| ✉️ **Email Composer** | Uses a human-written structural template. The LLM fills only the research-specific sentences. Output reads like you wrote it yourself. |
| 💰 **Scholarship Filter** | Filter universities to only those with 90%+ funded PhD/Masters programs. |
| 📊 **Live Graph Visualisation** | Streamlit sidebar shows the LangGraph execution graph in real time — which nodes ran, which agents were called, the decision path. |
| 🤖 **Dynamic Orchestration** | The orchestrator LLM decides at runtime which agents to call and in what order. No hardcoded sequences. Parallel dispatch supported. |
| 🔄 **Multi-turn Conversations** | Agents can ask for clarification mid-task via the A2A `INPUT_REQUIRED` state and continue with the same context. |

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (port 8501)                  │
│            Chat Interface + Live LangGraph Viz               │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│              Core Orchestrator (LangGraph + LLM)             │
│  Discovers agents via A2A · Decides dynamically · Routes     │
│                                                             │
│  START → orchestrator_node ──┬─→ invoke_ranking_agent ──┐   │
│               ▲              ├─→ invoke_professor_finder─┤   │
│               │              ├─→ invoke_research_matcher─┤   │
│               └──────────────├─→ invoke_email_composer ──┘   │
│                              ├─→ parallel_dispatcher         │
│                              └─→ build_response → END        │
└────────┬──────────┬──────────┬──────────┬───────────────────┘
         │  A2A     │  A2A     │  A2A     │  A2A
┌────────▼───┐ ┌────▼───┐ ┌───▼────┐ ┌───▼────────┐
│  Ranking   │ │ Prof.  │ │Research│ │  Email     │
│  Agent     │ │ Finder │ │Matcher │ │  Composer  │
│ :10001     │ │ :10002 │ │ :10003 │ │  :10004    │
│            │ │        │ │        │ │            │
│CSRankings  │ │Dept    │ │Reads   │ │Human       │
│CSV + Web   │ │Page    │ │prof    │ │template +  │
│Search      │ │Scraper │ │papers  │ │LLM fill    │
└────────────┘ └────────┘ └────────┘ └────────────┘
```

### Orchestrator LangGraph — Execution Graph

```
         ┌─────────┐
         │  START  │
         └────┬────┘
              │
    ┌─────────▼──────────┐
    │  orchestrator_node  │◄─────────────────────────┐
    │  (LLM brain)        │                          │
    └─────────┬───────────┘                          │
              │                                      │
    ┌─────────▼──────────────────────────────┐       │
    │         conditional_edge (router)       │       │
    └──┬───────┬────────┬──────────┬──────┬──┘       │
       │       │        │          │      │           │
  ┌────▼───┐ ┌─▼──────┐ ┌─▼──────┐ ┌─▼───┐ ┌─▼───┐  │
  │ranking │ │profes- │ │resear- │ │email│ │par- │  │
  │_agent  │ │sor_    │ │ch_     │ │_com-│ │alle-│  │
  │        │ │finder  │ │matcher │ │poser│ │l    │  │
  └────┬───┘ └─┬──────┘ └─┬──────┘ └─┬───┘ └─┬───┘  │
       │       │          │          │      │        │
       └───────┴──────────┴──────────┴──────┘        │
              (all agent nodes return to orchestrator)│
                              │                       │
                    ┌─────────▼───────────┐           │
                    │   build_response    │           │
                    └─────────┬───────────┘           │
                              │
                         ┌────▼────┐
                         │   END   │
                         └─────────┘
```

**Key behaviours:**
- The LLM inside `orchestrator_node` re-evaluates state after every agent returns
- It only calls an agent if the data it needs is not already in state
- It can dispatch multiple agents in parallel via `Send` when tasks are independent
- The loop has a recursion limit of 15 to prevent runaway calls

---

## 🔗 A2A Protocol — How It Works

**Agent-to-Agent (A2A)** is an open protocol by Google that defines how independent AI agents discover each other and communicate over HTTP.

### Discovery (runs once at startup)

```
Orchestrator boots
       │
       ├── GET http://localhost:10001/.well-known/agent.json  →  RankingAgent AgentCard
       ├── GET http://localhost:10002/.well-known/agent.json  →  ProfessorFinder AgentCard
       ├── GET http://localhost:10003/.well-known/agent.json  →  ResearchMatcher AgentCard
       └── GET http://localhost:10004/.well-known/agent.json  →  EmailComposer AgentCard

AgentCards (name, description, skills[]) injected into orchestrator system prompt.
LLM now "knows" all agents purely from their natural-language skill descriptions.
```

### Task execution (every agent call)

```
Orchestrator decides: "call professor_finder"
       │
       ├── POST /  (JSON-RPC)   →  Agent receives message
       │     message.parts[0].text = "Find professors at MIT in software engineering"
       │
       ├── Agent runs LangGraph internally, uses tools (web search, page fetch)
       │
       └── Response: Task { id, contextId, status: COMPLETED, artifacts: [...] }
                                                         │
                                          artifact.parts[0].text = professor JSON list
```

### Multi-turn (INPUT_REQUIRED)

```
Orchestrator → Agent (turn 1):  "Find professors at MIT"
Agent → Orchestrator:            Task { status: INPUT_REQUIRED, message: "Which dept?" }
Orchestrator → Agent (turn 2):  same contextId + taskId, message: "EECS department"
Agent → Orchestrator:            Task { status: COMPLETED, artifacts: [...] }
```

The `contextId` ties multiple turns together. The SDK's `InMemoryTaskStore` handles state persistence between turns automatically.

---

## 🤖 LLM Providers

PhD Hunter supports three LLM providers, switchable via a single environment variable:

| Provider | Best For | Free Tier | Tool Calling |
|---|---|---|---|
| **Groq** | Fast inference, low latency | 500K tokens/day, no card | ✅ Native |
| **OpenRouter** | High rate limits, many model choices | Free models available | ✅ Via `langchain-openrouter` |
| **Gemini** | Long context, multimodal tasks | Generous free tier | ✅ Native |

Recommended models:

```
# Groq (fastest, best for development)
GROQ_MODEL=llama-3.3-70b-versatile

# OpenRouter (best free option with tool calling)
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free

# Gemini
GEMINI_MODEL=gemini-1.5-flash
```

**Local dev (no API costs):** Set `LLM_BACKEND=ollama` and run Ollama in Docker:
```bash
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
docker exec ollama ollama pull qwen2.5:7b
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Protocol** | [Google A2A SDK](https://github.com/google/a2a) `0.3.25` |
| **Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) `1.0.6` |
| **LLM (production)** | Groq `llama-3.3-70b-versatile` / OpenRouter / Gemini |
| **LLM (local dev)** | Ollama `qwen2.5:7b` via Docker |
| **LLM framework** | LangChain Core `1.2.x` |
| **Agent servers** | Uvicorn + Starlette (one per agent) |
| **Web search** | Tavily API (1000 free req/month) |
| **Web scraping** | httpx + BeautifulSoup4 + lxml |
| **University data** | CSRankings CSV (open data, auto-downloaded) |
| **UI** | Streamlit `1.44` + pyvis graph visualisation |
| **Data validation** | Pydantic `2.10.6` |
| **Logging** | Loguru |
| **Testing** | pytest + pytest-asyncio + pytest-mock |
| **Runtime** | Python `3.11.9` |

---

## 📋 Prerequisites

- Python 3.11.9
- `pip` or `uv`
- At least one LLM API key (Groq recommended — free, no card)
- Tavily API key (free, 1000 req/month — replaces Brave)

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | If using Groq | — | [console.groq.com](https://console.groq.com) — free, no card |
| `OPENROUTER_API_KEY` | If using OpenRouter | — | [openrouter.ai](https://openrouter.ai) — free models available |
| `GEMINI_API_KEY` | If using Gemini | — | [aistudio.google.com](https://aistudio.google.com) — free tier |
| `TAVILY_API_KEY` | ✅ Yes | — | [app.tavily.com](https://app.tavily.com) — 1000 free req/month |
| `LLM_PROVIDER` | No | `groq` | `groq` / `openrouter` / `gemini` |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Any Groq-hosted model |
| `OPENROUTER_MODEL` | No | `meta-llama/llama-3.3-70b-instruct:free` | Any OpenRouter model ID |
| `GEMINI_MODEL` | No | `gemini-1.5-flash` | Any Gemini model ID |
| `LLM_BACKEND` | No | `groq` | `groq` / `ollama` (local Docker) |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama Docker URL |
| `OLLAMA_MODEL` | No | `qwen2.5:7b` | Local model name |
| `RANKING_AGENT_PORT` | No | `10001` | Override if port is taken |
| `PROFESSOR_FINDER_PORT` | No | `10002` | |
| `RESEARCH_MATCHER_PORT` | No | `10003` | |
| `EMAIL_COMPOSER_PORT` | No | `10004` | |
| `AGENT_HOST` | No | `localhost` | Change to `0.0.0.0` for Docker |
| `AGENT_URL` | No | `http://localhost:8000` | Change to the actual agent URL for Docker |
| `LOG_LEVEL` | No | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## 🚀 Installation & Running

### 1. Clone and set up

```bash
git clone https://github.com/yourusername/phd-hunter.git
cd phd-hunter

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — add at minimum GROQ_API_KEY and TAVILY_API_KEY
```

### 2. Start the backend (FastAPI)

All four A2A agents are mounted under a single FastAPI server. One command starts everything:

```bash
uvicorn main.fastapi:app --reload --host 0.0.0.0 --port 8000
```

The server exposes each agent at its own path:

| Agent | Endpoint |
|---|---|
| Ranking Agent | `http://localhost:8000/ranking_agent` |
| Professor Finder | `http://localhost:8000/professor_finder` |
| Research Matcher | `http://localhost:8000/research_matcher` |
| Email Composer | `http://localhost:8000/email_composer` |

### 3. Start the UI (Streamlit)

Open a second terminal and run:

```bash
streamlit run streamlit_app/app.py
```

Open **http://localhost:8501** in your browser.

### 4. Verify agents are running

Once the FastAPI server is up, confirm each agent is serving its AgentCard:

```bash
curl http://localhost:8000/ranking_agent/.well-known/agent-card.json
curl http://localhost:8000/professor_finder/.well-known/agent-card.json
curl http://localhost:8000/research_matcher/.well-known/agent-card.json
curl http://localhost:8000/email_composer/.well-known/agent-card.json
```

Each should return a JSON AgentCard with `name`, `description`, and `skills`.

---

## 💬 Example Queries

Once the UI is open, try these in the chat box:

```
Top 20 CS universities in US with full PhD funding
```
```
Find professors in software engineering at MIT — top 5
```
```
Which universities are in the top 20 for both CS and Civil Engineering in US?
```
```
Find top 15 AI/ML universities in Canada with scholarship coverage
```
```
Does Professor [Name]'s work at [University] align with my interest in NLP and transformers?
```
```
Write a PhD cold email to Professor Jane Smith at Stanford — distributed systems
```

**Before writing emails:** fill in your student profile in the left sidebar (name, degree, university, research interests, target semester). The email composer uses this to personalise the template.

---

## 📁 Project Structure

```
phd-hunter/
│
├── agents/
│   ├── ranking_agent/          # CSRankings CSV + web search, scholarship filter
│   │   ├── agent.py            # LangGraph ReAct + Tavily tools
│   │   ├── agent_executor.py   # A2A BaseAgentExecutor bridge
│   │   └── __main__.py         # Agent entry point
│   │
│   ├── professor_finder/       # Dept page scraping, email extraction
│   ├── research_matcher/       # Reads professor pages, alignment scoring
│   └── email_composer/         # Human template + LLM placeholder fill
│
├── orchestrator/
│   ├── discovery.py            # AgentCard fetching at startup (A2ACardResolver)
│   └── graph.py                # Full LangGraph with dynamic routing + parallel Send
│
├── shared/
│   ├── config.py               # Central config + get_llm() factory
│   ├── a2a_helpers.py          # A2ACardResolver, call_agent, response extraction
│   ├── types.py                # Pydantic models shared across agents
│   ├── logging.py              # Loguru setup
│   └── utils.py                # LLM provider wrappers (Groq, OpenRouter, Gemini)
│
├── streamlit_app/
│   ├── app.py                  # Chat UI, sidebar profile, agent status panel
│   └── graph_viz.py            # pyvis live execution graph
│
├── templates/
│   └── email_phd.txt           # Human-written email skeleton with {placeholders}
│
├── tests/
│   ├── test_agents/            # Unit tests for all 4 agents (fully mocked)
│   └── test_orchestrator/      # Router, discovery, response builder tests
│
├── main/
│   ├── __init__.py
│   └── fastapi.py              # FastAPI app — mounts all 4 agents under one server
│
├── .env.example
└── requirements.txt
```

---

## ✉️ Email Template Design

The email composer uses a **hybrid approach** intentionally designed to avoid AI-sounding output:

```
Subject: Prospective {email_type} Student — {field} — {student_name}

Dear {professor_title} {professor_last_name},

[FIXED: standard professional opening — human-written]

[VARIABLE: 2-3 sentences about the professor's specific paper/project]
← THIS is the only part the LLM writes, using real data from Research Matcher

[FIXED: student background block — filled from your saved profile]

[FIXED: research interest statement — filled from your profile]

[FIXED: standard closing — human-written]
```

The LLM fills **only the research-specific connection sentence** using data the Research Matcher scraped from the professor's actual lab page. Everything else is a fixed human-written structure. This means the output reads like a real human email, not a generated one.

**The composer is hardcoded to refuse:** "passionate about", "excited to", "cutting-edge", "leverage synergies", and similar AI tells.

---

## 🐛 Troubleshooting

**Agents not discovered on startup**
```
✗ Could not reach email_composer @ http://localhost:10004: ...
```
The agent is not running yet. Start it manually: `python -m agents.email_composer`. The discovery function retries 5 times with exponential backoff before marking an agent UNAVAILABLE. Click "🔄 Refresh Agents" in the sidebar once agents are up.

**`asyncio.run()` error in Streamlit**
Fixed in `discovery.py` — `run_discovery_sync()` uses a dedicated thread with its own event loop. If you see this, make sure you're using the latest `discovery.py`.

**503 on `/.well-known/agent-card.json`**
The `a2a-sdk 0.3.x` changed the default discovery path to `agent-card.json` but `A2AStarletteApplication` still serves `agent.json`. Fixed in `a2a_helpers.py` — `agent_card_path="/.well-known/agent.json"` is passed explicitly to every `A2ACardResolver`.

**`bind_tools` error with OpenRouter**
Caused by wrapping `ChatOpenRouter` inside a plain Python class that loses the LangChain interface. Fixed in `utils.py` — `get_llm()` returns the raw `ChatOpenRouter` / `ChatGroq` object directly so `create_react_agent` can call `.bind_tools()` on it.

**Raw Task dict in response instead of text**
```
University Rankings: {'contextId': '...', 'kind': 'task', ...}
```
Fixed in `a2a_helpers.py` — response is now unwrapped via `response.root.result` (typed SDK objects) instead of `response.model_dump().get("result")` (raw dict with wrong structure).

**Groq rate limits during heavy use**
Switch to OpenRouter: set `LLM_PROVIDER=openrouter` and `OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free` in `.env`. Free tier has much higher limits. Alternatively set `LLM_BACKEND=ollama` for fully local operation.

---

## ❓ FAQ

**Q: How many emails can this realistically help me send?**
A: The bottleneck is web scraping speed and LLM token limits, not the system design. In a typical session you can research 5–10 universities, find 3–5 professors per university, run alignment scoring on each, and generate email drafts — all within Groq's free 500K daily token budget if prompts are concise.

**Q: Will the emails get me replies?**
A: PhD Hunter handles research and template generation. Reply rates depend on your academic background, the professor's funding situation, and whether your interests genuinely align. The system maximises the quality of your outreach per email — but you still need to review, personalise the experience block, and decide which professors to actually contact.

**Q: Why does the email template not mention my GPA / publications upfront?**
A: The template follows the standard advice from faculty who receive these emails: subject + who you are + why THIS professor + your most relevant experience + clear ask. Listing all credentials upfront reads like a CV dump. The research connection is what gets professors to read further.

**Q: How accurate are the rankings?**
A: For CS subfields, CSRankings is considered one of the most objective sources (publication-count based, not survey-based). For other fields, the agent does web search against QS World University Rankings and US News — these are accurate but updated annually, so results reflect the most recent rankings available online.

**Q: Can I use this for countries other than the US?**
A: Yes. The ranking agent accepts any country. CSRankings covers global institutions. The professor finder works on any university website. The email template is field and country agnostic.

**Q: Is my student profile data stored anywhere?**
A: No. Profile data lives only in Streamlit's `session_state` for the duration of your browser session. It is never written to disk or sent anywhere except into the LLM prompt when generating an email.

**Q: Why A2A protocol instead of just calling functions directly?**
A: A2A gives you independently deployable agents, each with its own server, its own LLM call budget, and its own failure domain. If the professor finder crashes, the ranking agent keeps working. It also makes the system observable (each agent's `/.well-known/agent.json` is a live capability declaration) and extensible (add a new agent by starting a new server — the orchestrator discovers it automatically at next startup).

**Q: Can I add my own agent?**
A: Yes. Create a new folder under `agents/`, implement `AgentExecutor`, write a `__main__.py` that calls `build_a2a_server()`, and add the port to `AGENT_REGISTRY` in `shared/config.py`. The orchestrator discovers it automatically at next startup and its skill descriptions are injected into the LLM context.

---

## 🗺️ Roadmap

- [ ] Google Scholar integration for publication count and h-index
- [ ] Batch email generation across multiple professors in one command
- [ ] Export professor list to CSV / Google Sheets
- [ ] Funding database integration (NSF awards, university stipend data)
- [ ] Streamlit multi-page: saved sessions, professor shortlist management
- [ ] Docker Compose setup for one-command deployment
- [ ] Async parallel professor scraping for faster discovery
- [ ] Statement of Purpose (SOP) draft assistant agent

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built for the students who deserve better than 400 cold emails to a void.

*If this helped you get into grad school, star the repo. ⭐*

</div>
