"""
PhD Hunter — Streamlit UI

Layout:
  Left column  (65%): Chat interface
  Right column (35%): LangGraph execution visualization + agent status
"""

import asyncio
import json
import sys
import os
from pathlib import Path

import streamlit as st

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.discovery import discover_all_agents, get_registry
from orchestrator.graph import run_orchestrator
from streamlit_app.graph_viz import render_graph

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PhD Hunter",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.chat-message { padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 0.5rem; }
.user-msg     { background: #EEEDFE; color: #26215C; }
.agent-msg    { background: #E1F5EE; color: #04342C; }
.email-block  { 
    font-family: monospace; font-size: 13px;
    background: #F1EFE8; border-left: 3px solid #7F77DD;
    padding: 1rem; border-radius: 4px; white-space: pre-wrap;
}
.agent-badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 500; margin: 2px;
}
.badge-ranking   { background: #E1F5EE; color: #0F6E56; }
.badge-professor { background: #E6F1FB; color: #185FA5; }
.badge-matcher   { background: #FAEEDA; color: #854F0B; }
.badge-email     { background: #FAECE7; color: #993C1D; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agents_called" not in st.session_state:
    st.session_state.agents_called = []
if "current_node" not in st.session_state:
    st.session_state.current_node = ""
if "registry_loaded" not in st.session_state:
    st.session_state.registry_loaded = False
if "student_profile" not in st.session_state:
    st.session_state.student_profile = {}


# ── Sidebar — Student Profile ─────────────────────────────────────────────────

with st.sidebar:
    st.title("🎓 PhD Hunter")
    st.caption("AI-powered university & professor discovery")
    st.divider()

    st.subheader("Your Profile")
    st.caption("Used for personalized email templates")

    profile = st.session_state.student_profile

    full_name = st.text_input("Full Name", value=profile.get("full_name", ""))
    degree = st.text_input("Degree", value=profile.get("degree", "BSc in Computer Science"))
    university = st.text_input("Your University", value=profile.get("university", ""))
    grad_year = st.text_input("Graduation Year", value=profile.get("graduation_year", "2024"))
    cgpa = st.text_input("CGPA (optional)", value=profile.get("cgpa", ""))
    interests = st.text_area(
        "Research Interests (one per line)",
        value="\n".join(profile.get("research_interests", [])),
        height=100,
    )
    experience = st.text_area(
        "Relevant Experience (optional)",
        value=profile.get("relevant_experience", ""),
        height=80,
    )
    semester = st.selectbox(
        "Target Semester",
        ["Fall 2025", "Spring 2026", "Fall 2026"],
        index=0,
    )

    if st.button("💾 Save Profile", use_container_width=True):
        st.session_state.student_profile = {
            "full_name": full_name,
            "degree": degree,
            "university": university,
            "graduation_year": grad_year,
            "cgpa": cgpa if cgpa else None,
            "research_interests": [i.strip() for i in interests.splitlines() if i.strip()],
            "relevant_experience": experience if experience else None,
            "target_semester": semester,
        }
        st.success("Profile saved!")

    st.divider()
    st.subheader("Agent Status")

    # Show live agent status
    registry = get_registry()
    if registry:
        for agent_id, agent in registry.items():
            status = "🟢" if agent.description != "[UNAVAILABLE]" else "🔴"
            st.caption(f"{status} {agent.name}")
    else:
        st.caption("⏳ Agents not discovered yet")

    if st.button("🔄 Refresh Agents", use_container_width=True):
        with st.spinner("Discovering agents..."):
            asyncio.run(discover_all_agents())
        st.session_state.registry_loaded = True
        st.rerun()

    st.divider()
    st.caption("**Example queries:**")
    examples = [
        "Top 20 CS universities in US with full PhD funding",
        "Find professors in software engineering at MIT",
        "Find universities in top 20 for both CS and Civil in US",
        "Write a PhD email to Prof. John Doe at CMU, distributed systems",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["prefill_query"] = ex


# ── Main layout ───────────────────────────────────────────────────────────────

col_chat, col_graph = st.columns([0.65, 0.35])

# ── Right: Graph visualization ────────────────────────────────────────────────
with col_graph:
    st.subheader("Execution Graph")
    render_graph(
        agents_called=st.session_state.agents_called,
        current_node=st.session_state.current_node,
    )

    # Agents called badges
    if st.session_state.agents_called:
        st.caption("Agents invoked this session:")
        badge_map = {
            "ranking_agent":    ("Ranking",   "badge-ranking"),
            "professor_finder": ("Professor", "badge-professor"),
            "research_matcher": ("Matcher",   "badge-matcher"),
            "email_composer":   ("Email",     "badge-email"),
        }
        badge_html = ""
        for a in st.session_state.agents_called:
            label, cls = badge_map.get(a, (a, "badge-ranking"))
            badge_html += f'<span class="agent-badge {cls}">{label}</span>'
        st.markdown(badge_html, unsafe_allow_html=True)


# ── Left: Chat ────────────────────────────────────────────────────────────────
def render_chat_history():
    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-msg">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            content = msg["content"]
            # If it looks like an email draft, use monospace block
            if "Subject:" in content and "Dear" in content:
                st.markdown(
                    f'<div class="chat-message agent-msg">🤖 <strong>Email Draft</strong>'
                    f'<div class="email-block">{content}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-message agent-msg">🤖 {content}</div>',
                    unsafe_allow_html=True,
                )

with col_chat:
    st.subheader("Chat")

    # Discover agents on first load
    if not st.session_state.registry_loaded:
        with st.spinner("Connecting to agents..."):
            try:
                asyncio.run(discover_all_agents())
                st.session_state.registry_loaded = True
            except Exception as e:
                st.warning(f"Could not connect to agents: {e}. Start agents first.")

    render_chat_history()

    # Input box
    prefill = st.session_state.pop("prefill_query", "")
    user_input = st.chat_input(
        "Ask PhD Hunter... (e.g. 'Top 20 CS universities in US with full funding')",
    )
    query = user_input or prefill

    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        render_chat_history()
        # Inject student profile into query context if available
        profile_ctx = ""
        sp = st.session_state.student_profile
        if sp.get("full_name"):
            profile_ctx = (
                f"\n\n[Student profile: {sp.get('full_name')}, "
                f"{sp.get('degree')} from {sp.get('university')} ({sp.get('graduation_year')}), "
                f"interests: {', '.join(sp.get('research_interests', []))}, "
                f"target: {sp.get('target_semester')}]"
            )

        full_query = query + profile_ctx

        # Run orchestrator
        with st.spinner("PhD Hunter is working..."):
            st.session_state.current_node = "orchestrator_node"
            try:
                result = run_orchestrator(full_query)
                response = result.get("final_response", "No response generated.")
                new_agents = result.get("agents_called", [])

                # Update graph state
                st.session_state.agents_called = list(
                    set(st.session_state.agents_called + new_agents)
                )
                st.session_state.current_node = "build_response"

            except Exception as e:
                response = f"Error: {e}"
                st.session_state.current_node = ""

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
