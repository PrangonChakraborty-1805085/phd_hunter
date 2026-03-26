"""
LangGraph execution visualizer for Streamlit.

Renders a live pyvis network graph showing:
- All orchestrator nodes
- Which nodes have been visited (highlighted)
- Which agents were called and in what order
"""

import streamlit.components.v1 as components
from pyvis.network import Network


# Node colors
COLOR_DEFAULT  = "#B4B2A9"   # gray — not yet visited
COLOR_ACTIVE   = "#7F77DD"   # purple — currently active
COLOR_DONE     = "#1D9E75"   # teal — completed
COLOR_AGENT    = "#EF9F27"   # amber — agent nodes
COLOR_END      = "#639922"   # green — terminal


def build_graph_html(agents_called: list[str], current_node: str = "") -> str:
    """
    Build a pyvis graph HTML string showing the orchestrator execution flow.

    Args:
        agents_called: List of agent IDs that have been invoked so far
        current_node: The node currently being executed (highlighted differently)
    """
    net = Network(
        height="420px",
        width="100%",
        bgcolor="transparent",
        font_color="#3d3d3a",
        directed=True,
    )
    net.set_options("""
    {
      "physics": { "enabled": false },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.8 } },
        "color": { "color": "#B4B2A9" },
        "width": 1.5,
        "smooth": { "type": "curvedCW", "roundness": 0.2 }
      },
      "nodes": {
        "font": { "size": 13, "face": "sans-serif" },
        "borderWidth": 1.5,
        "shadow": false
      }
    }
    """)

    # ── Define all nodes ──────────────────────────────────────────────────────
    def node_color(node_id: str) -> str:
        if node_id == current_node:
            return COLOR_ACTIVE
        if node_id in agents_called:
            return COLOR_DONE
        if node_id in {"build_response", "END"}:
            return COLOR_END if node_id in agents_called else COLOR_DEFAULT
        if node_id.startswith("invoke_"):
            return COLOR_AGENT if node_id.replace("invoke_", "") in agents_called else COLOR_DEFAULT
        return COLOR_DEFAULT

    def shape(node_id: str) -> str:
        if node_id in {"START", "END"}:
            return "ellipse"
        if node_id == "orchestrator_node":
            return "box"
        if node_id == "parallel_dispatcher":
            return "diamond"
        return "box"

    nodes = {
        "START":                    "START",
        "orchestrator_node":        "Orchestrator\n(LLM brain)",
        "invoke_ranking_agent":     "Ranking\nAgent",
        "invoke_professor_finder":  "Professor\nFinder",
        "invoke_research_matcher":  "Research\nMatcher",
        "invoke_email_composer":    "Email\nComposer",
        "parallel_dispatcher":      "Parallel\nDispatch",
        "build_response":           "Build\nResponse",
        "END":                      "END",
    }

    # Fixed positions for a clean layout
    positions = {
        "START":                    (0,   0),
        "orchestrator_node":        (200, 0),
        "invoke_ranking_agent":     (420, -180),
        "invoke_professor_finder":  (420, -60),
        "invoke_research_matcher":  (420, 60),
        "invoke_email_composer":    (420, 180),
        "parallel_dispatcher":      (420, -240),
        "build_response":           (620, 0),
        "END":                      (800, 0),
    }

    for node_id, label in nodes.items():
        x, y = positions[node_id]
        net.add_node(
            node_id,
            label=label,
            color=node_color(node_id),
            shape=shape(node_id),
            x=x * 1.2,
            y=y * 1.2,
            physics=False,
            size=28 if node_id == "orchestrator_node" else 22,
        )

    # ── Edges ─────────────────────────────────────────────────────────────────
    edges = [
        ("START", "orchestrator_node"),
        ("orchestrator_node", "invoke_ranking_agent",    "if needed"),
        ("orchestrator_node", "invoke_professor_finder", "if needed"),
        ("orchestrator_node", "invoke_research_matcher", "if needed"),
        ("orchestrator_node", "invoke_email_composer",   "if needed"),
        ("orchestrator_node", "parallel_dispatcher",     "parallel"),
        ("orchestrator_node", "build_response",          "done"),
        ("invoke_ranking_agent",    "orchestrator_node", "↩"),
        ("invoke_professor_finder", "orchestrator_node", "↩"),
        ("invoke_research_matcher", "orchestrator_node", "↩"),
        ("invoke_email_composer",   "orchestrator_node", "↩"),
        ("parallel_dispatcher", "invoke_ranking_agent"),
        ("parallel_dispatcher", "invoke_professor_finder"),
        ("build_response", "END"),
    ]

    for edge in edges:
        src, dst = edge[0], edge[1]
        label = edge[2] if len(edge) > 2 else ""
        net.add_edge(src, dst, label=label, font={"size": 10, "color": "#888780"})

    # Generate HTML — pyvis writes to a temp file
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.write_html(f.name)
        tmp_path = f.name

    with open(tmp_path) as f:
        html = f.read()
    os.unlink(tmp_path)
    return html


def render_graph(agents_called: list[str], current_node: str = "", height: int = 440):
    """Render the execution graph inside a Streamlit component."""
    html = build_graph_html(agents_called, current_node)
    components.html(html, height=height, scrolling=False)
