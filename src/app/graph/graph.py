"""StateGraph wiring: route -> (extract_lead || retrieve) -> generate -> lead_strategy -> crm?"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import Nodes, should_submit
from app.graph.state import GraphState


def build_graph(nodes: Nodes):
    graph = StateGraph(GraphState)
    graph.add_node("route", nodes.route)
    graph.add_node("extract_lead", nodes.extract_lead)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("lead_strategy", nodes.lead_strategy)
    graph.add_node("crm_submit", nodes.crm_submit)

    graph.add_edge(START, "route")
    # Contact extraction and retrieval are independent, so they run in parallel.
    graph.add_edge("route", "extract_lead")
    graph.add_edge("route", "retrieve")
    graph.add_edge("extract_lead", "generate")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "lead_strategy")
    graph.add_conditional_edges(
        "lead_strategy", should_submit, {"crm_submit": "crm_submit", "end": END}
    )
    graph.add_edge("crm_submit", END)
    return graph.compile()
