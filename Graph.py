from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from Node import (
    normalize_query, semantic_cache_lookup,
    respond_from_cache, retrieve, generate,
    cache_write, RAGState)

#================GRAPH STATE=================#
graph = StateGraph(RAGState)
graph.add_node("normalize_query", normalize_query)
graph.add_node("semantic_cache_lookup", semantic_cache_lookup)
graph.add_node("respond_from_cache", respond_from_cache)
graph.add_node("generate", generate)
graph.add_node("retrieve", retrieve)
graph.add_node("cache_write", cache_write)

graph.set_entry_point("normalize_query")
graph.add_edge("normalize_query", "semantic_cache_lookup")

def _branch(state: RAGState) -> str:
    return "respond_from_cache" if state.get("cache_hit") else "retrieve"

graph.add_conditional_edges(
    "semantic_cache_lookup",
    _branch,
    {
        "respond_from_cache": "respond_from_cache",
        "retrieve": "retrieve"
    }
)

graph.add_edge("respond_from_cache", END)
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", "cache_write")
graph.add_edge("cache_write", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)
app

try:
    png_bytes = app.get_graph(xray=True).draw_mermaid_png()
    with open("app.png", "wb") as f:
        f.write(png_bytes)
    print("Saved graph visualization to app.png")
except Exception as e:
    print(f"Could not render graph image ({e}); printing mermaid source instead:\n")
    print(app.get_graph(xray=True).draw_mermaid())


if __name__ == "__main__":
    thread_cfg = {"configurable": {"thread_id": "demo-user-1"}}

    q1 = "What is LangGraph ?"
    out1 = app.invoke({"question": q1, "context_docs": [], "citations": []}, thread_cfg)
    print("Answer:", out1["answer"])
    print("Citations:", out1.get("citations"))
    print("Cache hit?:", out1.get("cache_hit"))
    
    q1 = "Explain about LangGraph ?"
    out1 = app.invoke({"question": q1, "context_docs": [], "citations": []}, thread_cfg)
    print("\nAnswer:", out1["answer"])
    print("Citations:", out1.get("citations"))
    print("Cache hit?:", out1.get("cache_hit"))