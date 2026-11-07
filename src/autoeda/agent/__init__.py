"""An AI data analyst built on autoeda: a LangGraph pipeline with a local model.

The query language, the number guard and the model clients work without LangGraph. Only
`ask`, `build_graph`, `Plan` and `AgentResult` need it, so those are loaded on first use.
That lets the web demo run the manual query builder without the `agent` extra installed.
"""

from __future__ import annotations

from typing import Any

from autoeda.agent.llm import LLM, FakeLLM, LLMError, OllamaClient
from autoeda.agent.queries import Filter, QueryError, QueryResult, QuerySpec, run_query

_NEEDS_LANGGRAPH = {"AgentResult", "Plan", "ask", "build_graph"}

__all__ = [
    "LLM",
    "AgentResult",
    "FakeLLM",
    "Filter",
    "LLMError",
    "OllamaClient",
    "Plan",
    "QueryError",
    "QueryResult",
    "QuerySpec",
    "ask",
    "build_graph",
    "run_query",
]


def __getattr__(name: str) -> Any:
    if name in _NEEDS_LANGGRAPH:
        from autoeda.agent import graph

        return getattr(graph, name)
    raise AttributeError(f"module 'autoeda.agent' has no attribute {name!r}")
