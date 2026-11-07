"""An AI data analyst built on autoeda: a LangGraph pipeline with a local model."""

from autoeda.agent.graph import AgentResult, Plan, ask, build_graph
from autoeda.agent.llm import LLM, FakeLLM, LLMError, OllamaClient
from autoeda.agent.queries import Filter, QueryError, QueryResult, QuerySpec, run_query

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
