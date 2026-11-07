"""The analyst pipeline, built with LangGraph.

    profile -> plan -> clean -> analyze -> chart -> critic -> write
                 ^                  |                   |
                 +--- on error -----+---- if rejected --+

Language models make three decisions (which fixes to apply and what to ask, whether the
result answers the question, and how to word the answer). Everything else is ordinary
code: profiling, cleaning, running the query, drawing the chart. The model never writes
code, and every number in its wording is checked against the result table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from autoeda.agent import prompts
from autoeda.agent.guards import plain_answer, ungrounded_numbers
from autoeda.agent.llm import LLM, LLMError, ask_for_model
from autoeda.agent.queries import QueryError, QueryResult, QuerySpec, run_query
from autoeda.agent.viz import result_figure
from autoeda.fixes import CleanResult, clean
from autoeda.loader import load_csv
from autoeda.models import Report
from autoeda.profile import prepare_frame, profile


class Plan(BaseModel):
    fix_ids: list[str] = Field(default_factory=list)
    query: QuerySpec
    rationale: str = ""


class Verdict(BaseModel):
    answers_question: bool = True
    concern: str = ""


class AgentState(TypedDict, total=False):
    question: str
    frame: pd.DataFrame
    report: Report
    plan: Plan
    feedback: str | None
    attempts: int
    revisions: int
    cleaning: CleanResult
    result: QueryResult | None
    error: str | None
    figure: Any
    answer: str
    notes: list[str]
    trace: list[dict[str, str]]
    grounded: bool
    plain_answer_used: bool


@dataclass
class AgentResult:
    question: str
    answer: str
    result: QueryResult | None
    figure: Any
    cleaning: CleanResult | None
    report: Report | None
    plan: Plan | None
    trace: list[dict[str, str]] = field(default_factory=list)
    grounded: bool = True
    plain_answer_used: bool = False
    error: str | None = None


def _step(state: AgentState, name: str, detail: str) -> list[dict[str, str]]:
    return [*state.get("trace", []), {"step": name, "detail": detail}]


def build_graph(
    llm: LLM,
    *,
    max_attempts: int = 3,
    always_safe: bool = True,
    use_critic: bool = True,
    clean_data: bool = True,
):
    """Compile the pipeline.

    Args:
        always_safe: apply every safe fix, and let the model add judgment fixes on top.
            If False only the fixes the model names are applied.
        clean_data: if False the query runs on the raw table (used to measure what the
            cleaning step is worth).
    """

    def profile_node(state: AgentState) -> dict:
        report = profile(state["frame"])
        return {
            "report": report,
            "trace": _step(
                state,
                "profile",
                f"{len(report.issues)} issues, quality score {report.quality_score:.0f}/100",
            ),
        }

    def plan_node(state: AgentState) -> dict:
        report = state["report"]
        prompt = prompts.planner_user(state["question"], report, state.get("feedback"))
        plan, error = ask_for_model(llm, prompts.PLANNER_SYSTEM, prompt, Plan)
        notes = list(state.get("notes", []))
        if plan is None:
            return {
                "plan": None,
                "error": f"The planner gave no usable plan ({error}).",
                "attempts": state.get("attempts", 0) + 1,
                "trace": _step(state, "plan", f"failed: {error}"),
            }
        known = {issue.id for issue in report.issues if issue.fix is not None}
        unknown = [item for item in plan.fix_ids if item not in known]
        if unknown:
            plan.fix_ids = [item for item in plan.fix_ids if item in known]
            notes.append(f"Ignored fixes that do not exist: {', '.join(unknown)}.")
        return {
            "plan": plan,
            "error": None,
            "notes": notes,
            "trace": _step(
                state,
                "plan",
                f"fixes={plan.fix_ids}; {plan.query.operation} on {plan.query.column}",
            ),
        }

    def clean_node(state: AgentState) -> dict:
        report = state["report"]
        plan = state["plan"]
        if not clean_data or plan is None:
            return {"cleaning": clean(state["frame"], report, fixes=[])}
        ids = set(plan.fix_ids)
        if always_safe:
            ids |= {i.id for i in report.issues if i.fix is not None and i.fix.safe_auto}
        cleaning = clean(state["frame"], report, fixes=sorted(ids))
        detail = f"{len(cleaning.applied)} fixes applied, {len(cleaning.skipped)} skipped"
        return {"cleaning": cleaning, "trace": _step(state, "clean", detail)}

    def analyze_node(state: AgentState) -> dict:
        plan = state["plan"]
        attempts = state.get("attempts", 0)
        if plan is None:
            return {"result": None}
        try:
            result = run_query(state["cleaning"].df, plan.query)
        except QueryError as error:
            return {
                "result": None,
                "error": str(error),
                "feedback": str(error),
                "attempts": attempts + 1,
                "trace": _step(state, "analyze", f"query error: {error}"),
            }
        return {
            "result": result,
            "error": None,
            "trace": _step(state, "analyze", f"{result.title}: {len(result.table)} rows"),
        }

    def chart_node(state: AgentState) -> dict:
        return {"figure": result_figure(state["result"])}

    def critic_node(state: AgentState) -> dict:
        result = state["result"]
        if not use_critic or result is None:
            return {}
        prompt = prompts.critic_user(
            state["question"],
            result.spec.model_dump_json(exclude_defaults=True),
            result.to_markdown(),
            result.rows_used,
        )
        verdict, error = ask_for_model(llm, prompts.CRITIC_SYSTEM, prompt, Verdict)
        if verdict is None:
            return {"trace": _step(state, "critic", f"skipped ({error})")}
        update: dict[str, Any] = {
            "trace": _step(
                state, "critic", "accepted" if verdict.answers_question else verdict.concern
            )
        }
        if not verdict.answers_question and state.get("revisions", 0) < 1:
            update["feedback"] = f"The result did not answer the question: {verdict.concern}"
            update["revisions"] = state.get("revisions", 0) + 1
            update["error"] = "rejected"
        return update

    def write_node(state: AgentState) -> dict:
        result = state.get("result")
        question = state["question"]
        if result is None:
            message = state.get("error") or "No answer could be produced."
            return {
                "answer": f"I could not answer this: {message}",
                "grounded": True,
                "plain_answer_used": True,
                "trace": _step(state, "write", "no result to describe"),
            }

        cleaning = state.get("cleaning")
        notes = list(state.get("notes", []))
        if cleaning and cleaning.applied:
            notes.append(
                "Data was cleaned first: "
                + "; ".join(f"{a.fix.description} ({a.effect})" for a in cleaning.applied)
            )
        prompt = prompts.writer_user(question, result.title, result.to_markdown(), notes)
        try:
            draft = llm.complete(prompts.WRITER_SYSTEM, prompt).strip()
        except LLMError:
            draft = ""

        allowed = question + "\n" + "\n".join(notes)
        bad = ungrounded_numbers(draft, result, allowed_text=allowed) if draft else []
        if not draft or bad:
            reason = "model unavailable or empty" if not draft else f"ungrounded numbers {bad}"
            return {
                "answer": plain_answer(question, result),
                "grounded": True,
                "plain_answer_used": True,
                "trace": _step(state, "write", f"used plain answer ({reason})"),
            }
        return {
            "answer": draft,
            "grounded": True,
            "plain_answer_used": False,
            "trace": _step(state, "write", "answer checked against the result table"),
        }

    def after_plan(state: AgentState) -> str:
        if state.get("plan") is None:
            return "plan" if state.get("attempts", 0) < max_attempts else "write"
        return "clean"

    def after_analyze(state: AgentState) -> str:
        if state.get("result") is not None:
            return "chart"
        return "plan" if state.get("attempts", 0) < max_attempts else "write"

    def after_critic(state: AgentState) -> str:
        return "plan" if state.get("error") == "rejected" else "write"

    graph = StateGraph(AgentState)
    graph.add_node("profile", profile_node)
    graph.add_node("plan", plan_node)
    graph.add_node("clean", clean_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("chart", chart_node)
    graph.add_node("critic", critic_node)
    graph.add_node("write", write_node)

    graph.add_edge(START, "profile")
    graph.add_edge("profile", "plan")
    graph.add_conditional_edges(
        "plan", after_plan, {"plan": "plan", "clean": "clean", "write": "write"}
    )
    graph.add_edge("clean", "analyze")
    graph.add_conditional_edges(
        "analyze", after_analyze, {"chart": "chart", "plan": "plan", "write": "write"}
    )
    graph.add_edge("chart", "critic")
    graph.add_conditional_edges("critic", after_critic, {"plan": "plan", "write": "write"})
    graph.add_edge("write", END)
    return graph.compile()


def ask(
    source: str | Path | pd.DataFrame,
    question: str,
    llm: LLM,
    **options: Any,
) -> AgentResult:
    """Answer a question about a table. See `build_graph` for the options."""
    if isinstance(source, pd.DataFrame):
        frame = prepare_frame(source)
    else:
        frame = prepare_frame(load_csv(source).df)

    app = build_graph(llm, **options)
    final = app.invoke({"question": question, "frame": frame, "attempts": 0, "revisions": 0})
    return AgentResult(
        question=question,
        answer=final.get("answer", ""),
        result=final.get("result"),
        figure=final.get("figure"),
        cleaning=final.get("cleaning"),
        report=final.get("report"),
        plan=final.get("plan"),
        trace=final.get("trace", []),
        grounded=bool(final.get("grounded", True)),
        plain_answer_used=bool(final.get("plain_answer_used", False)),
        error=None if final.get("result") is not None else final.get("error"),
    )
