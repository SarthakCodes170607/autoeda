import json
from pathlib import Path

import pandas as pd
import pytest

from autoeda.agent import FakeLLM, QuerySpec, ask
from autoeda.agent.graph import Plan
from autoeda.agent.guards import numbers_in, plain_answer, ungrounded_numbers
from autoeda.agent.llm import LLMError, ask_for_model, extract_json
from autoeda.agent.queries import run_query

SAMPLE = Path(__file__).parent.parent / "examples" / "sample_messy.csv"

TOTAL_BY_CITY = {
    "fix_ids": [],
    "query": {
        "operation": "top_k",
        "column": "amount",
        "agg": "sum",
        "group_by": ["city"],
        "k": 3,
    },
    "rationale": "total amount by city",
}


def scripted(plans, verdicts=None, answer="Delhi leads."):
    """A fake model: planner replies come from `plans` in order, others are fixed."""
    plan_replies = [json.dumps(plan) if isinstance(plan, dict) else plan for plan in plans]
    verdict_replies = [json.dumps(v) for v in (verdicts or [{"answers_question": True}])]
    counters = {"plan": 0, "critic": 0}

    def respond(system, user):
        if system.startswith("You plan"):
            index = min(counters["plan"], len(plan_replies) - 1)
            counters["plan"] += 1
            return plan_replies[index]
        if system.startswith("You check"):
            index = min(counters["critic"], len(verdict_replies) - 1)
            counters["critic"] += 1
            return verdict_replies[index]
        return answer

    return FakeLLM(respond)


# ---------------------------------------------------------------- helpers


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_ask_for_model_retries_once_with_the_error():
    llm = FakeLLM(["not json", '{"operation": "describe"}'])

    plan, error = ask_for_model(llm, "sys", "user", QuerySpec)

    assert plan is not None and error is None
    assert "not valid" in llm.calls[1][1]


def test_ask_for_model_gives_up_after_the_retry():
    plan, error = ask_for_model(FakeLLM(["nope"]), "s", "u", QuerySpec)

    assert plan is None
    assert error


def test_ask_for_model_reports_an_unreachable_model():
    def broken(system, user):
        raise LLMError("Ollama call failed: connection refused")

    plan, error = ask_for_model(FakeLLM(broken), "s", "u", QuerySpec)

    assert plan is None and "connection refused" in error


# ---------------------------------------------------------------- the pipeline


def test_without_the_right_fix_the_query_fails_and_the_planner_is_told_why():
    fixed = {**TOTAL_BY_CITY, "fix_ids": []}
    llm = scripted([fixed])

    result = ask(SAMPLE, "Total amount by city?", llm, clean_data=False, max_attempts=2)

    assert result.result is None
    assert "not numeric" in (result.error or "")
    assert "I could not answer" in result.answer


def test_safe_fixes_are_applied_even_if_the_model_names_none():
    llm = scripted([TOTAL_BY_CITY])

    result = ask(SAMPLE, "Total amount by city?", llm)

    assert result.result is not None
    assert result.cleaning is not None
    assert {a.issue_id for a in result.cleaning.applied} >= {"numeric_as_text:amount"}
    assert result.result.rows_used == 300
    assert [step["step"] for step in result.trace][:4] == ["profile", "plan", "clean", "analyze"]


def test_the_model_can_add_judgment_fixes():
    plan = {**TOTAL_BY_CITY, "fix_ids": ["outliers:age"]}

    result = ask(SAMPLE, "Total amount by city?", scripted([plan]))

    assert "outliers:age" in {a.issue_id for a in result.cleaning.applied}


def test_only_named_fixes_when_always_safe_is_off():
    plan = {**TOTAL_BY_CITY, "fix_ids": ["numeric_as_text:amount"]}

    result = ask(SAMPLE, "Total amount by city?", scripted([plan]), always_safe=False)

    assert [a.issue_id for a in result.cleaning.applied] == ["numeric_as_text:amount"]


def test_unknown_fix_ids_are_dropped_with_a_note():
    plan = {**TOTAL_BY_CITY, "fix_ids": ["made_up:column"]}

    result = ask(SAMPLE, "Total amount by city?", scripted([plan]))

    assert result.result is not None
    assert all(a.issue_id != "made_up:column" for a in result.cleaning.applied)


def test_a_bad_query_is_repaired_on_the_second_attempt():
    wrong = {**TOTAL_BY_CITY, "query": {**TOTAL_BY_CITY["query"], "column": "Amount"}}
    llm = scripted([wrong, TOTAL_BY_CITY])

    result = ask(SAMPLE, "Total amount by city?", llm)

    assert result.result is not None
    assert any("query error" in step["detail"] for step in result.trace)
    assert "Did you mean 'amount'" in llm.calls[2][1] or "Did you mean 'amount'" in llm.calls[1][1]


def test_planner_that_never_gives_json_ends_with_an_honest_message():
    result = ask(SAMPLE, "Anything?", scripted(["garbage"]), max_attempts=2)

    assert result.result is None
    assert "could not answer" in result.answer


def test_critic_rejection_sends_the_planner_back_once():
    first = {**TOTAL_BY_CITY, "query": {"operation": "describe"}}
    llm = scripted(
        [first, TOTAL_BY_CITY],
        verdicts=[
            {"answers_question": False, "concern": "wrong operation"},
            {"answers_question": True},
        ],
    )

    result = ask(SAMPLE, "Total amount by city?", llm)

    assert result.result.spec.operation == "top_k"
    assert any(step["detail"] == "wrong operation" for step in result.trace)


def test_critic_cannot_loop_forever():
    llm = scripted([TOTAL_BY_CITY], verdicts=[{"answers_question": False, "concern": "no"}])

    result = ask(SAMPLE, "Total amount by city?", llm)

    assert result.result is not None


def test_critic_can_be_switched_off():
    llm = scripted([TOTAL_BY_CITY])

    ask(SAMPLE, "Total amount by city?", llm, use_critic=False)

    assert not any(system.startswith("You check") for system, _ in llm.calls)


def test_answer_with_invented_numbers_is_replaced_by_a_plain_one():
    llm = scripted([TOTAL_BY_CITY], answer="Delhi made 9,876,543 in sales, a record.")

    result = ask(SAMPLE, "Total amount by city?", llm)

    assert result.plain_answer_used is True
    assert "9,876,543" not in result.answer
    assert "city" in result.answer.lower()


def test_answer_using_real_numbers_is_kept():
    probe = ask(SAMPLE, "Total amount by city?", scripted([TOTAL_BY_CITY]))
    top = probe.result.table.iloc[0]
    wording = f"{top['city']} is first with {top.iloc[1]:,.0f} in total."

    result = ask(SAMPLE, "Total amount by city?", scripted([TOTAL_BY_CITY], answer=wording))

    assert result.plain_answer_used is False
    assert result.answer == wording


def test_unreachable_writer_falls_back_to_a_plain_answer():
    def respond(system, user):
        if system.startswith("You plan"):
            return json.dumps(TOTAL_BY_CITY)
        if system.startswith("You check"):
            return '{"answers_question": true}'
        raise LLMError("down")

    result = ask(SAMPLE, "Total amount by city?", FakeLLM(respond))

    assert result.plain_answer_used is True
    assert result.result is not None


def test_dataframe_input_and_original_frame_untouched():
    frame = pd.read_csv(SAMPLE)
    before = frame.copy()

    ask(frame, "Total amount by city?", scripted([TOTAL_BY_CITY]))

    pd.testing.assert_frame_equal(frame, before)


def test_figure_is_drawn_for_grouped_results():
    result = ask(SAMPLE, "Total amount by city?", scripted([TOTAL_BY_CITY]))

    assert result.figure is not None


def test_plan_model_defaults():
    plan = Plan.model_validate({"query": {"operation": "describe"}})

    assert plan.fix_ids == []


# ---------------------------------------------------------------- number guard


@pytest.fixture
def small_result():
    frame = pd.DataFrame({"city": ["A", "B", "C"], "sales": [1234.5, 800.0, 66.25]})
    spec = QuerySpec(operation="aggregate", column="sales", group_by=["city"])
    return run_query(frame, spec)


def test_numbers_in_reads_commas_decimals_and_percents():
    assert numbers_in("Sales were 1,234.5 (12%) and -3") == [1234.5, 12.0, -3.0]


@pytest.mark.parametrize("text", ["A sold 1,234.5.", "About 1,235.", "800 for B", "66.3 units"])
def test_numbers_from_the_table_are_accepted(small_result, text):
    assert ungrounded_numbers(text, small_result) == []


def test_invented_numbers_are_caught(small_result):
    assert ungrounded_numbers("A sold 5,000 units", small_result) == [5000.0]


def test_numbers_from_the_question_and_small_counting_words_are_allowed(small_result):
    assert (
        ungrounded_numbers("Top 3 cities in 2024: A leads", small_result, allowed_text="2024") == []
    )


def test_plain_answer_for_a_single_number_and_for_a_table(small_result):
    single = run_query(
        pd.DataFrame({"x": [1.0, 2.5]}), QuerySpec(operation="aggregate", column="x", agg="sum")
    )

    assert "is 3.50" in plain_answer("q", single)
    assert "| city" in plain_answer("q", small_result)
