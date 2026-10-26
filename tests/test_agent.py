import time

from pr_review_agent.agent import NUDGE, SUBMIT_REVIEW, run_agent
from pr_review_agent.llm import FakeLLM, LLMResponse, text_reply, tool_call
from pr_review_agent.tools import Tool, default_tools

PR = "https://github.com/octo/demo/pull/7"
FAKE_DIFF = "diff --git a/utils.py b/utils.py\n@@ -0,0 +1,3 @@\n+import os\n+def name(u):\n+    return u.name\n"
REVIEW = {
    "summary": "Adds a helper. One unused import.",
    "issues": [{"file": "utils.py", "line": 1, "severity": "low", "category": "unused-import",
                "message": "`os` is imported but never used."}],
    "suggestions": ["Remove the import."],
}


def offline_tools(fetch=lambda pr_url: FAKE_DIFF) -> dict[str, Tool]:
    """The real tools, except fetch_pr_diff, which returns a canned diff instead of calling GitHub."""
    tools = default_tools()
    tools["fetch_pr_diff"].func = fetch
    return tools


def last_message(fake: FakeLLM, call_index: int) -> dict:
    return fake.calls[call_index]["messages"][-1]


def test_happy_path_fetch_lint_submit():
    fake = FakeLLM([
        tool_call("fetch_pr_diff", {"pr_url": PR}),
        tool_call("run_linter", {"file_path": "utils.py", "code": "import os\n"}),
        tool_call("submit_review", REVIEW),
    ])
    result = run_agent(PR, fake, offline_tools())

    assert result.status == "completed"
    assert result.iterations == 3
    assert result.tool_calls == ["fetch_pr_diff", "run_linter", "submit_review"]
    assert result.review.issues[0].category == "unused-import"
    assert result.usage.input_tokens == 300  # 3 turns x 100 fake tokens

    # Turn 2 received the diff as a tool_result linked to turn 1's tool_use id.
    turn1_call = fake.calls[1]["messages"][-2]["content"][0]
    result_block = last_message(fake, 1)["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == turn1_call["id"]
    assert result_block["content"] == FAKE_DIFF
    # Turn 3 received ruff's real output.
    assert "F401" in last_message(fake, 2)["content"][0]["content"]


def test_model_receives_tools_prompt_and_pr_url():
    fake = FakeLLM([tool_call("submit_review", REVIEW)])
    run_agent(PR, fake, offline_tools())
    first = fake.calls[0]
    assert first["messages"] == [{"role": "user", "content": f"Review this pull request: {PR}"}]
    assert "submit_review" in first["system"]
    assert [t["name"] for t in first["tools"]] == ["fetch_pr_diff", "run_linter", "search_docs", "submit_review"]


def test_tool_error_is_sent_to_model_not_raised():
    def not_found(pr_url):
        raise RuntimeError("PR not found (404)")

    fake = FakeLLM([
        tool_call("fetch_pr_diff", {"pr_url": PR}),
        tool_call("submit_review", {**REVIEW, "summary": "Could not fetch the PR.", "issues": []}),
    ])
    result = run_agent(PR, fake, offline_tools(fetch=not_found))

    block = last_message(fake, 1)["content"][0]
    assert block["is_error"] is True
    assert "PR not found (404)" in block["content"]
    assert result.status == "completed"
    assert result.review.summary == "Could not fetch the PR."


def test_unknown_tool_is_reported_and_loop_continues():
    fake = FakeLLM([tool_call("delete_repo", {}), tool_call("submit_review", REVIEW)])
    result = run_agent(PR, fake, offline_tools())
    assert "no tool named 'delete_repo'" in last_message(fake, 1)["content"][0]["content"]
    assert result.status == "completed"


def test_slow_tool_times_out_inside_the_loop():
    tools = offline_tools(fetch=lambda pr_url: time.sleep(2) or FAKE_DIFF)
    fake = FakeLLM([tool_call("fetch_pr_diff", {"pr_url": PR}), tool_call("submit_review", REVIEW)])
    result = run_agent(PR, fake, tools, tool_timeout_s=0.1)
    assert "timed out" in last_message(fake, 1)["content"][0]["content"]
    assert result.status == "completed"


def test_stops_after_max_iterations():
    # 11 scripted turns, but the loop may only use 10: one must be left unused.
    fake = FakeLLM([tool_call("search_docs", {"query": "sql"}) for _ in range(11)])
    result = run_agent(PR, fake, offline_tools())
    assert result.status == "max_iterations"
    assert result.review is None
    assert len(fake.calls) == 10
    assert len(fake.script) == 1


def test_text_only_reply_gets_a_nudge():
    fake = FakeLLM([text_reply("Looks fine to me!"), tool_call("submit_review", REVIEW)])
    result = run_agent(PR, fake, offline_tools())
    assert last_message(fake, 1) == {"role": "user", "content": NUDGE}
    assert result.status == "completed" and result.iterations == 2


def test_invalid_review_is_rejected_and_retried():
    fake = FakeLLM([tool_call("submit_review", {"issues": []}), tool_call("submit_review", REVIEW)])
    result = run_agent(PR, fake, offline_tools())
    block = last_message(fake, 1)["content"][0]
    assert block["is_error"] is True and "invalid submit_review input" in block["content"]
    assert result.status == "completed" and result.iterations == 2


def test_refusal_stops_cleanly():
    fake = FakeLLM([LLMResponse([], [], "", "refusal")])
    result = run_agent(PR, fake, offline_tools())
    assert result.status == "refused" and result.review is None


def test_submit_review_schema_is_strict():
    assert SUBMIT_REVIEW["strict"] is True
    assert SUBMIT_REVIEW["input_schema"]["additionalProperties"] is False
