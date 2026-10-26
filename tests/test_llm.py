import json

import pytest

from pr_review_agent.agent import run_agent
from pr_review_agent.llm import GroqClient, to_openai_messages
from pr_review_agent.tools import default_tools

PR = "https://github.com/octo/demo/pull/7"
REVIEW = {"summary": "Fine.", "issues": [], "suggestions": []}


def groq_reply(*, content=None, calls=(), finish="tool_calls"):
    """A response shaped like Groq's (OpenAI-compatible) chat completions API."""
    tool_calls = [{"id": f"call_{i}", "type": "function",
                   "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}
                  for i, (name, args) in enumerate(calls)]
    message = {"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {})}
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5}}


def fake_post(replies):
    sent = []

    def post(url, body, headers):
        sent.append({"url": url, "body": body, "headers": headers})
        return replies.pop(0)

    return post, sent


@pytest.fixture
def groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.delenv("GROQ_MODEL", raising=False)


def test_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        GroqClient()


def test_groq_full_agent_run_translates_both_ways(groq_key):
    post, sent = fake_post([
        groq_reply(content="Fetching.", calls=[("fetch_pr_diff", {"pr_url": PR})]),
        groq_reply(calls=[("submit_review", REVIEW)]),
    ])
    tools = default_tools()
    tools["fetch_pr_diff"].func = lambda pr_url: "diff --git a/x.py b/x.py\n+x = 1\n"

    result = run_agent(PR, GroqClient(post=post), tools)

    assert result.status == "completed" and result.review.summary == "Fine."
    assert result.usage.input_tokens == 100

    first = sent[0]
    assert first["url"] == GroqClient.URL
    assert first["headers"]["Authorization"] == "Bearer gsk_test"
    assert first["body"]["model"] == "llama-3.3-70b-versatile"
    assert first["body"]["messages"][0]["role"] == "system"
    fn = first["body"]["tools"][-1]["function"]
    assert fn["name"] == "submit_review" and "strict" not in fn  # Anthropic-only field is dropped

    # Turn 2: the assistant's tool call and the tool result in OpenAI's shapes.
    assistant, tool_msg = sent[1]["body"]["messages"][-2:]
    assert assistant["content"] == "Fetching."
    assert assistant["tool_calls"][0]["function"] == {"name": "fetch_pr_diff", "arguments": json.dumps({"pr_url": PR})}
    assert tool_msg == {"role": "tool", "tool_call_id": "call_0", "content": "diff --git a/x.py b/x.py\n+x = 1\n"}


def test_groq_invalid_json_arguments_become_a_tool_error(groq_key):
    post, sent = fake_post([
        groq_reply(calls=[("search_docs", "{not json")]),
        groq_reply(calls=[("submit_review", REVIEW)]),
    ])
    result = run_agent(PR, GroqClient(post=post), default_tools())
    tool_msg = sent[1]["body"]["messages"][-1]
    assert tool_msg["role"] == "tool" and "bad arguments for search_docs" in tool_msg["content"]
    assert result.status == "completed"


def test_groq_text_only_reply_maps_to_end_turn(groq_key):
    post, _ = fake_post([groq_reply(content="All good.", finish="stop")])
    response = GroqClient(post=post).create("sys", [{"role": "user", "content": "hi"}], [])
    assert response.stop_reason == "end_turn" and response.text == "All good." and response.tool_calls == []


def test_to_openai_messages_splits_multiple_tool_results():
    messages = [
        {"role": "user", "content": "Review it"},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "one"},
            {"type": "tool_result", "tool_use_id": "b", "content": "two", "is_error": True},
        ]},
    ]
    out = to_openai_messages("sys", messages)
    assert [m["role"] for m in out] == ["system", "user", "tool", "tool"]
    assert out[3] == {"role": "tool", "tool_call_id": "b", "content": "two"}
