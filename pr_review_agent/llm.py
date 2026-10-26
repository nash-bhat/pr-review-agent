"""The agent's "brain" behind one small interface, so tests can swap in a fake."""

from __future__ import annotations

import itertools
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-haiku-5-5"  # cheapest current Claude model; override with PR_REVIEW_MODEL
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"  # Groq's catalogue changes; override with GROQ_MODEL


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


@dataclass
class LLMResponse:
    """One model turn.

    `assistant_content` is appended to the conversation exactly as returned:
    for the real API it holds SDK blocks (text, tool_use, thinking) that must be
    sent back unchanged. The loop only reads the parsed fields below it.
    """

    assistant_content: list[Any]
    tool_calls: list[ToolCall]
    text: str
    stop_reason: str
    usage: Usage = field(default_factory=Usage)


class LLMClient(Protocol):
    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class AnthropicClient:
    """Calls the real Claude API. Needs ANTHROPIC_API_KEY (or another SDK credential)."""

    def __init__(self, model: str | None = None, max_tokens: int = 16000):
        import anthropic  # imported here so tests never need the SDK configured

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("PR_REVIEW_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        return LLMResponse(
            assistant_content=list(response.content),
            tool_calls=[
                ToolCall(id=b.id, name=b.name, input=dict(b.input))
                for b in response.content
                if b.type == "tool_use"
            ],
            text="".join(b.text for b in response.content if b.type == "text"),
            stop_reason=response.stop_reason or "",
            usage=Usage(response.usage.input_tokens, response.usage.output_tokens),
        )


def _post_json(url: str, body: dict, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Groq API returned HTTP {e.code}: {detail}") from e


class GroqClient:
    """Calls an open model on Groq through its OpenAI-compatible API (needs GROQ_API_KEY).

    An adapter: the loop speaks Anthropic's message format, so each call translates the
    conversation to OpenAI's format and the reply back into Anthropic-shaped blocks.
    """

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str | None = None, max_tokens: int = 4096,
                 post: Callable[[str, dict, dict[str, str]], dict] = _post_json):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.max_tokens = max_tokens
        self.post = post

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": to_openai_messages(system, messages),
            "tools": [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                        "parameters": t["input_schema"]}} for t in tools],
        }
        # A User-Agent is required: Groq's CDN rejects Python's default one.
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                   "User-Agent": "pr-review-agent"}
        data = self.post(self.URL, body, headers)
        choice = data["choices"][0]
        message = choice["message"]
        calls = [ToolCall(tc["id"], tc["function"]["name"], _parse_args(tc["function"]["arguments"]))
                 for tc in message.get("tool_calls") or []]
        text = message.get("content") or ""
        content: list[Any] = [{"type": "text", "text": text}] if text else []
        content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in calls]
        stop = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}
        usage = data.get("usage") or {}
        return LLMResponse(content, calls, text, stop.get(choice.get("finish_reason"), "end_turn"),
                           Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)))


def _parse_args(raw: str) -> dict[str, Any]:
    """OpenAI-style APIs send tool arguments as a JSON string, which a model can get wrong."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_invalid_json": raw}  # the tool then fails with "bad arguments" and the model can retry
    return args if isinstance(args, dict) else {"_invalid_json": raw}


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Translate the loop's Anthropic-format history (dict blocks) into OpenAI chat messages."""
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if isinstance(m["content"], str):
            out.append({"role": m["role"], "content": m["content"]})
        elif m["role"] == "assistant":
            text = "".join(b["text"] for b in m["content"] if b["type"] == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                     for b in m["content"] if b["type"] == "tool_use"]
            out.append({"role": "assistant", "content": text or None, **({"tool_calls": calls} if calls else {})})
        else:  # a user message holding tool results: one "tool" message per result
            out += [{"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]}
                    for b in m["content"] if b["type"] == "tool_result"]
    return out


class FakeLLM:
    """Plays back a script of pre-written replies, one per call. For tests only.

    It records every request in `calls` so a test can check what the loop sent,
    e.g. that a tool's result really reached the model on the next turn.
    """

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        # Copy the list: the loop keeps appending to it after this call returns.
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        if not self.script:
            raise AssertionError("FakeLLM script ran out: the loop called the model more times than expected")
        return self.script.pop(0)


_ids = itertools.count(1)


def tool_call(name: str, input: dict[str, Any], text: str = "") -> LLMResponse:
    """Script helper: a model turn that asks for one tool."""
    call = ToolCall(id=f"toolu_fake_{next(_ids)}", name=name, input=input)
    content: list[Any] = [{"type": "text", "text": text}] if text else []
    content.append({"type": "tool_use", "id": call.id, "name": name, "input": input})
    return LLMResponse(content, [call], text, "tool_use", Usage(100, 20))


def text_reply(text: str) -> LLMResponse:
    """Script helper: a model turn that only talks and calls no tool."""
    return LLMResponse([{"type": "text", "text": text}], [], text, "end_turn", Usage(100, 20))
