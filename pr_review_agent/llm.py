"""The agent's "brain" behind one small interface, so tests can swap in a fake."""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-haiku-5-5"  # cheapest current Claude model; override with PR_REVIEW_MODEL


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
