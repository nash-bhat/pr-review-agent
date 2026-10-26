"""The hand-written agent loop: ask the model, run the tools it asks for, repeat."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from pr_review_agent.llm import LLMClient, LLMResponse, Usage
from pr_review_agent.tools import Tool, ToolResult, default_tools, run_tool

log = logging.getLogger("pr_review_agent")  # silent unless the CLI's -v / -vv turns it on

MAX_ITERATIONS = 10
TOOL_TIMEOUT_S = 30.0

SYSTEM_PROMPT = """You are a careful senior engineer reviewing a GitHub pull request.

How to work:
1. Call fetch_pr_diff with the PR URL to see the changes.
2. For changed Python files, you may call run_linter on the new code. Only report linter findings that \
point at real problems in lines the PR changed.
3. Optionally call search_docs for guidance on an issue you found.
4. Finish by calling submit_review exactly once. That is the only way to deliver the review.

Rules for the review:
- Report only issues you can point to in the diff. Never invent files or lines.
- `line` is the line number in the new version of the file (from the diff's +start,count hunk headers), \
or null if the issue is not about one line.
- Prefer a few precise issues over many vague ones. A clean PR may have no issues.
- If a tool fails (e.g. the PR cannot be fetched), say so in the summary and submit what you can."""

SUBMIT_REVIEW = {
    "name": "submit_review",
    "description": "Submit the final code review. Call this exactly once, when you are done.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "issues", "suggestions"],
        "properties": {
            "summary": {"type": "string", "description": "2-4 sentences: what the PR does and its overall quality"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["file", "line", "severity", "category", "message"],
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": ["integer", "null"]},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "category": {"type": "string", "description": "short slug, e.g. null-check, sql-injection"},
                        "message": {"type": "string"},
                    },
                },
            },
            "suggestions": {"type": "array", "items": {"type": "string"}},
        },
    },
}

NUDGE = "You have not submitted the review yet. Call submit_review now with your findings."
LEAKED_MARKUP = re.compile(r"</?(parameter|invoke|function_calls)\b|</(summary|issues|suggestions)>")


@dataclass
class Issue:
    file: str
    line: int | None
    severity: str
    category: str
    message: str


@dataclass
class Review:
    summary: str
    issues: list[Issue]
    suggestions: list[str]

    @classmethod
    def from_tool_input(cls, data: dict) -> Review:
        """Validate submit_review's input. Raises ValueError so the model can be told what to fix."""
        try:
            issues = [Issue(**issue) for issue in data["issues"]]
            review = cls(str(data["summary"]), issues, [str(s) for s in data["suggestions"]])
        except (KeyError, TypeError) as e:
            raise ValueError(f"invalid submit_review input: {e!r}") from e
        for issue in issues:
            if issue.severity not in ("low", "medium", "high"):
                raise ValueError(f"invalid severity {issue.severity!r}; use low, medium or high")
        # Seen in a real run: the model wrote its issues as tool-call markup inside `summary`
        # and sent `issues: []`. The schema can't catch that, since every field has the right type.
        texts = [review.summary, *review.suggestions, *(i.message for i in issues)]
        if any(LEAKED_MARKUP.search(t) for t in texts):
            raise ValueError("tool-call markup (e.g. '<parameter name=...>') found inside a text field. "
                             "Put each issue in the `issues` array as a JSON object and resubmit.")
        return review


@dataclass
class AgentResult:
    review: Review | None
    status: str  # "completed" | "max_iterations" | "refused"
    iterations: int
    tool_calls: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def run_agent(
    pr_url: str,
    llm: LLMClient,
    tools: dict[str, Tool] | None = None,
    max_iterations: int = MAX_ITERATIONS,
    tool_timeout_s: float = TOOL_TIMEOUT_S,
) -> AgentResult:
    tools = tools if tools is not None else default_tools()
    schemas = [t.schema() for t in tools.values()] + [SUBMIT_REVIEW]
    messages: list[dict] = [{"role": "user", "content": f"Review this pull request: {pr_url}"}]
    usage, called, already_sent = Usage(), [], 0
    log.debug("system prompt (sent every turn):\n%s", SYSTEM_PROMPT)
    log.debug("tool definitions (sent every turn):\n%s", json.dumps(schemas, indent=2))
    log.info("user message: %s", messages[0]["content"])

    for iteration in range(1, max_iterations + 1):
        log.info("── turn %d/%d: sending %d message(s) to the model", iteration, max_iterations, len(messages))
        _log_request(messages, already_sent)
        already_sent = len(messages)
        started = time.monotonic()
        response = llm.create(SYSTEM_PROMPT, messages, schemas)
        _log_response(response, time.monotonic() - started, usage)
        usage.add(response.usage)
        messages.append({"role": "assistant", "content": response.assistant_content})

        if response.stop_reason == "refusal":
            log.warning("model refused; stopping")
            return AgentResult(None, "refused", iteration, called, usage)
        if not response.tool_calls:  # the model only talked; remind it how to finish
            log.info("no tool call: nudging the model. user message: %s", NUDGE)
            messages.append({"role": "user", "content": NUDGE})
            continue

        results = []
        for call in response.tool_calls:
            called.append(call.name)
            log.info("tool call → %s(%s)", call.name, _short(json.dumps(call.input), 300))
            if call.name == SUBMIT_REVIEW["name"]:
                try:
                    review = Review.from_tool_input(call.input)
                    log.info("review accepted: %d issue(s); stopping after %d turn(s)", len(review.issues), iteration)
                    return AgentResult(review, "completed", iteration, called, usage)
                except ValueError as e:  # malformed review: send the error back and let it retry
                    log.warning("review rejected: %s", e)
                    results.append(_tool_result(call.id, f"Error: {e}", is_error=True))
                    continue
            started = time.monotonic()
            result = run_tool(tools, call.name, call.input, tool_timeout_s)
            _log_tool_result(call.name, result, time.monotonic() - started)
            results.append(_tool_result(call.id, result.content, result.is_error))
        messages.append({"role": "user", "content": results})  # all results of a turn in one message

    log.warning("hit the %d-turn limit without a review", max_iterations)
    return AgentResult(None, "max_iterations", max_iterations, called, usage)


def _log_request(messages: list[dict], already_sent: int) -> None:
    """List every message in this request; earlier ones are re-sent history, the rest are new."""
    for i, m in enumerate(messages):
        tag = "re-sent" if i < already_sent else "NEW"
        content = m["content"]
        parts = [_short(content, 120)] if isinstance(content, str) else [_describe(b) for b in content]
        log.info("   [%d] %-9s %-7s %s", i + 1, m["role"], tag, " | ".join(parts))


def _describe(block) -> str:
    """One-line summary of a content block (a plain dict, or an SDK object from the real API)."""
    get = block.get if isinstance(block, dict) else lambda k, d=None: getattr(block, k, d)
    kind = get("type")
    if kind == "text":
        return f"text {_short(get('text', ''), 80)!r}"
    if kind == "thinking":
        return f"thinking ({len(get('thinking') or '')} chars shown; sent back as-is)"
    if kind == "tool_use":
        return f"tool_use {get('name')} {_short(json.dumps(get('input')), 60)}"
    if kind == "tool_result":
        status = "ERROR" if get("is_error") else "ok"
        return f"tool_result {status} {len(get('content') or '')} chars {_short(get('content') or '', 50)!r}"
    return str(kind)


def _short(text: str, limit: int) -> str:
    text = text.replace("\n", "⏎")
    return text if len(text) <= limit else f"{text[:limit]}… (+{len(text) - limit} chars)"


def _log_response(response: LLMResponse, seconds: float, before: Usage) -> None:
    u = response.usage
    log.info("model replied in %.1fs · stop_reason=%s · +%d input / +%d output tokens (run total %d / %d)",
             seconds, response.stop_reason, u.input_tokens, u.output_tokens,
             before.input_tokens + u.input_tokens, before.output_tokens + u.output_tokens)
    if response.thinking:
        log.info("model thinking: %s", _short(response.thinking, 600))
    if response.text:
        log.info("model text: %s", _short(response.text, 600))


def _log_tool_result(name: str, result: ToolResult, seconds: float) -> None:
    status = "ERROR" if result.is_error else "ok"
    log.info("tool result ← %s %s in %.2fs, %d chars: %s", name, status, seconds, len(result.content),
             _short(result.content, 300))
    log.debug("full %s result:\n%s", name, result.content)


def _tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block
