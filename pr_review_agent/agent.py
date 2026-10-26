"""The hand-written agent loop: ask the model, run the tools it asks for, repeat."""

from __future__ import annotations

from dataclasses import dataclass, field

from pr_review_agent.llm import LLMClient, Usage
from pr_review_agent.tools import Tool, default_tools, run_tool

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
    usage, called = Usage(), []

    for iteration in range(1, max_iterations + 1):
        response = llm.create(SYSTEM_PROMPT, messages, schemas)
        usage.add(response.usage)
        messages.append({"role": "assistant", "content": response.assistant_content})

        if response.stop_reason == "refusal":
            return AgentResult(None, "refused", iteration, called, usage)
        if not response.tool_calls:  # the model only talked; remind it how to finish
            messages.append({"role": "user", "content": NUDGE})
            continue

        results = []
        for call in response.tool_calls:
            called.append(call.name)
            if call.name == SUBMIT_REVIEW["name"]:
                try:
                    return AgentResult(Review.from_tool_input(call.input), "completed", iteration, called, usage)
                except ValueError as e:  # malformed review: send the error back and let it retry
                    results.append(_tool_result(call.id, f"Error: {e}", is_error=True))
                    continue
            result = run_tool(tools, call.name, call.input, tool_timeout_s)
            results.append(_tool_result(call.id, result.content, result.is_error))
        messages.append({"role": "user", "content": results})  # all results of a turn in one message

    return AgentResult(None, "max_iterations", max_iterations, called, usage)


def _tool_result(tool_use_id: str, content: str, is_error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block
