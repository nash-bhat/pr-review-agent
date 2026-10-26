"""`review <pr-url>`: run the agent on a PR and print the review."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict

from pr_review_agent.agent import MAX_ITERATIONS, AgentResult, Review, run_agent
from pr_review_agent.llm import AnthropicClient, LLMClient

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def format_review(review: Review) -> str:
    lines = ["Summary", f"  {review.summary}", "", f"Issues ({len(review.issues)})"]
    if not review.issues:
        lines.append("  none found")
    for issue in sorted(review.issues, key=lambda i: (SEVERITY_ORDER[i.severity], i.file, i.line or 0)):
        where = f"{issue.file}:{issue.line}" if issue.line is not None else issue.file
        lines.append(f"  [{issue.severity.upper()}] {where}  ({issue.category})")
        lines.append(f"      {issue.message}")
    lines += ["", "Suggestions"]
    lines += [f"  - {s}" for s in review.suggestions] or ["  none"]
    return "\n".join(lines)


def format_footer(result: AgentResult) -> str:
    tools = ", ".join(result.tool_calls) or "none"
    return (f"{result.iterations} iteration(s) · tools: {tools} · "
            f"{result.usage.input_tokens:,} input / {result.usage.output_tokens:,} output tokens")


def main(argv: list[str] | None = None, make_llm: Callable[[str | None], LLMClient] = AnthropicClient) -> int:
    parser = argparse.ArgumentParser(prog="review", description="Review a GitHub pull request with Claude.")
    parser.add_argument("pr_url", help="e.g. https://github.com/OWNER/REPO/pull/123")
    parser.add_argument("--model", help="Claude model id (default: $PR_REVIEW_MODEL or claude-haiku-5-5)")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    try:
        result = run_agent(args.pr_url, make_llm(args.model), max_iterations=args.max_iterations)
    except Exception as e:  # noqa: BLE001 - missing API key, network or API errors: report, don't trace
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        if "authentication" in str(e).lower():
            print("hint: export ANTHROPIC_API_KEY=... and try again (see README: Setup)", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    elif result.review is not None:
        print(format_review(result.review))
    else:
        reason = {"max_iterations": f"hit the {args.max_iterations}-iteration limit",
                  "refused": "the model declined the request"}.get(result.status, result.status)
        print(f"No review produced: {reason}.", file=sys.stderr)
    print(f"\n{format_footer(result)}", file=sys.stderr)
    return 0 if result.review is not None else 1


if __name__ == "__main__":
    sys.exit(main())
