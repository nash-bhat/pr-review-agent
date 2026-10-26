"""`review <pr-url>`: run the agent on a PR and print the review."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import asdict

from pr_review_agent.agent import MAX_ITERATIONS, AgentResult, Review, run_agent
from pr_review_agent.llm import AnthropicClient, GroqClient, LLMClient

PROVIDERS: dict[str, Callable[[str | None], LLMClient]] = {"anthropic": AnthropicClient, "groq": GroqClient}
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


def main(argv: list[str] | None = None, make_llm: Callable[[str | None], LLMClient] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="review", description="Review a GitHub pull request with an LLM agent.")
    parser.add_argument("pr_url", help="e.g. https://github.com/OWNER/REPO/pull/123")
    parser.add_argument("--provider", choices=sorted(PROVIDERS),
                        default=os.environ.get("PR_REVIEW_PROVIDER", "anthropic"),
                        help="LLM provider (default: $PR_REVIEW_PROVIDER or anthropic)")
    parser.add_argument("--model", help="model id (default: $PR_REVIEW_MODEL / $GROQ_MODEL or the provider default)")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="log each turn to stderr (-v: summaries, -vv: also full tool results)")
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
                            level=logging.DEBUG if args.verbose > 1 else logging.INFO)
        # Only our logger: the SDK's and urllib's debug logs would include request headers.
        logging.getLogger().handlers[0].addFilter(logging.Filter("pr_review_agent"))

    try:
        llm = (make_llm or PROVIDERS[args.provider])(args.model)
        result = run_agent(args.pr_url, llm, max_iterations=args.max_iterations)
    except Exception as e:  # noqa: BLE001 - missing API key, network or API errors: report, don't trace
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        if "authentication" in str(e).lower() or "API_KEY" in str(e):
            key = "GROQ_API_KEY" if args.provider == "groq" else "ANTHROPIC_API_KEY"
            print(f"hint: export {key}=... and try again (see README: Setup)", file=sys.stderr)
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
