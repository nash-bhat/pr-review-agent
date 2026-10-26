"""Tools the model can call. Each is a plain function returning a string."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path

# Rules that point at likely bugs, not style: syntax/runtime errors, pyflakes, bugbear.
LINT_RULES = "E4,E7,E9,F,B"
MAX_DIFF_CHARS = 60_000  # ~15k tokens; bigger diffs are cut with a visible note
PR_URL = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)/?$")


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    func: Callable[..., str]

    def schema(self) -> dict:
        """The definition sent to the model (everything except the Python function)."""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


def run_tool(tools: dict[str, Tool], name: str, args: dict, timeout_s: float) -> ToolResult:
    """Run one tool call safely: unknown names, bad args, exceptions and timeouts become error text."""
    tool = tools.get(name)
    if tool is None:
        return ToolResult(f"Error: no tool named {name!r}. Available: {', '.join(sorted(tools))}", True)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return ToolResult(pool.submit(tool.func, **args).result(timeout=timeout_s))
    except FutureTimeout:
        return ToolResult(f"Error: {name} timed out after {timeout_s:g}s", True)
    except TypeError as e:  # wrong or missing arguments from the model
        return ToolResult(f"Error: bad arguments for {name}: {e}", True)
    except Exception as e:  # noqa: BLE001 - deliberate: no tool failure may crash the agent
        return ToolResult(f"Error: {name} failed: {e}", True)
    finally:
        # Python can't kill a thread: a timed-out tool keeps running in the background.
        # Don't wait for it; the loop moves on.
        pool.shutdown(wait=False)


# --- fetch_pr_diff ----------------------------------------------------------------------------


def _github_get(url: str, headers: dict[str, str]) -> str:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_pr_diff(pr_url: str, http_get: Callable[[str, dict[str, str]], str] = _github_get) -> str:
    match = PR_URL.match(pr_url.strip())
    if not match:
        raise ValueError(f"not a GitHub PR URL (expected https://github.com/OWNER/REPO/pull/N): {pr_url}")
    owner, repo, number = match.groups()
    headers = {"Accept": "application/vnd.github.diff", "User-Agent": "pr-review-agent"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    try:
        diff = http_get(f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}", headers)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"PR not found (404): {pr_url}. It may not exist or the repo may be private.")
        raise RuntimeError(f"GitHub API returned HTTP {e.code} for {pr_url}")
    if not diff.strip():
        return "The PR has no changes (empty diff)."
    if len(diff) > MAX_DIFF_CHARS:
        cut = len(diff) - MAX_DIFF_CHARS
        diff = diff[:MAX_DIFF_CHARS] + f"\n\n[diff truncated: {cut} more characters not shown]"
    return diff


# --- run_linter -------------------------------------------------------------------------------


def run_linter(file_path: str, code: str, timeout_s: float = 20) -> str:
    if not file_path.endswith(".py"):
        return f"Skipped: run_linter only supports Python files, not {file_path}."
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / Path(file_path).name
        target.write_text(code)
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--isolated", "--no-cache",
             "--select", LINT_RULES, "--output-format", "json", str(target)],
            check=False, capture_output=True, text=True, timeout=timeout_s,
        )
    if proc.returncode not in (0, 1):  # 0 = clean, 1 = findings, anything else = ruff itself failed
        raise RuntimeError(f"ruff failed: {proc.stderr.strip()[:500]}")
    findings = json.loads(proc.stdout or "[]")
    if not findings:
        return f"ruff: no issues in {file_path}"
    lines = [
        f"{file_path}:{f['location']['row']}:{f['location']['column']}: {f['code']} {f['message']}"
        for f in findings
    ]
    return f"ruff found {len(lines)} issue(s):\n" + "\n".join(lines)


# --- search_docs (stub) -----------------------------------------------------------------------

_CANNED_DOCS = {
    "sql": "Use parameterized queries (cursor.execute(sql, params)); never build SQL with f-strings or +.",
    "none": "Check for None before attribute access; prefer early returns or `if x is None: ...`.",
    "exception": "Catch specific exceptions; a bare `except:` also swallows KeyboardInterrupt and SystemExit.",
    "import": "Remove unused imports; they slow startup and hide real dependencies.",
}


def search_docs(query: str) -> str:
    """Stub: keyword lookup in a few canned snippets. To be replaced with real search later."""
    hits = [text for key, text in _CANNED_DOCS.items() if key in query.lower()]
    return "\n".join(f"- {h}" for h in hits) if hits else f"No documentation found for {query!r}."


# --- registry ---------------------------------------------------------------------------------


def default_tools() -> dict[str, Tool]:
    tools = [
        Tool(
            "fetch_pr_diff",
            "Fetch the unified diff of a public GitHub pull request. Call this first to see what changed.",
            {"type": "object", "properties": {"pr_url": {"type": "string", "description": "Full GitHub PR URL"}},
             "required": ["pr_url"]},
            fetch_pr_diff,
        ),
        Tool(
            "run_linter",
            "Run the ruff linter on Python code and return its findings with line numbers. "
            "Pass the file's code as it appears after the change. If you only have diff hunks, the code "
            "is partial: expect findings caused by missing context (e.g. undefined names) and ignore those.",
            {"type": "object",
             "properties": {"file_path": {"type": "string", "description": "Path of the file in the repo"},
                            "code": {"type": "string", "description": "Python source to lint"}},
             "required": ["file_path", "code"]},
            run_linter,
        ),
        Tool(
            "search_docs",
            "Search coding best-practice docs for guidance on an issue, e.g. 'sql injection'.",
            {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            search_docs,
        ),
    ]
    return {t.name: t for t in tools}
