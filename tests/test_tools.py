import time
import urllib.error

import pytest

from pr_review_agent import tools
from pr_review_agent.tools import Tool, default_tools, fetch_pr_diff, run_linter, run_tool, search_docs

PR = "https://github.com/octo/demo/pull/7"


def fake_get(body: str):
    seen = {}

    def get(url, headers):
        seen["url"], seen["headers"] = url, headers
        return body

    return get, seen


def raising_get(code: int):
    def get(url, headers):
        raise urllib.error.HTTPError(url, code, "err", {}, None)

    return get


# --- fetch_pr_diff ---


def test_fetch_pr_diff_calls_github_api(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    get, seen = fake_get("diff --git a/x.py b/x.py\n+print(1)\n")
    assert fetch_pr_diff(PR, http_get=get).startswith("diff --git")
    assert seen["url"] == "https://api.github.com/repos/octo/demo/pulls/7"
    assert seen["headers"]["Accept"] == "application/vnd.github.diff"
    assert "Authorization" not in seen["headers"]


def test_fetch_pr_diff_uses_token_when_set(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    get, seen = fake_get("diff")
    fetch_pr_diff(PR, http_get=get)
    assert seen["headers"]["Authorization"] == "Bearer ghp_secret"


def test_fetch_pr_diff_404_has_clear_message():
    with pytest.raises(RuntimeError, match="PR not found"):
        fetch_pr_diff(PR, http_get=raising_get(404))


def test_fetch_pr_diff_rejects_non_pr_url():
    with pytest.raises(ValueError, match="not a GitHub PR URL"):
        fetch_pr_diff("https://github.com/octo/demo/issues/7", http_get=fake_get("")[0])


def test_fetch_pr_diff_truncates_huge_diffs(monkeypatch):
    monkeypatch.setattr(tools, "MAX_DIFF_CHARS", 10)
    out = fetch_pr_diff(PR, http_get=fake_get("x" * 25)[0])
    assert out.startswith("x" * 10)
    assert "[diff truncated: 15 more characters not shown]" in out


# --- run_linter ---


def test_run_linter_reports_findings_with_lines():
    out = run_linter("pkg/utils.py", "import os\n\ndef f(items=[]):\n    return items\n")
    assert "pkg/utils.py:1:8: F401" in out
    assert "B006" in out  # mutable default argument


def test_run_linter_clean_code():
    assert run_linter("ok.py", "def add(a, b):\n    return a + b\n") == "ruff: no issues in ok.py"


def test_run_linter_skips_non_python():
    assert run_linter("README.md", "# hi").startswith("Skipped")


# --- search_docs ---


def test_search_docs_stub():
    assert "parameterized" in search_docs("SQL injection risk")
    assert search_docs("kubernetes").startswith("No documentation found")


# --- run_tool: the safety wrapper ---


def test_run_tool_success():
    result = run_tool(default_tools(), "search_docs", {"query": "sql"}, timeout_s=5)
    assert not result.is_error and "parameterized" in result.content


def test_run_tool_unknown_tool():
    result = run_tool(default_tools(), "delete_repo", {}, timeout_s=5)
    assert result.is_error and "no tool named 'delete_repo'" in result.content


def test_run_tool_bad_arguments():
    result = run_tool(default_tools(), "search_docs", {"q": "sql"}, timeout_s=5)
    assert result.is_error and "bad arguments" in result.content


def test_run_tool_exception_becomes_error_text():
    def boom(**_):
        raise RuntimeError("disk on fire")

    result = run_tool({"boom": Tool("boom", "", {}, boom)}, "boom", {}, timeout_s=5)
    assert result.is_error and "boom failed: disk on fire" in result.content


def test_run_tool_timeout():
    slow = Tool("slow", "", {}, lambda: time.sleep(2) or "done")
    start = time.monotonic()
    result = run_tool({"slow": slow}, "slow", {}, timeout_s=0.1)
    assert result.is_error and "timed out after 0.1s" in result.content
    assert time.monotonic() - start < 1  # the loop did not wait for the slow tool
