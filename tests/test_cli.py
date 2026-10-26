import json

from pr_review_agent import cli
from pr_review_agent.llm import FakeLLM, tool_call

PR = "https://github.com/octo/demo/pull/7"
REVIEW = {
    "summary": "Adds a query helper.",
    "issues": [
        {"file": "db.py", "line": None, "severity": "low", "category": "naming", "message": "Vague name."},
        {"file": "db.py", "line": 12, "severity": "high", "category": "sql-injection",
         "message": "Query built with an f-string."},
    ],
    "suggestions": ["Use parameterized queries."],
}


def run_cli(capsys, script, *args):
    code = cli.main([PR, *args], make_llm=lambda model: FakeLLM(script))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_prints_structured_review_sorted_by_severity(capsys):
    code, out, err = run_cli(capsys, [tool_call("submit_review", REVIEW)])
    assert code == 0
    assert out.index("[HIGH] db.py:12  (sql-injection)") < out.index("[LOW] db.py  (naming)")
    assert "Issues (2)" in out and "- Use parameterized queries." in out
    assert "1 iteration(s) · tools: submit_review" in err


def test_json_output(capsys):
    code, out, _ = run_cli(capsys, [tool_call("submit_review", REVIEW)], "--json")
    data = json.loads(out)
    assert code == 0 and data["status"] == "completed"
    assert data["review"]["issues"][1]["category"] == "sql-injection"


def test_no_review_exits_1(capsys):
    script = [tool_call("search_docs", {"query": "sql"}) for _ in range(3)]
    code, _, err = run_cli(capsys, script, "--max-iterations", "2")
    assert code == 1
    assert "No review produced: hit the 2-iteration limit." in err


def test_setup_error_exits_2(capsys):
    def no_key(model):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    code = cli.main([PR], make_llm=no_key)
    assert code == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
