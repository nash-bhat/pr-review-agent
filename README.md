# pr-review-agent

A command-line agent that reviews GitHub pull requests. Give it a PR URL and it
fetches the diff, lints the changed Python files, and prints a structured review.
The agent loop is written by hand on the Anthropic Python SDK's tool-use API:
no agent frameworks.

```
$ review https://github.com/OWNER/REPO/pull/12
Summary
  Adds a user lookup helper. One real bug: the query is built with an f-string.

Issues (2)
  [HIGH] db.py:14  (sql-injection)
      The SQL string interpolates user input; use a parameterized query.
  [LOW] utils.py:1  (unused-import)
      `os` is imported but never used.

Suggestions
  - Add a test for a user name containing a quote.

3 iteration(s) · tools: fetch_pr_diff, run_linter, submit_review · 15,210 input / 410 output tokens
```
(Illustrative output; your numbers will differ.)

## How it works
1. The model gets only the PR URL plus four tool definitions.
2. It calls `fetch_pr_diff`, may call `run_linter` and `search_docs`, and reads each result.
3. It ends by calling `submit_review`, whose input schema *is* the review. The loop stops there.

Safety limits: at most 10 model calls per run, a timeout on every tool call, and
tool errors (a 404, a crash, an unknown tool name) are sent back to the model as
error results instead of crashing the run. See `CLAUDE.md` for the architecture.

## Setup
Python 3.11+.
```bash
git clone https://github.com/nash-bhat/pr-review-agent && cd pr-review-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```
For real runs, put your key in a `.env` file (it is git-ignored) and load it:
```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
set -a; source .env; set +a
```
Optional settings:
- `PR_REVIEW_MODEL`: model id, default `claude-haiku-5-5` (the cheapest current model).
- `GITHUB_TOKEN`: raises GitHub's API rate limit and allows private repos.

## Usage
```bash
review https://github.com/OWNER/REPO/pull/123          # human-readable review
review https://github.com/OWNER/REPO/pull/123 --json   # full result as JSON
review URL --model claude-sonnet-5-5 --max-iterations 6
```
Exit codes: `0` review printed, `1` the agent stopped without a review, `2` setup
or API error (e.g. no API key).

## Tests
```bash
pytest          # offline: a scripted FakeLLM stands in for Claude, no API key needed
ruff check .
```

## Known limitations
- `run_linter` lints code the model copies from the diff, which is often a
  partial file. Expect some findings caused by missing context.
- Line numbers come from the model reading the diff's hunk headers, so they can be off.
- Lint rules are fixed (`E4,E7,E9,F,B`), not the reviewed repo's own config, so
  framework idioms such as FastAPI's `Depends()` defaults get flagged.
- `search_docs` is a stub with canned answers.
