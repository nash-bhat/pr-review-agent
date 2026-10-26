# pr-review-agent

## Goal
A command-line agent that reviews a GitHub pull request. Given a PR URL it
fetches the diff, runs checks through tools, and prints a structured review
(summary, issues with file and line, suggestions). It is a learning and
portfolio project: the agent loop is hand-written on the Anthropic Python SDK's
tool-use API, with no agent frameworks (no LangChain, LangGraph, CrewAI, and not
the SDK's own Tool Runner either).

## Architecture
```
cli.py ──► agent.run_agent(pr_url, llm, tools)
              │  loop (max 10 iterations):
              │    llm.create(system, messages, tool schemas)
              │    model asks for a tool ──► tools.run_tool (timeout, errors → text)
              │    model calls submit_review ──► Review, stop
              ▼
           llm.py:   LLMClient protocol
                     ├─ AnthropicClient  (real API)
                     └─ FakeLLM          (scripted replies, tests only, no network)
           tools.py: fetch_pr_diff, run_linter (ruff), search_docs (stub)
```
- **The model drives.** It gets only the PR URL and decides which tools to call.
- **`submit_review` is the exit.** The final review is a tool call whose input
  schema *is* the review. That gives structured output and an explicit stop
  signal. It is defined in `agent.py`, not `tools.py`: the loop handles it
  itself and never executes it.
- **Errors are data.** A failing, slow or unknown tool becomes a `tool_result`
  with `is_error: true` so the model can react; the loop never crashes on a tool.
- **`LLMResponse.assistant_content` is opaque.** The loop appends it to the
  history unchanged (real SDK blocks, including thinking blocks, must be
  passed back as-is). The loop only *reads* the parsed `tool_calls`.

## Folder layout
- `pr_review_agent/llm.py` – `LLMClient` protocol, `LLMResponse`, `AnthropicClient`, `FakeLLM` + script helpers (`tool_call`, `text_reply`).
- `pr_review_agent/tools.py` – `Tool` dataclass, the three tools, `run_tool` (timeout + error handling), `default_tools()`.
- `pr_review_agent/agent.py` – system prompt, `submit_review` schema, `Review`/`Issue`, `run_agent` loop and its limits.
- `pr_review_agent/cli.py` – `review <pr-url>` entry point and output formatting. Exit codes: 0 review printed, 1 no review (limit hit or refusal), 2 setup/API error.
- `tests/` – pytest; every test uses `FakeLLM` and never touches the network (`test_tools.py`, `test_agent.py`, `test_cli.py`).
- `README.md` – user-facing setup, usage and known limitations.

## Coding conventions
- Python 3.11+, type hints everywhere, dataclasses for plain data.
- Tools are plain functions taking keyword args and returning a string.
  Raise an exception for failure; `run_tool` turns it into an error result.
- No network or API key in tests: fake the LLM with `FakeLLM` and fake HTTP
  by passing a stub `http_get` to `fetch_pr_diff`.
- Keep the loop short and readable (one screen); put helpers next to it.
- Lint with `ruff check .`.

## Configuration
- `ANTHROPIC_API_KEY` – required for real runs only.
- `PR_REVIEW_MODEL` – model id, default `claude-haiku-5-5` (the cheapest current model).
- `GITHUB_TOKEN` – optional; raises GitHub's rate limit and allows private repos.

## Run and test
```bash
pip install -e ".[dev]"
pytest                      # offline, no API key needed
ruff check .
review https://github.com/OWNER/REPO/pull/123     # real run, needs ANTHROPIC_API_KEY
```
