# Sampling Demo

A terminal UI for exploring **MCP (Model Context Protocol) sampling** — watch JSON-RPC messages flow between a host, MCP client, and MCP server in real time while chatting with an agent.

## What it does

- **Left panel** — chat window: send queries, approve/deny sampling requests, and see formatted agent responses with markdown rendering
- **Right panel** — live JSON-RPC inspector: four panes show every message sent and received between the client and server, grouped by direction
- **Flow diagram** — animated diagram at the top of the right panel highlights the active message direction and method name as traffic flows

## Requirements

- [uv](https://docs.astral.sh/uv/)
- `OPENAI_API_KEY` — used by the client-side sampling handler to call the LLM
- `TAVILY_API_KEY` — used by the `research_agent` tool for web search
- Optional: `OPENAI_MODEL` (default: `gpt-4.1`)

## Run

```bash
uv run main.py
```

## Tools

The MCP server (`sampling_server.py`) exposes four tools:

| Tool             | Trigger                                | Description                             |
| ---------------- | -------------------------------------- | --------------------------------------- |
| `research_agent` | any plain text query                   | Web research via Tavily + LLM synthesis |
| `summarize_pdf`  | query containing `.pdf`                | Summarize a PDF file at the given path  |
| `calculator`     | `/use calculator {"expression":"..."}` | Evaluate an arithmetic expression       |

You can also call any tool directly with:

```MCPSamplingTUI
/use <tool-name> <json-args>
```

Example:

```MCPSamplingTUI
/use calculator {"expression": "(2 + 3) * 4"}
```

## Keyboard shortcuts

| Key           | Action                  |
| ------------- | ----------------------- |
| `Enter`       | Submit message          |
| `Shift+Enter` | Insert newline in input |
| `Ctrl+L`      | Clear JSON-RPC logs     |
| `Ctrl+C`      | Quit                    |

## Sampling approval

Tools that use `ctx.sample(...)` (like `research_agent`) trigger a sampling request that requires your approval before the LLM is called. The chat window will prompt you — type `y` and press Enter to approve, `n` to deny.
