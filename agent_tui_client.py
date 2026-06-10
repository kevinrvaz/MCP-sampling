from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, cast

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from fastmcp.client.sampling import create_sampling_callback
from fastmcp.client.sampling.handlers.openai import OpenAISamplingHandler
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.message import SessionMessage
from mcp.types import (
    SamplingCapability,
    SamplingToolsCapability,
    Tool,
)
from rich.markdown import Markdown
from rich.markup import escape
from rich.syntax import Syntax
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Label,
    RichLog,
    Static,
    TextArea,
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your-api-key")

_SERVER_PARAMS = StdioServerParameters(
    command="uv",
    args=["run", "sampling_server.py"],
    env={"TAVILY_API_KEY": TAVILY_API_KEY},
)


def _fmt(msg: SessionMessage | Exception) -> str:
    if isinstance(msg, Exception):
        return f"Exception: {msg}"
    try:
        data = msg.message.model_dump(by_alias=True, exclude_none=True)
        return json.dumps(data, indent=2, default=str)
    except Exception as exc:
        return f"(format error: {exc})\n{msg!r}"


def _message_title(msg: SessionMessage | Exception) -> str:
    if isinstance(msg, Exception):
        return f"exception: {type(msg).__name__}"
    try:
        root = msg.message.root
        method = getattr(root, "method", None)
        msg_id = getattr(root, "id", None)
        if method is not None and msg_id is not None:
            return f"{method}  [dim]id={msg_id}[/dim]"
        if method is not None:
            return method
        err = getattr(root, "error", None)
        if err is not None:
            err_msg = str(getattr(err, "message", ""))
            return f"error: {err_msg[:40]}  [dim]id={msg_id}[/dim]"
        return f"result  [dim]id={msg_id}[/dim]"
    except Exception:
        return "message"


def _method_of(msg: SessionMessage | Exception) -> str:
    if isinstance(msg, Exception):
        return "error"
    try:
        root = msg.message.root
        method = getattr(root, "method", None)
        if method:
            return str(method)
        if getattr(root, "error", None) is not None:
            return "error"
        return "result"
    except Exception:
        return ""


class _LoggingReadStream:
    """Proxy around read_stream; fires callback for every server→client message."""

    def __init__(
        self,
        inner: MemoryObjectReceiveStream[SessionMessage | Exception],
        callback: Any,
    ) -> None:
        self._inner = inner
        self._cb = callback

    async def receive(self) -> SessionMessage | Exception:
        msg = await self._inner.receive()
        await self._cb(msg)
        return msg

    def __aiter__(self) -> _LoggingReadStream:
        return self

    async def __anext__(self) -> SessionMessage | Exception:
        try:
            return await self.receive()
        except anyio.EndOfStream:
            raise StopAsyncIteration

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> _LoggingReadStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


class _LoggingSendStream:
    """Proxy around write_stream; fires callback for every client→server message."""

    def __init__(
        self,
        inner: MemoryObjectSendStream[SessionMessage],
        callback: Any,
    ) -> None:
        self._inner = inner
        self._cb = callback

    async def send(self, msg: SessionMessage) -> None:
        await self._cb(msg)
        await self._inner.send(msg)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> _LoggingSendStream:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


def _pick_tool(query: str, tools: list[Tool]) -> tuple[str, dict[str, Any]]:
    names = {t.name for t in tools}

    if query.startswith("/use "):
        parts = query.split(" ", 2)
        if len(parts) < 3:
            raise ValueError("Usage: /use <tool> <json-args>")
        name = parts[1]
        if name not in names:
            raise ValueError(f"Unknown tool: {name}")
        args = json.loads(parts[2])
        if not isinstance(args, dict):
            raise ValueError("Tool args must be a JSON object.")
        return name, cast(dict[str, Any], args)

    if "summarize_pdf" in names and ".pdf" in query.lower():
        return "summarize_pdf", {"pdf_path": query.strip()}

    if "generate_code" in names and query.lower().startswith("code:"):
        return "generate_code", {"concept": query.split(":", 1)[1].strip()}

    for r in ("research_agent", "research"):
        if r in names:
            return r, {"question": query}

    raise ValueError("No suitable tool found for this query.")


def _stringify_result(result: Any) -> str:
    parts: list[str] = []
    if getattr(result, "structured_content", None):
        parts.append(json.dumps(result.structured_content, indent=2, ensure_ascii=True))
    for item in result.content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
        else:
            parts.append(str(item))
    return "\n\n".join(parts) if parts else "(no result)"


class FlowDiagram(Static):
    """Animated 3-node diagram that pulses arrows as JSON-RPC messages flow.

    Layout (47 chars wide, arrows connect directly to box walls):
      ┌──────┐   ┌────────────┐        ┌────────────┐
      │ Host │──►│ MCP Client │════════►│ MCP Server │
      └──────┘   └────────────┘        └────────────┘
                               method
    """

    # Widths: host_box=8, h2c=3, client_box=14, c2s=8, server_box=14  → total 47
    _TOP = "┌──────┐   ┌────────────┐        ┌────────────┐"
    _BOT = "└──────┘   └────────────┘        └────────────┘"
    _H2C = "[dim]───[/dim]"  # 3 display chars, no arrowhead

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("", **kwargs)
        self._direction = "idle"
        self._method = ""
        self._queue: list[tuple[str, str]] = []
        self._display_timer: Any = None
        self._reset_timer: Any = None

    def on_mount(self) -> None:
        self._refresh_diagram()

    def _c2s(self) -> str:
        # No arrowhead — box-drawing chars sit at cell mid-height,
        # arrow glyphs sit at baseline; mixing them causes misalignment.
        # Direction is conveyed by colour + the label row.
        if self._direction == "outgoing":
            return "[bold cyan]════════[/bold cyan]"  # 8 display chars
        if self._direction == "incoming":
            return "[bold magenta]════════[/bold magenta]"  # 8 display chars
        return "[dim]────────[/dim]"  # 8 display chars

    def _label_line(self) -> str:
        if self._direction == "idle" or not self._method:
            return ""
        if self._direction == "outgoing":
            text = f"{self._method} ->"
            color = "cyan"
        else:
            text = f"<- {self._method}"
            color = "magenta"
        return f"[dim {color}]{text}[/dim {color}]"

    def _refresh_diagram(self) -> None:
        mid = f"│ Host │{self._H2C}│ MCP Client │{self._c2s()}│ MCP Server │"
        label = self._label_line()
        self.update(f"{self._TOP}\n{mid}\n{self._BOT}\n{label}")

    def pulse(self, direction: str, method: str = "") -> None:
        self._queue.append((direction, method))
        # Cancel pending fade-out; a new message arrived
        if self._reset_timer is not None:
            self._reset_timer.stop()
            self._reset_timer = None
        # Start draining the queue only if nothing is currently displayed
        if self._display_timer is None:
            self._advance_queue()

    def _advance_queue(self) -> None:
        if not self._queue:
            # All messages shown; hold state then fade to idle
            self._display_timer = None
            self._reset_timer = self.set_timer(2.0, self._reset)
            return
        direction, method = self._queue.pop(0)
        self._direction = direction
        self._method = method
        self._refresh_diagram()
        self._display_timer = self.set_timer(0.5, self._advance_queue)

    def _reset(self) -> None:
        self._direction = "idle"
        self._method = ""
        self._queue.clear()
        self._display_timer = None
        self._reset_timer = None
        self._refresh_diagram()


class ChatInput(TextArea):
    """Multi-line chat input: Enter submits, Shift+Enter inserts a newline."""

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            value = self.text.strip()
            self.clear()
            self.post_message(self.Submitted(value))
        else:
            await super()._on_key(event)



CSS = """
Screen { layout: vertical; }

#body { layout: horizontal; height: 1fr; }

/* Left column: chat */
#left {
    width: 40%;
    layout: vertical;
    border: solid $panel-lighten-2;
}

#chat-log {
    height: 1fr;
    padding: 0 1;
    scrollbar-size: 1 0;
}

#tools-bar {
    height: auto;
    max-height: 8;
    border-top: solid $panel-lighten-2;
    padding: 0 1;
    color: $text-muted;
}

#chat-input {
    height: 6;
    margin: 1 1 1 1;
    border: solid $primary-lighten-2;
    background: $surface;
    color: $text;
}

/* Flow diagram bar — height 6 = 1 top border + 4 content rows + 1 bottom border */
#flow-diagram {
    height: 6;
    background: $panel;
    border-top: solid $panel-lighten-2;
    border-bottom: solid $panel-lighten-2;
    text-align: center;
}

/* Right side: 2 rows × 2 cols of JSON-RPC panels */
#right          { width: 60%; layout: vertical; border: solid $panel-lighten-2; }
#client-section { layout: vertical; height: 1fr; border-bottom: solid $panel-lighten-2; }
#server-section { layout: vertical; height: 1fr; }
#right-top      { layout: horizontal; height: 1fr; }
#right-bottom   { layout: horizontal; height: 1fr; }

.group-heading {
    height: 1;
    background: $primary-darken-3;
    color: $text-muted;
    padding: 0 1;
    text-style: bold italic;
    text-align: center;
}

/* Individual panels */
.rpc-panel       { layout: vertical; width: 1fr; }
.rpc-panel-left  { layout: vertical; width: 1fr; border-right: solid $panel-lighten-2; }

.panel-title {
    height: 1;
    background: $panel-lighten-1;
    color: $text;
    padding: 0 1;
    text-style: bold;
}

.section-heading {
    height: 1;
    background: $primary-darken-2;
    color: $text;
    padding: 0 1;
    text-style: bold;
    text-align: center;
}

/* Chat log */
#chat-log { height: 1fr; padding: 0 1; }

/* JSON-RPC scroll panels */
.rpc-scroll {
    height: 1fr;
    padding: 0;
    background: $surface;
}

/* Each collapsible message row */
.rpc-scroll Collapsible {
    height: auto;
    border: none;
    padding: 0;
    margin: 0;
    background: $panel;
}

/* JSON body shown when expanded */
.json-body {
    padding: 0 3 1 3;
}
"""


class MCPSamplingTUI(App[None]):
    CSS = CSS  # type: ignore[assignment]

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_logs", "Clear logs"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._query_queue: asyncio.Queue[str] = asyncio.Queue()
        self._approval_queue: asyncio.Queue[bool] = asyncio.Queue()
        self._awaiting_approval: bool = False
        self._tools: list[Tool] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            # Left: agent chat
            with Vertical(id="left"):
                yield Label("Host Application", classes="section-heading")
                yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True)
                yield Static("Connecting to MCP server…", id="tools-bar")
                yield ChatInput("", id="chat-input", soft_wrap=True)
            # Right: flow diagram + 2×2 JSON-RPC inspector
            with Vertical(id="right"):
                yield Label("MCP JSON-RPC Inspector", classes="section-heading")
                yield FlowDiagram(id="flow-diagram")
                with Vertical(id="client-section"):
                    yield Label("MCP Client", classes="group-heading")
                    with Horizontal(id="right-top"):
                        with Vertical(classes="rpc-panel-left"):
                            yield Label(" ↑ Client Sends", classes="panel-title")
                            yield VerticalScroll(id="client-send-log", classes="rpc-scroll")
                        with Vertical(classes="rpc-panel"):
                            yield Label(" ↓ Client Receives", classes="panel-title")
                            yield VerticalScroll(id="client-recv-log", classes="rpc-scroll")
                with Vertical(id="server-section"):
                    yield Label("MCP Server", classes="group-heading")
                    with Horizontal(id="right-bottom"):
                        with Vertical(classes="rpc-panel-left"):
                            yield Label(" ↓ Server Receives", classes="panel-title")
                            yield VerticalScroll(id="server-recv-log", classes="rpc-scroll")
                        with Vertical(classes="rpc-panel"):
                            yield Label(" ↑ Server Sends", classes="panel-title")
                            yield VerticalScroll(id="server-send-log", classes="rpc-scroll")
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self._mcp_loop(), exclusive=True, name="mcp-session")

    @on(ChatInput.Submitted)
    async def _on_query(self, event: ChatInput.Submitted) -> None:
        value = event.value
        if self._awaiting_approval:
            denied = value.lower() in ("n", "no", "deny")
            await self._approval_queue.put(not denied)
        elif value:
            await self._query_queue.put(value)

    def action_clear_logs(self) -> None:
        for log_id in (
            "#client-send-log",
            "#client-recv-log",
            "#server-recv-log",
            "#server-send-log",
        ):
            self.query_one(log_id, VerticalScroll).query(Collapsible).remove()

    def _chat(self, text: str) -> None:
        self.query_one("#chat-log", RichLog).write(text.rstrip("\n"))

    def _chat_md(self, text: str) -> None:
        self.query_one("#chat-log", RichLog).write(Markdown(text))

    async def _append_rpc(
        self, panel_id: str, msg: SessionMessage | Exception, color: str
    ) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        title = f"[dim]{ts}[/dim]  [{color}]{_message_title(msg)}[/{color}]"
        body = Syntax(_fmt(msg), "json", theme="monokai", word_wrap=True)
        collapsible = Collapsible(
            Static(body, classes="json-body"),
            title=title,
            collapsed=True,
        )
        container = self.query_one(panel_id, VerticalScroll)
        await container.mount(collapsible)
        container.scroll_end(animate=False)

    def _pulse_flow(self, direction: str, method: str = "") -> None:
        try:
            self.query_one("#flow-diagram", FlowDiagram).pulse(direction, method)
        except Exception:
            pass

    def _update_tools_bar(self, tools: list[Tool]) -> None:
        names = "  ".join(f"[cyan]{t.name}[/cyan]" for t in tools)
        self.query_one("#tools-bar", Static).update(f"[dim]Tools:[/dim] {names}")

    async def _request_approval(self, messages: list, params: Any) -> bool:
        self._chat("[bold yellow]Sampling Request[/bold yellow]")
        self._chat(f"  [dim]Messages:[/dim] {len(messages)}")
        if params.systemPrompt:
            self._chat(f"  [dim]System:[/dim] {escape(params.systemPrompt[:120])}")
        if getattr(params, "tools", None):
            names = ", ".join(t.name for t in params.tools)
            self._chat(f"  [dim]Tools:[/dim] {escape(names)}")
        self._chat("[dim]Type y + Enter to approve, n + Enter to deny[/dim]")

        self._awaiting_approval = True
        self.query_one("#chat-input", ChatInput).focus()
        try:
            return await self._approval_queue.get()
        finally:
            self._awaiting_approval = False

    async def _mcp_loop(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            self._chat(
                "[red]OPENAI_API_KEY is not set. Cannot handle sampling requests.[/red]"
            )
            return

        _raw_handler = OpenAISamplingHandler(default_model=cast(Any, OPENAI_MODEL))

        async def _sampling_with_log(messages: list, params: Any, context: Any) -> Any:
            approved = await self._request_approval(messages, params)
            if not approved:
                self._chat("[red]Sampling denied.[/red]")
                raise Exception("Sampling request denied by user")
            self._chat(
                f"  [dim yellow]LLM call — {len(messages)} message(s)[/dim yellow]"
            )
            result = await _raw_handler(messages, params, context)
            content = getattr(result, "content", None)
            if isinstance(content, list):
                for item in content:
                    item_name = getattr(item, "name", None)
                    item_text = getattr(item, "text", None)
                    if item_name is not None:
                        args_preview = escape(
                            json.dumps(getattr(item, "input", {}), default=str)[:80]
                        )
                        self._chat(
                            f"  [dim]↳ tool_use: [cyan]{item_name}[/cyan]({args_preview})[/dim]"
                        )
                    elif isinstance(item_text, str) and item_text:
                        self._chat(
                            f"  [dim]↳ text: {escape(item_text[:100])}{'…' if len(item_text) > 100 else ''}[/dim]"
                        )
            elif content is not None:
                text = getattr(content, "text", None)
                if isinstance(text, str) and text:
                    preview = text[:100]
                    self._chat(
                        f"  [dim]↳ {escape(preview)}{'…' if len(text) > 100 else ''}[/dim]"
                    )
            return result

        sampling_cb = create_sampling_callback(_sampling_with_log)

        # C→S: client-send panel + server-recv panel
        async def on_outgoing(msg: SessionMessage) -> None:
            self._pulse_flow("outgoing", _method_of(msg))
            await self._append_rpc("#client-send-log", msg, "cyan")
            await self._append_rpc("#server-recv-log", msg, "green")

        # S→C: client-recv panel + server-send panel
        async def on_incoming(msg: SessionMessage | Exception) -> None:
            self._pulse_flow("incoming", _method_of(msg))
            await self._append_rpc("#client-recv-log", msg, "magenta")
            await self._append_rpc("#server-send-log", msg, "yellow")

        try:
            async with stdio_client(_SERVER_PARAMS) as (raw_read, raw_write):
                logged_read = _LoggingReadStream(raw_read, on_incoming)
                logged_write = _LoggingSendStream(raw_write, on_outgoing)

                async with ClientSession(
                    logged_read,  # type: ignore[arg-type]
                    logged_write,  # type: ignore[arg-type]
                    sampling_callback=sampling_cb,
                    sampling_capabilities=SamplingCapability(
                        tools=SamplingToolsCapability()
                    ),
                ) as session:
                    await session.initialize()

                    result = await session.list_tools()
                    self._tools = result.tools
                    self._update_tools_bar(self._tools)
                    self._chat(
                        f"[green]Connected.[/green] Loaded [bold]{len(self._tools)}[/bold] tools.\n"
                        "Type a question, [cyan]code: <concept>[/cyan], a PDF path, "
                        "or [cyan]/use <tool> <json-args>[/cyan].\n"
                    )

                    while True:
                        query = await self._query_queue.get()
                        await self._run_query(session, query)

        except Exception as exc:
            self._chat(f"[red]MCP session error: {exc}[/red]")

    def _divider(self) -> None:
        self._chat("[dim]" + "─" * 42 + "[/dim]")

    def _section(self, label: str, color: str) -> None:
        inner = f"  {label}  "
        pad = max(0, 42 - len(inner))
        left = pad // 2
        right = pad - left
        self._chat(f"[{color}]{'─' * left}{inner}{'─' * right}[/{color}]")

    async def _run_query(self, session: ClientSession, query: str) -> None:
        self._chat("")
        self._divider()
        self._chat("")
        self._chat(f"[bold white]You:[/bold white]  {escape(query)}")
        self._chat("")
        try:
            tool_name, args = _pick_tool(query, self._tools)
            self._section(f"tool  {tool_name}", "dim cyan")
            self._chat("")
            self._chat(f"[dim]{escape(json.dumps(args, indent=2, default=str))}[/dim]")
            self._chat("")

            t0 = time.perf_counter()
            result = await session.call_tool(tool_name, args)
            elapsed = time.perf_counter() - t0

            self._section(f"Agent  {elapsed:.1f}s", "bold green")
            self._chat("")
            self._chat_md(_stringify_result(result))
            self._chat("")

        except Exception as exc:
            self._section("error", "bold red")
            self._chat("")
            self._chat(f"[red]{escape(str(exc))}[/red]")
            self._chat("")
        finally:
            self._divider()


async def run_tui() -> None:
    app = MCPSamplingTUI()
    await app.run_async()
