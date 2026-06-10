from pathlib import Path
import ast
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

from fastmcp import FastMCP, Context
from pypdf import PdfReader
from tavily import TavilyClient  # type: ignore[import-untyped]

mcp = FastMCP()


def _eval_arithmetic(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_arithmetic(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_arithmetic(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _eval_arithmetic(node.left)
        right = _eval_arithmetic(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported binary operator")

    raise ValueError("Only numeric arithmetic expressions are allowed")


@mcp.tool
async def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '(2 + 3) * 4 / 5'."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _eval_arithmetic(parsed)
        if result.is_integer():
            return str(int(result))
        return str(result)
    except Exception as exc:
        return f"Invalid expression: {exc}"


@mcp.tool
async def research_agent(question: str, ctx: Context) -> str:
    """Research a question by letting the model call tools (Tavily and others)."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY is not set. Please configure it and try again."

    def search_web(
        query: str,
        max_results: int = 5,
        year: int | None = None,
        time_range: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        """Search the web via Tavily and return concise source snippets.

        Args:
            query: Search query string.
            max_results: Number of results to return (default 5).
            year: Filter results to a specific year, e.g. 2024. Sets start_date/end_date automatically.
            time_range: Broad time filter — one of: day, week, month, year.
            start_date: Earliest date for results in YYYY-MM-DD format.
            end_date: Latest date for results in YYYY-MM-DD format.
        """
        client = TavilyClient(api_key=api_key)
        search_fn = cast(
            Callable[..., dict[str, Any]],
            getattr(client, "search"),
        )
        # Convert year shorthand to explicit date range
        if year and not start_date and not end_date:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

        kwargs: dict[str, Any] = dict(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
        )
        if time_range:
            kwargs["time_range"] = time_range
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date

        response: dict[str, Any] = search_fn(**kwargs)

        raw_results_obj = response.get("results", [])
        if not isinstance(raw_results_obj, list) or not raw_results_obj:
            return "No results found."

        raw_results = cast(list[dict[str, Any]], raw_results_obj)
        snippets: list[str] = []
        for i, item in enumerate(raw_results[:max_results], start=1):
            title = str(item.get("title", "Untitled"))
            url = str(item.get("url", ""))
            content = str(item.get("content", ""))
            snippets.append(f"{i}. {title}\nURL: {url}\n{content[:800]}")

        if not snippets:
            return "No results found."

        return "\n\n".join(snippets)

    def get_time() -> str:
        """Return the current local date and time."""
        return datetime.now().strftime("%A, %d %B %Y %H:%M:%S")

    tools: list[Callable[..., Any]] = [search_web, get_time]
    res = await ctx.sample(
        messages=question,
        system_prompt=(
            "You are a research assistant. Use tools when needed, especially web search. "
            "Provide a concise answer with source URLs and note uncertainty when applicable."
        ),
        tools=tools,
        temperature=0.2,
    )
    return res.text or ""


async def _summarize_pdf_impl(pdf_path: str, ctx: Context) -> str:
    path = Path(pdf_path)
    if not path.exists() or not path.is_file():
        return f"PDF file not found: {pdf_path}"

    if path.suffix.lower() != ".pdf":
        return f"Not a PDF file: {pdf_path}"

    reader = PdfReader(str(path))
    extracted: list[str] = []
    for page in reader.pages:
        extracted.append(page.extract_text() or "")

    content = "\n".join(extracted).strip()
    if not content:
        return "The PDF appears to contain no extractable text."

    content = content[:20000]
    res = await ctx.sample(
        messages=f"Please summarize this PDF content:\n\n{content}", temperature=0.5
    )
    return res.text or ""


@mcp.tool
async def summarize_pdf(pdf_path: str, ctx: Context) -> str:
    """Generate a summary of a PDF file at the given path."""
    return await _summarize_pdf_impl(pdf_path, ctx)


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False, log_level="WARNING")
