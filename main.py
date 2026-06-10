import asyncio

from agent_tui_client import run_tui


def main() -> None:
    asyncio.run(run_tui())


if __name__ == "__main__":
    main()
