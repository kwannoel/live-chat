import asyncio
import os
import sys
from pathlib import Path

from rich.console import Console

from live_chat.config import Config
from live_chat.pipeline import Pipeline, State


def _load_dotenv():
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

console = Console()

_STATE_DISPLAY = {
    State.IDLE: "[dim]Press Enter to start...[/dim]",
    State.LISTENING: "[bold green]Listening...[/bold green]",
    State.THINKING: "[bold yellow]Thinking...[/bold yellow]",
    State.SPEAKING: "[bold blue]Speaking...[/bold blue]",
}


async def run():
    _load_dotenv()
    config = Config.load()

    console.print("[bold]Live Chat[/bold] — voice-first agent")
    console.print(f"Fast model: [cyan]{config.fast_model}[/cyan]")
    console.print(f"Deep model: [cyan]{config.deep_model}[/cyan]")
    console.print("Press [bold]Enter[/bold] to start. [bold]Ctrl+C[/bold] to quit.\n")

    console.print("[dim]Loading models...[/dim]")
    pipeline = Pipeline(config)
    console.print("[green]Ready![/green]\n")

    def on_state_change(state: State):
        console.print(f"  {_STATE_DISPLAY[state]}")

    def on_transcript(role: str, text: str, model: str | None):
        if role == "user":
            console.print(f"\n  [bold cyan]You:[/bold cyan] {text}")
        else:
            model_short = model.split("-")[1] if model and "-" in model else "?"
            console.print(f"  [bold magenta]Agent ({model_short}):[/bold magenta] {text}\n")

    pipeline.on_state_change(on_state_change)
    pipeline.on_transcript(on_transcript)

    # Run pipeline audio loop in background
    audio_task = asyncio.create_task(pipeline.run())

    # Keyboard input loop
    loop = asyncio.get_event_loop()
    try:
        while True:
            # Wait for Enter key (non-blocking via executor)
            await loop.run_in_executor(None, sys.stdin.readline)
            pipeline.activate()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Goodbye.[/dim]")
    finally:
        audio_task.cancel()
        try:
            await audio_task
        except asyncio.CancelledError:
            pass


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
