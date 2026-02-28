import asyncio
import os
from pathlib import Path

from rich.console import Console

from anthropic import AsyncAnthropic

from live_chat.config import Config
from live_chat.pipeline import Pipeline, State


def _load_dotenv():
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                os.environ.setdefault(k.strip(), v)

console = Console()

_STATE_DISPLAY = {
    State.IDLE: "[dim]Idle[/dim]",
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
    console.print(f"TTS voice:  [cyan]{config.tts_voice}[/cyan]")
    console.print("[bold]Ctrl+C[/bold] to quit.\n")

    # Verify API key before loading heavy models
    console.print("[dim]Checking API key...[/dim]")
    try:
        client = AsyncAnthropic()
        await client.messages.create(
            model=config.fast_model,
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
    except Exception as e:
        console.print(f"[bold red]API key check failed:[/bold red] {e}")
        return

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
    pipeline.activate()

    # Wait until interrupted
    try:
        await audio_task
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        console.print("\n[dim]Goodbye.[/dim]")


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
