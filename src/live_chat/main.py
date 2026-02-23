import asyncio
import signal
import sys

from rich.console import Console
from rich.live import Live
from rich.text import Text

from live_chat.config import Config
from live_chat.pipeline import Pipeline, State

console = Console()

_STATE_DISPLAY = {
    State.WAITING_FOR_WAKE_WORD: ("[dim]Waiting for wake word...[/dim]", "dots"),
    State.LISTENING: ("[bold green]Listening...[/bold green]", "dots"),
    State.THINKING: ("[bold yellow]Thinking...[/bold yellow]", "dots"),
    State.SPEAKING: ("[bold blue]Speaking...[/bold blue]", "dots"),
}


async def run():
    config = Config.load()
    pipeline = Pipeline(config)

    console.print(f"[bold]Live Chat[/bold] — voice-first agent")
    console.print(f"Wake word: [cyan]{config.wake_word}[/cyan]")
    console.print(f"Fast model: [cyan]{config.fast_model}[/cyan]")
    console.print(f"Deep model: [cyan]{config.deep_model}[/cyan]")
    console.print(f"Press [bold]Ctrl+C[/bold] to quit.\n")

    def on_state_change(state: State):
        label, _ = _STATE_DISPLAY[state]
        console.print(f"  {label}")

    pipeline.on_state_change(on_state_change)

    try:
        await pipeline.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/dim]")


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
