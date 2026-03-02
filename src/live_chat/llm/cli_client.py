import asyncio
import json
import os
from collections.abc import AsyncIterator

from live_chat.config import Config


class CLIClient:
    def __init__(self, config: Config):
        self._cli_path = config.cli_path
        self._env = _clean_env()
        self._warm_proc: asyncio.subprocess.Process | None = None
        self._warm_system: str | None = None

    async def warm_up(self, system: str):
        """Pre-spawn a CLI process so the next stream() call starts faster."""
        self._warm_proc = await self._spawn(system, stdin_pipe=True)
        self._warm_system = system

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Stream text chunks from the Claude CLI."""
        prompt = _format_prompt(messages)

        # Use pre-spawned process if available and system prompt matches
        proc = None
        if (
            self._warm_proc is not None
            and self._warm_system == system
            and self._warm_proc.returncode is None
        ):
            proc = self._warm_proc
            self._warm_proc = None
            self._warm_system = None
            proc.stdin.write(prompt.encode())
            proc.stdin.write_eof()

        if proc is None:
            proc = await self._spawn(system, stdin_pipe=False, prompt=prompt)

        first_token = True
        try:
            async for line in proc.stdout:
                text = _parse_ndjson_line(line)
                if text is None:
                    continue
                if first_token:
                    text = text.lstrip("\n")
                    first_token = False
                    if not text:
                        continue
                yield text
        finally:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            # Pre-spawn next process for the next turn
            await self.warm_up(system)

    async def _spawn(
        self,
        system: str,
        *,
        stdin_pipe: bool = False,
        prompt: str | None = None,
    ) -> asyncio.subprocess.Process:
        """Spawn a claude CLI subprocess."""
        args = [
            self._cli_path,
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--tools", "",
            "--no-session-persistence",
            "--system-prompt", system,
        ]
        if prompt is not None:
            args.extend(["-p", prompt])

        return await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if stdin_pipe else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=self._env,
        )


def _format_prompt(messages: list[dict[str, str]]) -> str:
    """Convert conversation messages into a single prompt string for the CLI."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"Human: {content}")
        else:
            parts.append(f"Assistant: {content}")
    return "\n\n".join(parts)


def _clean_env() -> dict[str, str]:
    """Return a copy of the environment without CLAUDECODE to avoid nesting errors."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    return env


def _parse_ndjson_line(raw: bytes) -> str | None:
    """Extract text from a stream-json NDJSON line, or None if not a text delta."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None

    if obj.get("type") != "stream_event":
        return None

    event = obj.get("event", {})
    if event.get("type") != "content_block_delta":
        return None

    delta = event.get("delta", {})
    if delta.get("type") != "text_delta":
        return None

    return delta.get("text")
