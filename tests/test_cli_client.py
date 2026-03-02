import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from live_chat.llm.cli_client import CLIClient, _format_prompt, _parse_ndjson_line
from live_chat.config import Config


def test_format_prompt_single_user():
    messages = [{"role": "user", "content": "Hello"}]
    assert _format_prompt(messages) == "Human: Hello"


def test_format_prompt_conversation():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    assert _format_prompt(messages) == (
        "Human: Hi\n\nAssistant: Hello!\n\nHuman: How are you?"
    )


def test_format_prompt_empty():
    assert _format_prompt([]) == ""


def test_parse_ndjson_text_delta():
    line = json.dumps({
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        },
    }).encode()
    assert _parse_ndjson_line(line) == "Hello"


def test_parse_ndjson_non_text_delta():
    line = json.dumps({
        "type": "stream_event",
        "event": {
            "type": "content_block_start",
            "content_block": {"type": "text"},
        },
    }).encode()
    assert _parse_ndjson_line(line) is None


def test_parse_ndjson_non_stream_event():
    line = json.dumps({"type": "system", "message": "starting"}).encode()
    assert _parse_ndjson_line(line) is None


def test_parse_ndjson_invalid_json():
    assert _parse_ndjson_line(b"not json at all") is None


def test_parse_ndjson_empty():
    assert _parse_ndjson_line(b"") is None


@pytest.mark.asyncio
async def test_cli_client_stream():
    """CLIClient.stream should yield text from NDJSON, stripping leading newlines."""
    config = Config(cli_path="echo")

    ndjson_lines = [
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "\n\nHello"},
            },
        }),
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": " world"},
            },
        }),
    ]
    script = "\n".join(ndjson_lines)

    import asyncio
    import os

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)

    # Use python to print the NDJSON lines to stdout
    proc = await asyncio.create_subprocess_exec(
        "python3", "-c", f"print({script!r})",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )

    from live_chat.llm.cli_client import _parse_ndjson_line

    tokens = []
    first_token = True
    async for line in proc.stdout:
        text = _parse_ndjson_line(line)
        if text is None:
            continue
        if first_token:
            text = text.lstrip("\n")
            first_token = False
            if not text:
                continue
        tokens.append(text)

    proc.kill()
    await proc.wait()

    assert tokens == ["Hello", " world"]


@pytest.mark.asyncio
async def test_cli_client_warm_up():
    """warm_up() should pre-spawn a process with stdin pipe."""
    config = Config(backend="cli", cli_path="claude")
    client = CLIClient(config)

    mock_proc = MagicMock()
    mock_proc.returncode = None

    with patch.object(client, "_spawn", new_callable=AsyncMock, return_value=mock_proc):
        await client.warm_up("You are helpful.")
        assert client._warm_proc is mock_proc
        assert client._warm_system == "You are helpful."
        client._spawn.assert_called_once_with("You are helpful.", stdin_pipe=True)
