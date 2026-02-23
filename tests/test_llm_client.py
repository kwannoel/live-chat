import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from live_chat.llm.client import LLMClient
from live_chat.config import Config


@pytest.mark.asyncio
async def test_llm_client_stream():
    config = Config()

    with patch("live_chat.llm.client.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Mock the streaming context manager
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def fake_text_stream():
            for word in ["Hello", " there", "!"]:
                yield word

        mock_stream.text_stream = fake_text_stream()
        mock_stream.get_final_message = AsyncMock(return_value=MagicMock(
            content=[MagicMock(text="Hello there!")]
        ))

        mock_client.messages.stream.return_value = mock_stream

        client = LLMClient(config)
        chunks = []
        async for chunk in client.stream("Tell me something", "Hello", []):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello there!"
