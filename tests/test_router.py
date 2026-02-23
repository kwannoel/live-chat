import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from live_chat.llm.router import Router
from live_chat.config import Config


@pytest.mark.asyncio
async def test_router_classifies_fast():
    config = Config()

    with patch("live_chat.llm.router.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="FAST")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        router = Router(config)
        result = await router.classify("Hey, how's it going?", [])

        assert result == "fast"


@pytest.mark.asyncio
async def test_router_classifies_deep():
    config = Config()

    with patch("live_chat.llm.router.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="DEEP")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        router = Router(config)
        result = await router.classify(
            "Compare the trade-offs between event sourcing and CQRS", []
        )

        assert result == "deep"


@pytest.mark.asyncio
async def test_router_returns_model_name():
    config = Config()

    with patch("live_chat.llm.router.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="FAST")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        router = Router(config)
        model = await router.route("Hi there", [])

        assert model == config.fast_model
