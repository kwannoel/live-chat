from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from live_chat.config import Config


class LLMClient:
    def __init__(self, config: Config):
        self._client = AsyncAnthropic()
        self._config = config

    async def stream(
        self,
        model: str,
        system: str,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Stream text chunks from the LLM."""
        async with self._client.messages.stream(
            model=model,
            max_tokens=2048,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
