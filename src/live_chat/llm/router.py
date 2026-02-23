from anthropic import AsyncAnthropic

from live_chat.config import Config

_ROUTER_PROMPT = """\
Classify the user's message as FAST or DEEP.

FAST: casual conversation, greetings, acknowledgments, simple factual questions, short clarifications, follow-ups that don't need analysis.
DEEP: multi-step reasoning, analysis, comparisons, creative brainstorming, nuanced topics, complex questions, anything requiring sustained thought.

Consider the conversation history — a simple follow-up to a deep topic should stay DEEP.

Respond with exactly one word: FAST or DEEP\
"""


class Router:
    def __init__(self, config: Config):
        self._client = AsyncAnthropic()
        self._config = config

    async def classify(
        self, text: str, history: list[dict[str, str]]
    ) -> str:
        """Classify a message as 'fast' or 'deep'."""
        context = history[-6:] if history else []  # last 3 exchanges
        messages = [*context, {"role": "user", "content": text}]

        response = await self._client.messages.create(
            model=self._config.fast_model,
            max_tokens=4,
            system=_ROUTER_PROMPT,
            messages=messages,
        )

        result = response.content[0].text.strip().upper()
        return "deep" if "DEEP" in result else "fast"

    async def route(
        self, text: str, history: list[dict[str, str]]
    ) -> str:
        """Returns the model ID to use for this message."""
        classification = await self.classify(text, history)
        if classification == "deep":
            return self._config.deep_model
        return self._config.fast_model
