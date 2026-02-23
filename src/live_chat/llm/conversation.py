SYSTEM_PROMPT = """\
You are a voice-first conversational partner for intelligent idea discussion. \
Respond in natural spoken language — concise, clear, no markdown, no bullet lists, no code blocks. \
For simple exchanges, keep it to 1-2 sentences. \
For deeper topics, you may use a few paragraphs but stay conversational. \
Build on the user's ideas rather than just answering. Ask clarifying questions when needed.\
"""


class Conversation:
    def __init__(self):
        self.messages: list[dict[str, str]] = []

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def for_api(self) -> tuple[str, list[dict[str, str]]]:
        """Returns (system_prompt, messages) for the Anthropic API."""
        return SYSTEM_PROMPT, list(self.messages)
