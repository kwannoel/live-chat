from live_chat.llm.conversation import Conversation


def test_conversation_starts_empty():
    conv = Conversation()
    assert conv.messages == []


def test_conversation_add_user_message():
    conv = Conversation()
    conv.add_user("Hello")
    assert conv.messages == [{"role": "user", "content": "Hello"}]


def test_conversation_add_assistant_message():
    conv = Conversation()
    conv.add_user("Hi")
    conv.add_assistant("Hello!")
    assert len(conv.messages) == 2
    assert conv.messages[1] == {"role": "assistant", "content": "Hello!"}


def test_conversation_for_api_includes_system():
    conv = Conversation()
    conv.add_user("Hi")
    system, messages = conv.for_api()
    assert "spoken" in system.lower() or "concise" in system.lower()
    assert messages == [{"role": "user", "content": "Hi"}]
