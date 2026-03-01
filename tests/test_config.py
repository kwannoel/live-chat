from live_chat.config import Config


def test_default_config():
    config = Config()
    assert config.fast_model == "claude-haiku-4-5-20251001"
    assert config.deep_model == "claude-sonnet-4-6"
    assert config.sample_rate == 16000
    assert config.tts_voice == "en_US-lessac-medium"


def test_config_from_dict():
    config = Config.from_dict({"deep_model": "claude-opus-4-6"})
    assert config.deep_model == "claude-opus-4-6"
    # defaults preserved
    assert config.fast_model == "claude-haiku-4-5-20251001"


def test_default_config_new_fields():
    config = Config()
    assert config.persona is None
    assert config.auto_speak is False


def test_config_from_dict_persona():
    config = Config.from_dict({"persona": "You are Alice."})
    assert config.persona == "You are Alice."
    assert config.auto_speak is False


def test_config_from_dict_auto_speak():
    config = Config.from_dict({"auto_speak": True})
    assert config.auto_speak is True
