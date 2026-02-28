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
