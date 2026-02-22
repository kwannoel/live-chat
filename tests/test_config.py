from live_chat.config import Config


def test_default_config():
    config = Config()
    assert config.wake_word == "hey_jarvis"
    assert config.vad_silence_ms == 700
    assert config.active_timeout_s == 30
    assert config.fast_model == "claude-haiku-4-5-20241022"
    assert config.deep_model == "claude-sonnet-4-5-20250929"
    assert config.tts_voice == "af_heart"
    assert config.sample_rate == 16000
    assert config.tts_sample_rate == 24000


def test_config_from_dict():
    config = Config.from_dict({"wake_word": "alexa", "deep_model": "claude-opus-4-6"})
    assert config.wake_word == "alexa"
    assert config.deep_model == "claude-opus-4-6"
    # defaults preserved
    assert config.fast_model == "claude-haiku-4-5-20241022"
