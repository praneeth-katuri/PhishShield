from backend.utils import config


def test_config_loaded():
    assert config.FLASK_SECRET_KEY is not None
    assert config.RECAPTCHA_SECRET_KEY is not None
    assert config.RECAPTCHA_SITE_KEY is not None
