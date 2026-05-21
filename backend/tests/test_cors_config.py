import os
import importlib


def test_cors_origins_parses_comma_separated_string(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://example.com, https://app.example.com")
    from app.core import config
    importlib.reload(config)
    assert config.settings.cors_origins == ["https://example.com", "https://app.example.com"]


def test_env_defaults_to_development(monkeypatch):
    monkeypatch.delenv("ENV", raising=False)
    from app.core import config
    importlib.reload(config)
    assert config.settings.env == "development"


def test_env_reads_from_environment(monkeypatch):
    monkeypatch.setenv("ENV", "production")
    from app.core import config
    importlib.reload(config)
    assert config.settings.env == "production"
