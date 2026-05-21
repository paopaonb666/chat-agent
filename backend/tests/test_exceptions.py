from unittest.mock import MagicMock, patch
import pytest
from fastapi import Request
from app.main import global_exception_handler


@pytest.mark.asyncio
async def test_global_exception_handler_returns_generic_in_production():
    mock_request = MagicMock(spec=Request)
    mock_request.method = "GET"
    mock_request.url.path = "/test"
    mock_request.client = MagicMock(host="127.0.0.1")

    with patch("app.main.settings") as mock_settings:
        mock_settings.env = "production"
        response = await global_exception_handler(mock_request, ValueError("boom"))
        assert response.status_code == 500
        assert response.body == b'{"detail":"Internal server error"}'


@pytest.mark.asyncio
async def test_global_exception_handler_returns_detail_in_development():
    mock_request = MagicMock(spec=Request)
    mock_request.method = "GET"
    mock_request.url.path = "/test"
    mock_request.client = MagicMock(host="127.0.0.1")

    with patch("app.main.settings") as mock_settings:
        mock_settings.env = "development"
        response = await global_exception_handler(mock_request, ValueError("boom"))
        assert response.status_code == 500
        body = response.body.decode()
        assert "ValueError" in body
        assert "boom" in body
