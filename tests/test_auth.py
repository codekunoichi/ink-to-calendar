"""
Tests for HTTP Basic Auth middleware (Step 9).
Run with: venv/bin/python -m pytest tests/test_auth.py -v
"""

import base64
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.auth import hash_password, verify_password


# ---------------------------------------------------------------------------
# hash_password / verify_password
# ---------------------------------------------------------------------------

class TestPasswordUtils:
    def test_hash_is_not_plaintext(self):
        h = hash_password("mysecret")
        assert h != "mysecret"

    def test_verify_correct_password(self):
        h = hash_password("mysecret")
        assert verify_password("mysecret", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("mysecret")
        assert verify_password("wrong", h) is False

    def test_different_hashes_for_same_password(self):
        # bcrypt uses random salt
        h1 = hash_password("mysecret")
        h2 = hash_password("mysecret")
        assert h1 != h2

    def test_verify_still_works_with_different_hashes(self):
        h1 = hash_password("mysecret")
        h2 = hash_password("mysecret")
        assert verify_password("mysecret", h1) is True
        assert verify_password("mysecret", h2) is True


# ---------------------------------------------------------------------------
# Auth middleware on routes
# ---------------------------------------------------------------------------

CORRECT_PASSWORD = "testpass123"
CORRECT_HASH = None  # set in fixture


@pytest.fixture()
def auth_client(tmp_path):
    from app.auth import hash_password
    pw_hash = hash_password(CORRECT_PASSWORD)

    import app.main as main_module
    with patch.object(main_module, "DB_PATH", tmp_path / "test.db"), \
         patch.object(main_module, "UPLOAD_DIR", tmp_path / "uploads"), \
         patch("app.auth.get_settings") as mock_settings:
        mock_settings.return_value.app_username = "rumpa"
        mock_settings.return_value.app_password_hash = pw_hash

        from app.database import init_db
        init_db(tmp_path / "test.db")
        (tmp_path / "uploads").mkdir()

        from app.main import app
        yield TestClient(app, raise_server_exceptions=False)


def _basic_header(username: str, password: str) -> dict:
    creds = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}


class TestAuthMiddleware:
    def test_unauthenticated_returns_401(self, auth_client):
        resp = auth_client.get("/health")
        assert resp.status_code == 401

    def test_wrong_password_returns_401(self, auth_client):
        resp = auth_client.get("/health", headers=_basic_header("rumpa", "wrongpass"))
        assert resp.status_code == 401

    def test_wrong_username_returns_401(self, auth_client):
        resp = auth_client.get("/health", headers=_basic_header("admin", CORRECT_PASSWORD))
        assert resp.status_code == 401

    def test_correct_credentials_returns_200(self, auth_client):
        resp = auth_client.get("/health", headers=_basic_header("rumpa", CORRECT_PASSWORD))
        assert resp.status_code == 200

    def test_auth_required_for_upload(self, auth_client):
        resp = auth_client.post("/upload", files={"file": ("w.png", b"img", "image/png")})
        assert resp.status_code == 401

    def test_auth_required_for_shopping(self, auth_client):
        resp = auth_client.get("/shopping")
        assert resp.status_code == 401

    def test_auth_required_for_insights(self, auth_client):
        resp = auth_client.get("/insights")
        assert resp.status_code == 401

    def test_www_authenticate_header_present(self, auth_client):
        resp = auth_client.get("/health")
        assert "www-authenticate" in resp.headers

    def test_static_files_not_blocked(self, auth_client):
        # Static files served without auth — browser needs CSS/JS before login page loads
        resp = auth_client.get("/static/style.css")
        # Either 200 (file exists) or 404 (not mounted in test) — not 401
        assert resp.status_code != 401
