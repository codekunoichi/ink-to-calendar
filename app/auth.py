"""
HTTP Basic Auth for ink-to-calendar (Step 9).

Single-user authentication using bcrypt password hashing.
The password hash is stored in .env as APP_PASSWORD_HASH.

To generate a hash for your password:
    venv/bin/python -c "from app.auth import hash_password; print(hash_password('yourpassword'))"
"""

import secrets

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.config import get_settings

security = HTTPBasic()


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of plain. Store the result in APP_PASSWORD_HASH."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Constant-time bcrypt check."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def require_auth(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    FastAPI dependency — validates HTTP Basic credentials against .env settings.
    Returns the authenticated username.
    Raises 401 if credentials are wrong.
    """
    settings = get_settings()

    username_ok = secrets.compare_digest(
        credentials.username.encode(), settings.app_username.encode()
    )
    password_ok = verify_password(credentials.password, settings.app_password_hash)

    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
