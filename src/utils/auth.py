from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from rich.console import Console


@dataclass(frozen=True)
class AuthSettings:
    username: str
    password_sha256: str
    demo_mode: bool

logger = Console()

def password_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_auth_settings(
    environ: Mapping[str, str], secrets: Mapping[str, Any] | None
) -> AuthSettings:
    """Resolve credentials from environment, Streamlit secrets, or local demo defaults."""
    env_user = environ.get("AIJOB_APP_USER")
    env_hash = environ.get("AIJOB_APP_PASSWORD_SHA256")
    logger.log("env: user=%s, hash=%s", env_user, env_hash)
    if env_user and env_hash:
        return AuthSettings(env_user, env_hash.lower(), False)
    
    try:
        auth = secrets.get("auth", {}) if secrets is not None else {}
        username = auth.get("username")
        password_sha256 = auth.get("password_sha256")
    except (AttributeError, KeyError, TypeError):
        username = None
        password_sha256 = None

    if username and password_sha256:
        logger.log("secrets: user=%s, hash=%s", username, password_sha256)
        return AuthSettings(str(username), str(password_sha256).lower(), False)

    # logger.log("secrets: user=%s, hash=%s", username, password_sha256)
    return AuthSettings("admin", password_hash("AIJob2026!"), True)
