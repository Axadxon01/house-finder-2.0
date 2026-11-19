"""Password hashing (salted PBKDF2-SHA256) and input validation."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

ITERATIONS = 200_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex),
                                     int(iterations))
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def username_ok(username: str) -> bool:
    return bool(USERNAME_RE.match(username or ""))


def password_ok(password: str) -> bool:
    return len(password or "") >= 6


def email_ok(email: str) -> bool:
    return not email or bool(EMAIL_RE.match(email))
