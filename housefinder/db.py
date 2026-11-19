"""SQLite storage for users, listings, favourites and interest signals."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import auth, config
from .features import INPUT_KEYS

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
    email TEXT,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS listings (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    is_demo INTEGER NOT NULL DEFAULT 0,
    source_pid TEXT,
    title TEXT,
    description TEXT,
    price REAL NOT NULL,
    features TEXT NOT NULL,
    lat REAL,
    lon REAL,
    contact TEXT,
    image BLOB,
    image_mime TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT
);
CREATE TABLE IF NOT EXISTS favorites (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    listing_id INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, listing_id)
);
CREATE TABLE IF NOT EXISTS interests (
    listing_id INTEGER NOT NULL REFERENCES listings(id) ON DELETE CASCADE,
    viewer TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (listing_id, viewer)
);
CREATE INDEX IF NOT EXISTS idx_listings_user ON listings(user_id);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect(path: Path | None = None):
    path = Path(path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init(path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)


# ---- users ---------------------------------------------------------------
def create_user(username: str, password: str, email: str = "", path: Path | None = None) -> str | None:
    """Returns an error key, or None on success."""
    username = (username or "").strip()
    email = (email or "").strip()
    if not auth.username_ok(username):
        return "err_username"
    if not auth.password_ok(password):
        return "err_password"
    if not auth.email_ok(email):
        return "err_email"
    try:
        with connect(path) as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username, email, auth.hash_password(password), _now()),
            )
    except sqlite3.IntegrityError:
        return "err_user_exists"
    return None


def authenticate(username: str, password: str, path: Path | None = None) -> dict | None:
    with connect(path) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", ((username or "").strip(),)).fetchone()
    if row and auth.verify_password(password or "", row["password_hash"]):
        return {"id": row["id"], "username": row["username"], "email": row["email"],
                "created_at": row["created_at"]}
    return None


# ---- listings ------------------------------------------------------------
def add_listing(user_id: int | None, price: float, features: dict, *, title: str = "",
                description: str = "", lat: float | None = None, lon: float | None = None,
                contact: str = "", image: bytes | None = None, image_mime: str | None = None,
                days: int | None = 30, is_demo: bool = False, source_pid: str | None = None,
                created_at: str | None = None, path: Path | None = None) -> int:
    clean = {k: features[k] for k in INPUT_KEYS}
    expires = (datetime.now() + timedelta(days=days)).isoformat(timespec="seconds") if days else None
    with connect(path) as conn:
        cur = conn.execute(
            """INSERT INTO listings (user_id, is_demo, source_pid, title, description, price, features,
                   lat, lon, contact, image, image_mime, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, int(is_demo), source_pid, title.strip()[:120], description.strip()[:2000],
             float(price), json.dumps(clean), lat, lon, contact.strip()[:80], image, image_mime,
             created_at or _now(), expires),
        )
        return int(cur.lastrowid)


def seed_demo_listings(market: pd.DataFrame, pids: list[str], n: int = config.DEMO_LISTINGS,
                       path: Path | None = None) -> int:
    """Publish real hold-out sales (never seen in training) as demo listings."""
    with connect(path) as conn:
        if conn.execute("SELECT COUNT(*) FROM listings WHERE is_demo = 1").fetchone()[0]:
            return 0
    pool = market[market["pid"].isin(pids)]
    pool = pool.sample(n=min(n, len(pool)), random_state=config.RANDOM_STATE)
    rng = np.random.default_rng(config.RANDOM_STATE)
    for _, row in pool.iterrows():
        created = datetime.now() - timedelta(days=int(rng.integers(0, 30)), hours=int(rng.integers(0, 24)))
        add_listing(None, round(row["sale_price"] / 500) * 500, row.to_dict(), lat=row["lat"],
                    lon=row["lon"], days=None, is_demo=True, source_pid=row["pid"],
                    created_at=created.isoformat(timespec="seconds"), path=path)
    return len(pool)


_LISTING_COLUMNS = """l.id, l.user_id, l.is_demo, l.title, l.description, l.price, l.features, l.lat, l.lon,
    l.contact, l.created_at, l.expires_at, (l.image IS NOT NULL) AS has_image,
    (SELECT COUNT(*) FROM interests i WHERE i.listing_id = l.id) AS interest_count,
    (SELECT COUNT(*) FROM favorites f WHERE f.listing_id = l.id) AS favorite_count,
    u.username AS seller"""


def _frame(rows) -> pd.DataFrame:
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    feats = pd.DataFrame([json.loads(f) for f in df.pop("features")])
    return pd.concat([df.reset_index(drop=True), feats], axis=1)


def active_listings(path: Path | None = None) -> pd.DataFrame:
    with connect(path) as conn:
        rows = conn.execute(
            f"""SELECT {_LISTING_COLUMNS} FROM listings l LEFT JOIN users u ON u.id = l.user_id
                WHERE l.expires_at IS NULL OR l.expires_at > ? ORDER BY l.created_at DESC""",
            (_now(),),
        ).fetchall()
    return _frame(rows)


def user_listings(user_id: int, path: Path | None = None) -> pd.DataFrame:
    with connect(path) as conn:
        rows = conn.execute(
            f"""SELECT {_LISTING_COLUMNS} FROM listings l LEFT JOIN users u ON u.id = l.user_id
                WHERE l.user_id = ? ORDER BY l.created_at DESC""",
            (user_id,),
        ).fetchall()
    return _frame(rows)


def listing_image(listing_id: int, path: Path | None = None) -> tuple[bytes, str] | None:
    with connect(path) as conn:
        row = conn.execute("SELECT image, image_mime FROM listings WHERE id = ?", (listing_id,)).fetchone()
    return (row["image"], row["image_mime"]) if row and row["image"] else None


def delete_listing(listing_id: int, user_id: int, path: Path | None = None) -> bool:
    with connect(path) as conn:
        cur = conn.execute("DELETE FROM listings WHERE id = ? AND user_id = ?", (listing_id, user_id))
        return cur.rowcount > 0


def extend_listing(listing_id: int, user_id: int, days: int = 30, path: Path | None = None) -> bool:
    new_expiry = (datetime.now() + timedelta(days=days)).isoformat(timespec="seconds")
    with connect(path) as conn:
        cur = conn.execute("UPDATE listings SET expires_at = ? WHERE id = ? AND user_id = ?",
                           (new_expiry, listing_id, user_id))
        return cur.rowcount > 0


# ---- favourites & interest ----------------------------------------------
def toggle_favorite(user_id: int, listing_id: int, path: Path | None = None) -> bool:
    """Returns True if the listing is now a favourite."""
    with connect(path) as conn:
        cur = conn.execute("DELETE FROM favorites WHERE user_id = ? AND listing_id = ?",
                           (user_id, listing_id))
        if cur.rowcount:
            return False
        conn.execute("INSERT INTO favorites (user_id, listing_id, created_at) VALUES (?, ?, ?)",
                     (user_id, listing_id, _now()))
        return True


def favorite_ids(user_id: int, path: Path | None = None) -> set[int]:
    with connect(path) as conn:
        rows = conn.execute("SELECT listing_id FROM favorites WHERE user_id = ?", (user_id,)).fetchall()
    return {r[0] for r in rows}


def add_interest(listing_id: int, viewer: str, path: Path | None = None) -> bool:
    """Returns True if this viewer's interest is new."""
    with connect(path) as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO interests (listing_id, viewer, created_at) VALUES (?, ?, ?)",
            (listing_id, viewer, _now()),
        )
        return cur.rowcount > 0
