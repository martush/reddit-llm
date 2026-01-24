#!/usr/bin/env python3
"""
reddit_postprocess.py

Post-processing pipeline (no ticker loading/creation):
- Loads config from .env (BASE_DIR required)
- Uses existing DuckDB 'tickers' table (must already exist and be populated)
- Ensures comment_tickers and post_tickers exists
- Extracts WSB-safe tickers from new comments
- Tags direction (bullish/bearish/neutral)
- Inserts into comment_tickers without duplicates

Run:
  python reddit_postprocess.py
  python reddit_postprocess.py --rebuild --hours=48
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta, timezone

import duckdb
from dotenv import load_dotenv, find_dotenv


# -------------------------
# Config
# -------------------------

@dataclass(frozen=True)
class Config:
    base_dir: Path
    db_path: Path
    lookback_hours: int
    min_comment_len: int
    min_comment_score: int
    verbose: bool


def load_config() -> Config:
    dotenv_path = find_dotenv(usecwd=False)
    if not dotenv_path:
        raise RuntimeError("Could not find .env")

    load_dotenv(dotenv_path)

    base_dir_str = os.environ.get("BASE_DIR")
    if not base_dir_str:
        raise RuntimeError("Missing BASE_DIR in .env (e.g. BASE_DIR=/home/martina/Desktop/Git/reddit-llm)")

    base_dir = Path(base_dir_str).expanduser().resolve()
    db_path = Path(os.environ.get("DB_PATH", str(base_dir / "data" / "reddit.duckdb"))).expanduser().resolve()

    lookback_hours = int(os.environ.get("LOOKBACK_HOURS", "36"))
    min_comment_len = int(os.environ.get("MIN_COMMENT_LEN", "20"))
    min_comment_score = int(os.environ.get("MIN_COMMENT_SCORE", "-2"))
    verbose = os.environ.get("VERBOSE", "0") in ("1", "true", "True", "yes", "YES")

    return Config(
        base_dir=base_dir,
        db_path=db_path,
        lookback_hours=lookback_hours,
        min_comment_len=min_comment_len,
        min_comment_score=min_comment_score,
        verbose=verbose,
    )


# -------------------------
# DB setup
# -------------------------

def ensure_comment_tickers_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS comment_tickers (
            comment_id TEXT,
            ticker TEXT,
            direction TEXT,     -- bullish|bearish|neutral|unknown
            confidence DOUBLE,  -- 0..1
            method TEXT,        -- dollar|caps_context|caps_repeat|caps_plain
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (comment_id, ticker)
        );
    """)

def ensure_post_tickers_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS post_tickers (
            post_id TEXT,
            ticker TEXT,
            direction TEXT,
            confidence DOUBLE,
            method TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (post_id, ticker)
        );
    """)


def assert_tickers_table_exists(con: duckdb.DuckDBPyConnection) -> None:
    # Ensure the tickers table exists and has rows
    try:
        n = con.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]
    except Exception as e:
        raise RuntimeError(
            "Missing DuckDB table 'tickers'. You said you've created it, but I can't query it."
        ) from e

    if n == 0:
        raise RuntimeError("DuckDB table 'tickers' exists but is empty. Populate it first.")


# -------------------------
# Extraction
# -------------------------

# Capture tokens that could be tickers ($TSLA, TSLA, etc.) but do NOT uppercase the text first.
CAND_RE = re.compile(r'(?<![A-Za-z0-9])(\$?[A-Za-z]{1,5})(?![A-Za-z0-9])')

CONTEXT_WORDS = {
    "ticker", "stock", "shares", "share", "equity",
    "calls", "puts", "option", "options",
    "buy", "sell", "long", "short",
    "earnings", "guidance", "revenue", "ipo",
    "pump", "dump", "squeeze", "dd", "pt",
}

# Extremely ambiguous tickers that are also common English words
AMBIGUOUS_ENGLISH = {
    "A", "AN", "ON", "OR", "FOR", "AS", "AT", "IN", "BY", "UP",
    "NOW", "HAS", "HAD", "ANY", "ALL", "ONE", "WAY", "MOVE",
    "NICE", "GOLD", "BOND", "FAST", "GOOD", "BEST", "NEXT",
    "ARE", "WAS", "YOU", "ME", "WE", "US", "IT", "ITS",
}

# Common acronyms / reddit terms that shouldn’t be treated as tickers
BLACKLIST = {
    "CEO", "CFO", "SEC", "FED", "GDP", "USA", "IMO",
    "YOLO", "FOMO", "ATH", "EPS", "TOS", "EDIT",
}

HIGH_FALSE_POSITIVE = {
    "ICE",
    "NATO",
    "UAE",
    "IBKR",
    "USA",
    "EU",
    "UK",
    "IRA",
    "IRS",
    "SEC",
    "FED",
    "GDP",
    "CPI",
    "PPI",
    "FOMC",
    "IMF",
    "OPEC",
    "ETF",
}


# Acronyms that are extremely common in discussion; treat as ticker only if $-prefixed
RESERVED_ACRONYMS_REQUIRE_DOLLAR = {"AI", "TACO"}


# Strong trading-intent words; used for short tickers
TIGHT_CONTEXT_WORDS = {
    "calls", "puts", "call", "put",
    "buy", "sell", "long", "short",
    "earnings", "guidance"
}

def _has_tight_context(text_lower: str, start: int, end: int) -> bool:
    left = max(0, start - 12)
    right = min(len(text_lower), end + 12)
    neighborhood = text_lower[left:right]
    return any(w in neighborhood for w in TIGHT_CONTEXT_WORDS)

def _looks_like_article_A(text: str, start: int, end: int) -> bool:
    # If token is "A" and next non-space char starts a lowercase word, it's likely the article "A"
    i = end
    while i < len(text) and text[i].isspace():
        i += 1
    return i < len(text) and text[i].islower()



def _has_context(text_lower: str, start: int, end: int, window: int = 60) -> bool:
    left = max(0, start - window)
    right = min(len(text_lower), end + window)
    neighborhood = text_lower[left:right]
    return any(w in neighborhood for w in CONTEXT_WORDS)


def extract_tickers_wsb(text: str) -> dict[str, str]:
    """
    Returns dict[ticker] = method
    method in: dollar | caps_context | caps_repeat | caps_plain | caps_tight_context

    Rules:
      - Accept $TICKER always (strong intent), except BLACKLIST
      - Accept bare symbol only if ALL CAPS in original text
      - 1-letter tickers: accept ONLY with $ or tight trading context (and reject article-like "A ...")
      - 2-letter tickers: require context (looser ok)
      - Reserved acronyms (e.g. AI): only accept with $
      - Ambiguous English-word tickers: require context or repetition (but repetition NOT allowed for 1-letter)
    """
    if not text:
        return {}

    text_lower = text.lower()
    out: dict[str, str] = {}

    for m in CAND_RE.finditer(text):
        raw = m.group(1)
        sym = raw.lstrip("$")

        if not sym.isalpha():
            continue

        sym_up = sym.upper()

        if sym_up in BLACKLIST:
            continue

        if sym_up in HIGH_FALSE_POSITIVE and not raw.startswith("$"):
            # require context much stronger than normal
            if not _has_tight_context(text_lower, m.start(1), m.end(1)):
                continue

        # Reserved acronym: require $-prefix
        if sym_up in RESERVED_ACRONYMS_REQUIRE_DOLLAR and not raw.startswith("$"):
            continue

        # Strong signal: $ prefix
        if raw.startswith("$"):
            out[sym_up] = "dollar"
            continue

        # Must be ALL CAPS in original text (prevents "nice" -> NICE, "Gold" -> GOLD)
        if sym != sym_up:
            continue

        # Special handling: 1-letter tickers (A, U, F, etc.) are almost always noise
        if len(sym_up) == 1:
            # Filter article-like "A such", "A company", etc.
            if sym_up == "A" and _looks_like_article_A(text, m.start(1), m.end(1)):
                continue
            # Require tight trading context (e.g., "A calls", "buy A", "A earnings")
            if _has_tight_context(text_lower, m.start(1), m.end(1)):
                out[sym_up] = "caps_tight_context"
            continue

        # Short tickers (2 letters) require some context
        if len(sym_up) == 2 and not _has_context(text_lower, m.start(1), m.end(1)):
            continue

        # English-word tickers require context or repetition
        if sym_up in AMBIGUOUS_ENGLISH:
            # IMPORTANT: never use repetition for 1-letter tickers (handled above)
            repeated = (len(sym_up) >= 2) and (text.count(sym_up) >= 2)
            if repeated:
                out[sym_up] = "caps_repeat"
            elif _has_context(text_lower, m.start(1), m.end(1)):
                out[sym_up] = "caps_context"
            else:
                continue
        else:
            out[sym_up] = "caps_context" if _has_context(text_lower, m.start(1), m.end(1)) else "caps_plain"

    return out


# -------------------------
# Direction tagging (simple rules)
# -------------------------

BULLISH = {
    "buy", "calls", "call", "moon", "rocket", "bull", "long",
    "undervalued", "breakout", "upside", "squeeze", "rip", "pump",
}
BEARISH = {
    "sell", "puts", "put", "dump", "short", "overvalued",
    "bagholder", "crash", "downside", "rug", "drill",
}


def detect_direction(text: str) -> tuple[str, float]:
    t = (text or "").lower()
    bull = sum(1 for w in BULLISH if w in t)
    bear = sum(1 for w in BEARISH if w in t)

    if bull == 0 and bear == 0:
        return "neutral", 0.0
    if bull > bear:
        return "bullish", bull / (bull + bear)
    if bear > bull:
        return "bearish", bear / (bull + bear)
    return "neutral", 0.5


def comment_is_useful(body: str, score: int | None, cfg: Config) -> bool:
    if not body:
        return False
    if body in ("[deleted]", "[removed]"):
        return False
    if len(body.strip()) < cfg.min_comment_len:
        return False
    if score is not None and score < cfg.min_comment_score:
        return False
    return True


# -------------------------
# Validation against real tickers
# -------------------------

def filter_real_tickers(con: duckdb.DuckDBPyConnection, symbols: list[str]) -> set[str]:
    if not symbols:
        return set()

    q = f"SELECT ticker FROM tickers WHERE ticker IN ({','.join(['?'] * len(symbols))})"
    rows = con.execute(q, symbols).fetchall()
    return {r[0] for r in rows}


# -------------------------
# Main processing
# -------------------------

def process_new_comments(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    since = datetime.now(timezone.utc) - timedelta(hours=cfg.lookback_hours)

    # Only process comments that are not yet in comment_tickers
    rows = con.execute("""
        SELECT c.comment_id, c.body, c.score
        FROM comments c
        LEFT JOIN (SELECT DISTINCT comment_id FROM comment_tickers) ct
          ON c.comment_id = ct.comment_id
        WHERE ct.comment_id IS NULL
          AND c.created_utc >= ?
    """, [since.replace(tzinfo=None)]).fetchall()

    if cfg.verbose:
        print(f"[info] candidate new comments since {since.isoformat()}: {len(rows)}")

    inserted = 0
    skipped_not_useful = 0
    skipped_no_ticker = 0
    skipped_not_real = 0

    for comment_id, body, score in rows:
        if not comment_is_useful(body, score, cfg):
            skipped_not_useful += 1
            continue

        extracted = extract_tickers_wsb(body)  # dict[ticker]=method
        if not extracted:
            skipped_no_ticker += 1
            continue

        real = filter_real_tickers(con, list(extracted.keys()))
        if not real:
            skipped_not_real += 1
            continue

        direction, conf = detect_direction(body)

        for ticker in real:
            con.execute("""
                INSERT OR IGNORE INTO comment_tickers (comment_id, ticker, direction, confidence, method)
                VALUES (?, ?, ?, ?, ?)
            """, (comment_id, ticker, direction, float(conf), extracted.get(ticker, "unknown")))
            inserted += 1

    print(f"[ok] inserted rows into comment_tickers: {inserted}")
    if cfg.verbose:
        print(f"[info] skipped_not_useful={skipped_not_useful} skipped_no_ticker={skipped_no_ticker} skipped_not_real={skipped_not_real}")


def rebuild_comment_tickers(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    """
    Recompute comment_tickers for the lookback window.
    Use when you change extraction rules and want a clean recompute.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=cfg.lookback_hours)

    # Wipe recent entries (or wipe all if you prefer)
    con.execute("DELETE FROM comment_tickers")

    rows = con.execute("""
        SELECT c.comment_id, c.body, c.score
        FROM comments c
        WHERE c.created_utc >= ?
    """, [since.replace(tzinfo=None)]).fetchall()

    print(f"[info] rebuilding from comments since {since.isoformat()} -> {len(rows)} comments")

    inserted = 0
    for comment_id, body, score in rows:
        if not comment_is_useful(body, score, cfg):
            continue

        extracted = extract_tickers_wsb(body)
        if not extracted:
            continue

        real = filter_real_tickers(con, list(extracted.keys()))
        if not real:
            continue

        direction, conf = detect_direction(body)

        for ticker in real:
            con.execute("""
                INSERT OR IGNORE INTO comment_tickers (comment_id, ticker, direction, confidence, method)
                VALUES (?, ?, ?, ?, ?)
            """, (comment_id, ticker, direction, float(conf), extracted.get(ticker, "unknown")))
            inserted += 1

    print(f"[ok] rebuild inserted rows: {inserted}")


def process_new_posts(con, cfg):
    since = datetime.now(timezone.utc) - timedelta(hours=cfg.lookback_hours)

    rows = con.execute("""
        SELECT p.post_id, p.title, p.body
        FROM posts p
        LEFT JOIN (SELECT DISTINCT post_id FROM post_tickers) pt
          ON p.post_id = pt.post_id
        WHERE pt.post_id IS NULL
          AND p.created_utc >= ?
    """, [since.replace(tzinfo=None)]).fetchall()

    inserted = 0

    for post_id, title, body in rows:
        text = (title or "") + "\n\n" + (body or "")
        extracted = extract_tickers_wsb(text)
        if not extracted:
            continue

        real = filter_real_tickers(con, list(extracted.keys()))
        if not real:
            continue

        direction, conf = detect_direction(text)

        for ticker in real:
            method = extracted.get(ticker, "unknown")
            con.execute("""
                INSERT OR IGNORE INTO post_tickers (post_id, ticker, direction, confidence, method)
                VALUES (?, ?, ?, ?, ?)
            """, (post_id, ticker, direction, float(conf), method))
            inserted += 1

    print(f"[ok] inserted rows into post_tickers: {inserted}")


def rebuild_post_tickers(con, cfg):
    since = datetime.now(timezone.utc) - timedelta(hours=cfg.lookback_hours)

    # wipe extracted post tickers
    con.execute("DELETE FROM post_tickers")

    rows = con.execute("""
        SELECT post_id, title, body
        FROM posts
        WHERE created_utc >= ?
    """, [since.replace(tzinfo=None)]).fetchall()

    inserted = 0
    for post_id, title, body in rows:
        text = (title or "") + "\n\n" + (body or "")

        extracted = extract_tickers_wsb(text)  # dict[ticker]=method
        if not extracted:
            continue

        real = filter_real_tickers(con, list(extracted.keys()))
        if not real:
            continue

        direction, conf = detect_direction(text)

        for ticker in real:
            method = extracted.get(ticker, "unknown")
            con.execute("""
                INSERT OR IGNORE INTO post_tickers (post_id, ticker, direction, confidence, method)
                VALUES (?, ?, ?, ?, ?)
            """, (post_id, ticker, direction, float(conf), method))
            inserted += 1

    print(f"[ok] rebuild inserted rows into post_tickers: {inserted}")


# -------------------------
# Entrypoint
# -------------------------

def main() -> int:
    cfg = load_config()
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)

    args = sys.argv[1:]
    do_rebuild = "--rebuild" in args
    do_top = "--top" in args

    top_hours = 24
    for a in args:
        if a.startswith("--hours="):
            top_hours = int(a.split("=", 1)[1])

    if cfg.verbose:
        print(f"[info] BASE_DIR={cfg.base_dir}")
        print(f"[info] DB_PATH={cfg.db_path}")

    con = duckdb.connect(str(cfg.db_path))
    try:
        ensure_comment_tickers_table(con)
        ensure_post_tickers_table(con)
        assert_tickers_table_exists(con)

        if do_rebuild:
            rebuild_post_tickers(con, cfg)
            rebuild_comment_tickers(con, cfg)
        else:
            process_new_posts(con, cfg)
            process_new_comments(con, cfg)

    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
