#!/usr/bin/env python3
"""
refresh_tickers.py

Downloads Nasdaq's daily-updated symbol directory (covers all US exchanges)
and upserts into the DuckDB tickers table.

On first run it also migrates the existing table:
  - deduplicates rows
  - adds exchange, active, refreshed_at columns
  - enforces a PRIMARY KEY on ticker

Sources (updated each trading day by Nasdaq):
  NASDAQ   : https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
  All other: https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt

Suggested schedule: weekly (new listings are infrequent).
Example crontab (every Monday 07:00):
  0 7 * * 1  /home/martina/anaconda3/envs/reddit-llm/bin/python /home/martina/Desktop/Git/reddit-llm/scripts/refresh_tickers.py

Usage:
  python scripts/refresh_tickers.py
  python scripts/refresh_tickers.py --dry-run
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import requests
from dotenv import load_dotenv, find_dotenv

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL  = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# otherlisted.txt Exchange column codes
EXCHANGE_CODE = {
    "A": "NYSE MKT",
    "N": "NYSE",
    "P": "NYSE ARCA",
    "Z": "BATS",
    "V": "IEX",
}


# -------------------------
# Fetch + parse
# -------------------------

def _fetch(url: str) -> str:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "reddit-llm/ticker-refresh"})
    resp.raise_for_status()
    return resp.text


def _parse_nasdaq(text: str) -> list[tuple[str, str, str]]:
    """
    nasdaqlisted.txt columns (pipe-delimited):
      [0] Symbol  [1] Security Name  [2] Market Category
      [3] Test Issue  [4] Financial Status  [5] Round Lot Size
      [6] ETF  [7] NextShares
    """
    rows: list[tuple[str, str, str]] = []
    lines = text.splitlines()
    for line in lines[1:]:                         # skip header
        if line.startswith("File Creation Time"):
            break
        parts = line.split("|")
        if len(parts) < 4:
            continue
        ticker = parts[0].strip()
        name   = parts[1].strip()
        test   = parts[3].strip()                  # Test Issue
        if not ticker or test == "Y":
            continue
        rows.append((ticker, name, "NASDAQ"))
    return rows


def _parse_other(text: str) -> list[tuple[str, str, str]]:
    """
    otherlisted.txt columns (pipe-delimited):
      [0] ACT Symbol  [1] Security Name  [2] Exchange
      [3] CQS Symbol  [4] ETF  [5] Round Lot Size
      [6] Test Issue  [7] NASDAQ Symbol
    """
    rows: list[tuple[str, str, str]] = []
    lines = text.splitlines()
    for line in lines[1:]:                         # skip header
        if line.startswith("File Creation Time"):
            break
        parts = line.split("|")
        if len(parts) < 7:
            continue
        ticker    = parts[0].strip()               # ACT Symbol (what traders use)
        name      = parts[1].strip()
        exch_code = parts[2].strip()
        test      = parts[6].strip()               # Test Issue
        if not ticker or test == "Y":
            continue
        exchange = EXCHANGE_CODE.get(exch_code, exch_code)
        rows.append((ticker, name, exchange))
    return rows


def _dedupe(rows: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Keep first occurrence per ticker (NASDAQ file is processed first)."""
    seen: set[str] = set()
    out = []
    for ticker, name, exchange in rows:
        if ticker not in seen:
            seen.add(ticker)
            out.append((ticker, name, exchange))
    return out


# -------------------------
# Schema migration
# -------------------------

def migrate_schema(con: duckdb.DuckDBPyConnection) -> None:
    """
    Ensure tickers has the correct schema. Handles:
      - table missing entirely
      - old 2-column schema (ticker TEXT, name TEXT) with possible duplicates and no PK
    """
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

    if "tickers" not in tables:
        con.execute("""
            CREATE TABLE tickers (
                ticker       TEXT PRIMARY KEY,
                name         TEXT,
                exchange     TEXT,
                active       BOOLEAN DEFAULT TRUE,
                refreshed_at TIMESTAMP
            )
        """)
        print("[schema] Created tickers table.")
        return

    cols = {r[0] for r in con.execute("DESCRIBE tickers").fetchall()}
    if "exchange" in cols and "active" in cols:
        return  # already on new schema

    print("[migrate] Rebuilding tickers table: deduplicating + adding columns...")
    old_count = con.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]

    con.execute("DROP TABLE IF EXISTS tickers_new")
    con.execute("""
        CREATE TABLE tickers_new (
            ticker       TEXT PRIMARY KEY,
            name         TEXT,
            exchange     TEXT,
            active       BOOLEAN DEFAULT TRUE,
            refreshed_at TIMESTAMP
        )
    """)
    # MIN(name) to pick one name deterministically when rows are exact duplicates.
    # Filter out the two NULL-ticker rows that existed in the original data load.
    con.execute("""
        INSERT INTO tickers_new (ticker, name)
        SELECT ticker, MIN(name)
        FROM tickers
        WHERE ticker IS NOT NULL
        GROUP BY ticker
    """)
    new_count = con.execute("SELECT COUNT(*) FROM tickers_new").fetchone()[0]
    con.execute("DROP TABLE tickers")
    con.execute("ALTER TABLE tickers_new RENAME TO tickers")
    print(f"[migrate] Done: {old_count} rows → {new_count} unique tickers.")


# -------------------------
# Refresh
# -------------------------

def refresh(con: duckdb.DuckDBPyConnection, rows: list[tuple], dry_run: bool) -> None:
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    con.execute("""
        CREATE TEMP TABLE _staging (ticker TEXT, name TEXT, exchange TEXT)
    """)
    con.executemany("INSERT INTO _staging VALUES (?, ?, ?)", rows)

    staging_count = con.execute("SELECT COUNT(*) FROM _staging").fetchone()[0]

    # Preview counts before any writes
    new_count = con.execute("""
        SELECT COUNT(*) FROM _staging s
        WHERE NOT EXISTS (SELECT 1 FROM tickers t WHERE t.ticker = s.ticker)
    """).fetchone()[0]

    deactivate_count = con.execute("""
        SELECT COUNT(*) FROM tickers t
        WHERE t.active = TRUE
        AND NOT EXISTS (SELECT 1 FROM _staging s WHERE s.ticker = t.ticker)
    """).fetchone()[0]

    print(f"[refresh] Source: {staging_count} tickers | New: {new_count} | To deactivate: {deactivate_count}")

    if dry_run:
        print("[dry-run] No changes written.")
        con.execute("DROP TABLE _staging")
        return

    # Insert brand-new tickers
    con.execute("""
        INSERT INTO tickers (ticker, name, exchange, active, refreshed_at)
        SELECT s.ticker, s.name, s.exchange, TRUE, ?
        FROM _staging s
        WHERE NOT EXISTS (SELECT 1 FROM tickers t WHERE t.ticker = s.ticker)
    """, [now])

    # Update existing: refresh name, exchange, mark active
    con.execute("""
        UPDATE tickers
        SET name         = s.name,
            exchange     = s.exchange,
            active       = TRUE,
            refreshed_at = ?
        FROM _staging s
        WHERE tickers.ticker = s.ticker
    """, [now])

    # Mark tickers no longer in source as inactive
    con.execute("""
        UPDATE tickers
        SET active = FALSE
        WHERE active = TRUE
        AND ticker NOT IN (SELECT ticker FROM _staging)
    """)

    con.execute("DROP TABLE _staging")

    total    = con.execute("SELECT COUNT(*) FROM tickers").fetchone()[0]
    active   = con.execute("SELECT COUNT(*) FROM tickers WHERE active = TRUE").fetchone()[0]
    inactive = total - active
    print(f"[ok] Tickers table: {total} total | {active} active | {inactive} inactive.")


# -------------------------
# Entrypoint
# -------------------------

def main() -> int:
    load_dotenv(find_dotenv(usecwd=False))
    base_dir = Path(os.environ["BASE_DIR"]).expanduser().resolve()
    db_path  = Path(os.environ.get("DB_PATH", base_dir / "data" / "reddit.duckdb")).expanduser().resolve()

    dry_run = "--dry-run" in sys.argv[1:]

    print("[info] Fetching NASDAQ listed symbols...")
    nasdaq_rows = _parse_nasdaq(_fetch(NASDAQ_URL))
    print(f"[info] NASDAQ: {len(nasdaq_rows)} symbols")

    print("[info] Fetching other US listed symbols...")
    other_rows = _parse_other(_fetch(OTHER_URL))
    print(f"[info] Other: {len(other_rows)} symbols")

    all_rows = _dedupe(nasdaq_rows + other_rows)
    print(f"[info] Combined unique: {len(all_rows)} symbols")

    with duckdb.connect(str(db_path)) as con:
        migrate_schema(con)
        refresh(con, all_rows, dry_run=dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
