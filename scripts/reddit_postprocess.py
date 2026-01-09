#!/home/martina/anaconda3/envs/reddit-llm/bin/python

import re
import os
import duckdb
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

load_dotenv(find_dotenv(usecwd=False))
BASE_DIR = Path(os.environ["BASE_DIR"])
print(f'Base dir: {BASE_DIR}')
DB_PATH = f"{BASE_DIR}/data/reddit.duckdb"


TICKER_RE = re.compile(r'\$?[A-Z]{1,5}\b')

# List of words which are commonly misinterpreted as tickers
BLACKLIST = {
    "CEO", "CFO", "IMO", "USA", "GDP", "YOLO", "FOMO",
    "ETF", "ATH", "EPS", "IRS", "SEC", "FED"
}

# Function to retrieve actual tickers
def extract_tickers(text):
    '''
    Function which extracts potential tickers
    from input text
    '''

    if not text:
        # return an emtpy set
        return set()

    possible_tickers = {
        m.group(0).lstrip("$")
        for m in TICKER_RE.finditer(text.upper())
    }

    return {
        t for t in possible_tickers
        if t not in BLACKLIST
    }


def filter_real_tickers(con, tickers):
    '''
    Function which checks a list of potential tickers
    against a table with actual tickers
    '''


    if not tickers:
        return set()

    query = """
        SELECT ticker FROM tickers
        WHERE ticker IN ({})
    """.format(",".join("?" * len(tickers)))

    rows = con.execute(query, list(tickers)).fetchall()
    return {r[0] for r in rows}


BULLISH = {
    "buy", "calls", "moon", "rocket", "bull", "long",
    "undervalued", "breakout", "upside", "squeeze"
}

BEARISH = {
    "sell", "puts", "dump", "short", "overvalued",
    "bagholder", "crash", "downside"
}


def detect_direction(text):
    text = text.lower()

    bull = sum(1 for w in BULLISH if w in text)
    bear = sum(1 for w in BEARISH if w in text)

    if bull > bear and bull > 0:
        return "bullish", bull / (bull + bear)
    if bear > bull and bear > 0:
        return "bearish", bear / (bull + bear)

    return "neutral", 0.0



con = duckdb.connect(DB_PATH)

comments = con.execute("""
    SELECT comment_id, body
    FROM comments
    WHERE comment_id NOT IN (
        SELECT DISTINCT comment_id FROM comment_tickers
    )
""").fetchall()

for comment_id, body in comments:
    raw = extract_tickers(body)
    real = filter_real_tickers(con, raw)

    for ticker in real:
        direction, confidence = detect_direction(body)

        con.execute("""
            INSERT OR IGNORE INTO comment_tickers
            VALUES (?, ?, ?, ?)
        """, (comment_id, ticker, direction, confidence))

con.close()
print('Reddit ticker and sentiment extraction completed.')