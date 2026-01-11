# scripts/init_db.py
import duckdb
from pathlib import Path

DB_PATH = Path("data/reddit.duckdb")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH))

con.execute("""
CREATE TABLE IF NOT EXISTS posts (
    post_id TEXT PRIMARY KEY,
    subreddit TEXT,
    title TEXT,
    body TEXT,
    score INTEGER,
    num_comments INTEGER,
    created_utc TIMESTAMP
);
""")

con.execute("""
CREATE TABLE IF NOT EXISTS comments (
    comment_id TEXT PRIMARY KEY,
    post_id TEXT,
    parent_id TEXT,
    body TEXT,
    score INTEGER,
    created_utc TIMESTAMP
);
""")

con.execute("""
CREATE TABLE IF NOT EXISTS comment_tickers (
    comment_id TEXT,
    ticker TEXT,
    direction TEXT,   -- bullish | bearish | neutral | unknown
    confidence DOUBLE,
    PRIMARY KEY (comment_id, ticker)
);
""")

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

con.execute("""
CREATE TABLE IF NOT EXISTS embedding_queue (
  doc_id TEXT PRIMARY KEY,              -- stable unique id
  source_type TEXT NOT NULL,            -- 'post' or 'comment'
  post_id TEXT,                         -- for both posts and comments
  comment_id TEXT,                      -- only for comments
  subreddit TEXT,
  created_utc TIMESTAMP,
  score BIGINT,                         -- upvotes score
  num_comments BIGINT,                  -- only for posts (nullable for comments)
  url TEXT,                             -- reddit URL if available
  title TEXT,                           -- posts only
  text TEXT NOT NULL,                   -- text we embed
  char_len BIGINT NOT NULL,
  inserted_at TIMESTAMP DEFAULT NOW(),
  embedded BOOLEAN DEFAULT FALSE,
  embedded_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_embedding_queue_embedded ON embedding_queue(embedded);
CREATE INDEX IF NOT EXISTS idx_embedding_queue_created ON embedding_queue(created_utc);


""")


con.close()
print("DuckDB initialized.")
