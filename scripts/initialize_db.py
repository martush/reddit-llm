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

con.close()
print("DuckDB initialized.")
