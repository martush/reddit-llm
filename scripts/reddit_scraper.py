#!/home/martina/anaconda3/envs/reddit-llm/bin/python

import os
import duckdb
import praw
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

#load_dotenv(f"{BASE_DIR}/.env")
load_dotenv(find_dotenv(usecwd=False))

#BASE_DIR = "/home/martina/Desktop/Git/reddit-llm"
BASE_DIR = Path(os.environ["BASE_DIR"])

DB_PATH = f"{BASE_DIR}/data/reddit.duckdb"

SUBREDDITS = {
    "wallstreetbets" : {"limit": 30, "mode": "top_day", "min_comments": 200},
    "stocks"         : {"limit": 20, "mode": "hot", "min_comments": 50},
    "investing"      : {"limit": 20, "mode": "hot", "min_comments": 30},
    "StockMarket"    : {"limit": 20, "mode": "hot", "min_comments": 30},
    "options"        : {"limit": 20, "mode": "hot", "min_comments": 20},
    "ValueInvesting" : {"limit": 20, "mode": "hot", "min_comments": 10}
}


reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent=os.environ["REDDIT_USER_AGENT"],
)
print('Created reddit client')

con = duckdb.connect(DB_PATH)
print('Connected to db')

for subreddit_name, cfg in SUBREDDITS.items():
    subreddit = reddit.subreddit(subreddit_name)

    if cfg["mode"] == "top_day":
        posts = subreddit.top(time_filter="day", limit=cfg["limit"])
    elif cfg["mode"] == "new":
        posts = subreddit.new(limit=cfg["limit"])
    else:
        posts = subreddit.hot(limit=cfg["limit"])

    for post in posts:
        con.execute("""
            INSERT OR IGNORE INTO posts
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            post.id,
            subreddit_name,
            post.title,
            post.selftext,
            post.score,
            post.num_comments,
            datetime.fromtimestamp(post.created_utc)
        ))

        # Skip low-engagement posts to reduce noise
        if post.num_comments < 50:
            continue

        post.comments.replace_more(limit=0)

        for c in post.comments.list():
            if not c.body or c.body in ("[deleted]", "[removed]"):
                continue

            con.execute("""
                INSERT OR IGNORE INTO comments
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                c.id,
                post.id,
                c.parent_id,
                c.body,
                c.score,
                datetime.fromtimestamp(c.created_utc)
            ))

con.close()
print("Reddit scrape completed.")
