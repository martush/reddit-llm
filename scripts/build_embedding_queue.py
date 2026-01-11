#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import duckdb
from dotenv import load_dotenv, find_dotenv


def get_paths():
    load_dotenv(find_dotenv(usecwd=False))
    base_dir = Path(os.environ["BASE_DIR"]).expanduser().resolve()
    db_path = Path(os.environ.get("DB_PATH", base_dir / "data" / "reddit.duckdb")).expanduser().resolve()
    return base_dir, db_path


def ensure_queue(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
    CREATE TABLE IF NOT EXISTS embedding_queue (
      doc_id TEXT PRIMARY KEY,
      source_type TEXT NOT NULL,
      post_id TEXT,
      comment_id TEXT,
      subreddit TEXT,
      created_utc TIMESTAMP,
      score BIGINT,
      num_comments BIGINT,
      url TEXT,
      title TEXT,
      text TEXT NOT NULL,
      char_len BIGINT NOT NULL,
      inserted_at TIMESTAMP DEFAULT NOW(),
      embedded BOOLEAN DEFAULT FALSE,
      embedded_at TIMESTAMP
    );
    """)


def build_post_text(title: str | None, body: str | None) -> str:
    title = (title or "").strip()
    body = (body or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def main() -> int:
    _, db_path = get_paths()

    # Tune via env if you want
    lookback_hours = int(os.environ.get("EMBED_LOOKBACK_HOURS", "24"))
    top_posts_per_sub = int(os.environ.get("TOP_POSTS_PER_SUBREDDIT", "20"))
    top_comments_per_sub = int(os.environ.get("TOP_COMMENTS_PER_SUBREDDIT", "80"))
    min_comment_chars = int(os.environ.get("MIN_COMMENT_CHARS", "80"))
    max_text_chars = int(os.environ.get("MAX_TEXT_CHARS", "6000"))  # truncate to keep embeddings cheap

    since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    con = duckdb.connect(str(db_path))
    print('Connecting to duckdb')
    try:
        ensure_queue(con)

        # --------
        # POSTS
        # --------
        # We take top posts per subreddit by (num_comments, score)
        # Use a window function to pick top N per subreddit.
        posts = con.execute(f"""
            WITH ranked AS (
              SELECT
                post_id,
                subreddit,
                title,
                body,
                score,
                num_comments,
                created_utc,
                COALESCE(url, 'https://www.reddit.com/comments/' || post_id) AS url,
                ROW_NUMBER() OVER (
                  PARTITION BY subreddit
                  ORDER BY num_comments DESC, score DESC
                ) AS rn
              FROM posts
              WHERE created_utc >= ?
            )
            SELECT post_id, subreddit, title, body, score, num_comments, created_utc, url
            FROM ranked
            WHERE rn <= {top_posts_per_sub}
        """, [since.replace(tzinfo=None)]).fetchall()

        inserted_posts = 0
        for post_id, subreddit, title, body, score, num_comments, created_utc, url in posts:
            text = build_post_text(title, body)
            text = (text or "").strip()
            if not text:
                continue
            if len(text) > max_text_chars:
                text = text[:max_text_chars]

            doc_id = f"post:{post_id}"
            con.execute("""
                INSERT OR IGNORE INTO embedding_queue
                (doc_id, source_type, post_id, comment_id, subreddit, created_utc, score, num_comments, url, title, text, char_len)
                VALUES (?, 'post', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                post_id,
                subreddit,
                created_utc,
                int(score) if score is not None else 0,
                int(num_comments) if num_comments is not None else 0,
                url,
                title,
                text,
                len(text)
            ))
            inserted_posts += 1

        # --------
        # COMMENTS
        # --------
        # Top comments per subreddit by score, excluding deleted/removed, enforcing min length.
        print('getting comments')
        comments = con.execute(f"""
            WITH filtered AS (
              SELECT
                comment_id,
                post_id,
                subreddit,
                body,
                score,
                created_utc,
                ROW_NUMBER() OVER (
                  PARTITION BY subreddit
                  ORDER BY score DESC
                ) AS rn
              FROM comments
              WHERE created_utc >= ?
                AND body IS NOT NULL
                AND body NOT IN ('[deleted]', '[removed]')
                AND LENGTH(TRIM(body)) >= {min_comment_chars}
            )
            SELECT comment_id, post_id, subreddit, body, score, created_utc
            FROM filtered
            WHERE rn <= {top_comments_per_sub}
        """, [since.replace(tzinfo=None)]).fetchall()

        inserted_comments = 0
        for comment_id, post_id, subreddit, body, score, created_utc in comments:
            text = (body or "").strip()
            if not text:
                continue
            if len(text) > max_text_chars:
                text = text[:max_text_chars]

            doc_id = f"comment:{comment_id}"
            # We can link comments to the post URL; comment permalinks require extra work.
            url = f"https://www.reddit.com/comments/{post_id}"

            con.execute("""
                INSERT OR IGNORE INTO embedding_queue
                (doc_id, source_type, post_id, comment_id, subreddit, created_utc, score, num_comments, url, title, text, char_len)
                VALUES (?, 'comment', ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?)
            """, (
                doc_id,
                post_id,
                comment_id,
                subreddit,
                created_utc,
                int(score) if score is not None else 0,
                url,
                text,
                len(text)
            ))
            inserted_comments += 1

        print(f"[ok] queue build done. candidates: posts={len(posts)} comments={len(comments)}")
        print(f"[ok] inserted/ignored: posts={inserted_posts} comments={inserted_comments}")

    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
