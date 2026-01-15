#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import duckdb
from dotenv import load_dotenv, find_dotenv

import chromadb
from sentence_transformers import SentenceTransformer


def main() -> int:
    load_dotenv(find_dotenv(usecwd=False))
    base_dir = Path(os.environ["BASE_DIR"]).expanduser().resolve()
    db_path = Path(os.environ.get("DB_PATH", base_dir / "data" / "reddit.duckdb")).expanduser().resolve()

    chroma_dir = Path(os.environ.get("CHROMA_DIR", base_dir / "data" / "chroma")).expanduser().resolve()
    collection_name = os.environ.get("CHROMA_COLLECTION", "reddit_high_engagement")
    batch_size = int(os.environ.get("EMBED_BATCH_SIZE", "200"))

    # Select an embedding model
    model_name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.get_or_create_collection(collection_name)

    # Read unembedded rows
    with duckdb.connect(str(db_path), read_only=True) as con:
        rows = con.execute("""
            SELECT
              doc_id, source_type, post_id, comment_id, subreddit, created_utc, score, num_comments, url, title, text
            FROM embedding_queue
            WHERE embedded = FALSE
            ORDER BY created_utc DESC
            LIMIT ?
        """, [batch_size]).fetchall()

    if not rows:
        print("[ok] nothing to embed.")
        return 0

    ids, docs, metas = [], [], []
    for (doc_id, source_type, post_id, comment_id, subreddit, created_utc,
         score, num_comments, url, title, text) in rows:
        ids.append(doc_id)
        docs.append(text)
        # create metadata for each doc to embed
        metas.append({
            "source_type"  : source_type,
            "post_id"      : post_id or "",
            "comment_id"   : comment_id or "",
            "subreddit"    : subreddit or "",
            "created_utc"  : str(created_utc) if created_utc is not None else "",
            "score"        : int(score) if score is not None else 0,
            "num_comments" : int(num_comments) if num_comments is not None else 0,
            "url"          : url or "",
            "title"        : title or "",
        })

    embs = model.encode(docs, normalize_embeddings=True).tolist()

    col.upsert(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

    # Mark embedded in db table
    with duckdb.connect(str(db_path)) as conw:
        conw.executemany(
            "UPDATE embedding_queue SET embedded=TRUE, embedded_at=NOW() WHERE doc_id=?",
            [(i,) for i in ids]
        )

    print(f"[ok] embedded {len(ids)} docs into Chroma collection '{collection_name}' at {chroma_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())