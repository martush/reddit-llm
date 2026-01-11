#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

import chromadb
from sentence_transformers import SentenceTransformer


def main() -> int:
    load_dotenv(find_dotenv(usecwd=False))
    base_dir = Path(os.environ["BASE_DIR"]).expanduser().resolve()

    chroma_dir = Path(os.environ.get("CHROMA_DIR", base_dir / "data" / "chroma")).expanduser().resolve()
    collection_name = os.environ.get("CHROMA_COLLECTION", "reddit_high_engagement")
    model_name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("Usage: python scripts/search_chroma.py <your query here>")
        return 2

    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.get_collection(collection_name)

    q_emb = model.encode([query], normalize_embeddings=True).tolist()[0]
    res = col.query(query_embeddings=[q_emb], n_results=5)

    for i in range(len(res["ids"][0])):
        doc_id = res["ids"][0][i]
        meta = res["metadatas"][0][i]
        doc = res["documents"][0][i]
        print("\n---")
        print("id:", doc_id)
        print("subreddit:", meta.get("subreddit"))
        print("score:", meta.get("score"), "num_comments:", meta.get("num_comments"))
        print("url:", meta.get("url"))
        print("text:", (doc[:400] + "…") if len(doc) > 400 else doc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
