#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

import chromadb
from sentence_transformers import SentenceTransformer

# Ollama python client is optional; you can also call HTTP directly.
import requests


def main() -> int:
    load_dotenv(find_dotenv(usecwd=False))
    base_dir = Path(os.environ["BASE_DIR"]).expanduser().resolve()

    chroma_dir = Path(os.environ.get("CHROMA_DIR", base_dir / "data" / "chroma")).expanduser().resolve()
    collection_name = os.environ.get("CHROMA_COLLECTION", "reddit_high_engagement")
    embed_model_name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    llm_model = os.environ.get("LLM_MODEL", "llama3.1:8b")

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("Usage: python scripts/ask.py <your question>")
        return 2

    # Embed query
    embedder = SentenceTransformer(embed_model_name)
    q_emb = embedder.encode([question], normalize_embeddings=True).tolist()[0]

    # Retrieve docs
    client = chromadb.PersistentClient(path=str(chroma_dir))
    col = client.get_collection(collection_name)

    res = col.query(query_embeddings=[q_emb], n_results=8, include=["documents", "metadatas"])
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    # Build context with citations
    context_blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        url = meta.get("url", "")
        subreddit = meta.get("subreddit", "")
        score = meta.get("score", 0)
        title = meta.get("title", "")
        header = f"[{i}] subreddit={subreddit} score={score} url={url}"
        if title:
            header += f" title={title}"
        context_blocks.append(header + "\n" + doc)

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are summarizing and answering questions using Reddit content.
Use ONLY the context below. If you are unsure, say so.
When you make a claim, cite sources like [1], [2] based on the context items.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER (with citations):
"""

    # Call Ollama generate endpoint
    r = requests.post(
        f"{ollama_host}/api/generate",
        json={
            "model": llm_model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()

    print(data.get("response", "").strip())
    print("\nSources:")
    for i, meta in enumerate(metas, start=1):
        url = meta.get("url", "")
        title = meta.get("title", "")
        print(f"[{i}] {title} {url}".strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
