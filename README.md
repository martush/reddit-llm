# reddit-llm

A Streamlit dashboard that scrapes stock-related Reddit discussions, extracts ticker mentions, and lets you query the data through a locally running LLM.

https://martina.kibik.org/

## What it does

- Scrapes posts and comments from r/wallstreetbets, r/stocks, r/investing, r/StockMarket, r/options, and r/ValueInvesting using the Reddit API (PRAW)
- Extracts ticker mentions from comments and post titles using context-aware rules (dollar-prefix, all-caps, trading context words) and validates them against a reference list of all US-listed securities
- Tags each mention as bullish, bearish, or neutral based on keyword signals
- Embeds high-engagement posts and comments into a ChromaDB vector store using sentence-transformers
- Serves everything through a Streamlit app with three tabs:
  - **Overview** — top tickers by score-weighted mentions, top posts, and a market data summary (via yfinance)
  - **Ticker drill-down** — price chart, volume, key metrics, Reddit sentiment counts, and the posts/comments that drove the mention
  - **Ask AI** — RAG interface backed by ChromaDB and a locally running Ollama model (llama3.2:3b by default)

## Stack

| Component | Technology |
|---|---|
| Data store | DuckDB |
| Vector store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Ollama (local) |
| Reddit API | PRAW |
| Market data | yfinance |
| Dashboard | Streamlit + Plotly |

## Setup

**1. Clone and create the conda environment**
```bash
git clone <repo>
cd reddit-llm
conda create -n reddit-llm python=3.11
conda activate reddit-llm
pip install -r requirements.txt
# Note: torch is a large download (~2GB) - needed for sentence-transformers
```

**2. Copy and fill in the environment file**
```bash
cp .env.example .env
```
Edit `.env` with your Reddit API credentials and base directory path. The Reddit API credentials are free.

**3. Initialise the database**
```bash
python scripts/initialize_duckdb.py
```

**4. Populate the ticker reference list**
```bash
python scripts/refresh_tickers.py
```
This downloads all US-listed tickers from Nasdaq's public symbol directory (covers NASDAQ, NYSE, NYSE ARCA, BATS, and IEX). Re-run weekly to pick up new listings.

**5. Start Ollama and pull a model**
```bash
ollama pull llama3.2:3b
ollama serve
```

**6. Run the Streamlit app**
```bash
cd streamlit
streamlit run app_streamlit.py
```

## Daily pipeline

`run_daily.sh` runs the full scrape-to-embed pipeline. Set it up as a cron job:

```
# Example: run every day at 08:00
0 8 * * *  /home/martina/Desktop/Git/reddit-llm/run_daily.sh
```

The four steps it runs:

| Step | Script | What it does |
|---|---|---|
| 1 | `reddit_scraper.py` | Fetches posts and comments from configured subreddits |
| 2 | `reddit_postprocess.py` | Extracts ticker mentions and sentiment from new comments |
| 3 | `build_embedding_queue.py` | Selects high-engagement posts/comments for embedding |
| 4 | `embed_to_chroma.py` | Embeds queued items into ChromaDB |

Logs are written to `logs/daily_YYYY-MM-DD.log`.

## Project structure

```
scripts/
  reddit_scraper.py          # Reddit API scraper (PRAW)
  reddit_postprocess.py      # Ticker extraction and sentiment tagging
  build_embedding_queue.py   # Selects content for embedding
  embed_to_chroma.py         # Embeds into ChromaDB
  refresh_tickers.py         # Updates the ticker reference list from Nasdaq
  initialize_duckdb.py       # Creates the DuckDB schema
  ask_llama.py               # CLI interface to the RAG pipeline
  search_chroma.py           # CLI search against ChromaDB

streamlit/
  app_streamlit.py           # Main dashboard
  finance_utils.py           # yfinance helpers

data/
  reddit.duckdb              # Main database
  chroma/                    # ChromaDB vector store

logs/                        # Daily pipeline logs
```
