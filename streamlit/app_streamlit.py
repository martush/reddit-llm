import os
import time
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=False))

BASE_DIR = Path(os.environ["BASE_DIR"]).expanduser().resolve()
DB_PATH = Path(os.environ.get("DB_PATH", BASE_DIR / "data" / "reddit.duckdb")).expanduser().resolve()

# Use after adding the snapshot version
SNAP = BASE_DIR / "data" / "reddit_snapshot.duckdb"
if SNAP.exists():
    DB_PATH = SNAP


def read_df(sql: str, params=None) -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        if params is None:
            return con.execute(sql).fetchdf()
        return con.execute(sql, params).fetchdf()


@st.cache_data(ttl=60)  # cache for 60s
def get_top_tickers(hours: int, limit: int) -> pd.DataFrame:
    q = f"""
    SELECT
      ct.ticker,
      t.name AS company_name,
      SUM(c.score) AS score_weighted,
      COUNT(DISTINCT ct.comment_id) AS unique_comments,
      COUNT(DISTINCT c.post_id) AS threads
    FROM comment_tickers ct
    JOIN comments c ON c.comment_id = ct.comment_id
    LEFT JOIN tickers t ON t.ticker = ct.ticker
    WHERE c.created_utc >= NOW() - INTERVAL '{hours} hours'
      AND (LENGTH(ct.ticker) >= 2 OR ct.method='dollar')
    GROUP BY ct.ticker, t.name
    HAVING threads >= 2 AND unique_comments >= 5
    ORDER BY score_weighted DESC
    LIMIT {limit};
    """
    return read_df(q)


@st.cache_data(ttl=60)
def get_top_posts(hours: int, limit: int) -> pd.DataFrame:
    q = f"""
    SELECT
      p.subreddit,
      p.title,
      COALESCE(p.url, 'https://www.reddit.com/comments/' || p.post_id) AS reddit_url,
      p.num_comments,
      p.score,
      p.created_utc
    FROM posts p
    WHERE p.created_utc >= NOW() - INTERVAL '{hours} hours'
    ORDER BY p.num_comments DESC
    LIMIT {limit};
    """
    df = read_df(q)
    if not df.empty:
        df["reddit_url"] = df["reddit_url"].astype(str)
    return df


@st.cache_data(ttl=120)
def get_recent_tickers(hours: int) -> pd.DataFrame:
    q = f"""
    WITH recent AS (
      SELECT DISTINCT ct.ticker
      FROM comment_tickers ct
      JOIN comments c ON c.comment_id = ct.comment_id
      WHERE c.created_utc >= NOW() - INTERVAL '{hours} hours'
        AND (LENGTH(ct.ticker) >= 2 OR ct.method='dollar')
    )
    SELECT r.ticker, t.name AS company_name
    FROM recent r
    LEFT JOIN tickers t ON t.ticker = r.ticker
    ORDER BY r.ticker;
    """
    return read_df(q)


def main():
    st.set_page_config(page_title="Reddit Stock Monitor", layout="wide")
    st.title("Reddit Stock Monitor")

    with st.sidebar:
        hours = st.slider("Lookback (hours)", min_value=6, max_value=168, step=6, value=24)
        limit = st.number_input("Limit", min_value=5, max_value=100, step=5, value=20)
        st.caption(f"DB: {DB_PATH}")

    tab1, tab2 = st.tabs(["Overview", "Ticker drill-down"])

    with tab1:
        st.subheader("Top tickers (comments, score-weighted)")
        df_t = get_top_tickers(hours, limit)
        if df_t.empty:
            st.info("No ticker data yet. Run scraper + postprocess.")
        else:
            df_t["label"] = df_t.apply(
                lambda r: f'{r["ticker"]} — {r["company_name"]}' if pd.notna(r["company_name"]) else r["ticker"],
                axis=1
            )
            fig = px.bar(df_t, x="label", y="score_weighted")
            st.plotly_chart(fig, width='stretch')
            st.dataframe(df_t)

        st.subheader("Top posts (by comment count)")
        df_p = get_top_posts(hours, limit)
        if df_p.empty:
            st.info("No posts yet.")
        else:
            df_show = df_p.copy()
            df_show = df_show.rename(columns={"reddit_url": "link"})
            st.data_editor(
                df_show,
                column_config={
                    "link": st.column_config.LinkColumn("reddit link"),
                },
                disabled=True,
                width="stretch",
            )

    with tab2:
        st.subheader("Drill-down")
        recent = get_recent_tickers(hours)
        if recent.empty:
            st.info("No tickers found in the selected window.")
            return

        tickers = recent["ticker"].tolist()

        label_map = {}
        for _, r in recent.iterrows():
            name = r.get("company_name")
            if pd.notna(name) and str(name).strip():
                label_map[r["ticker"]] = f"{r['ticker']} — {name}"
            else:
                label_map[r["ticker"]] = r["ticker"]

        ticker = st.selectbox(
            "Ticker",
            tickers,
            format_func=lambda t: label_map.get(t, t),
        )

        drill_hours = st.number_input("Drill-down window (hours)", min_value=6, max_value=720, step=6, value=72)

        # Posts mentioning ticker
        q_posts = f"""
        SELECT
          p.subreddit,
          p.title,
          COALESCE(p.url, 'https://www.reddit.com/comments/' || p.post_id) AS reddit_url,
          p.num_comments,
          p.score,
          p.created_utc
        FROM post_tickers pt
        JOIN posts p ON p.post_id = pt.post_id
        WHERE pt.ticker = ?
          AND p.created_utc >= NOW() - INTERVAL '{int(drill_hours)} hours'
        ORDER BY p.num_comments DESC
        LIMIT 30;
        """
        df_tp = read_df(q_posts, [ticker])
        st.markdown("### Posts mentioning this ticker")
        if df_tp.empty:
            st.write("No posts.")
        else:
            st.dataframe(df_tp)
            for _, r in df_tp.head(10).iterrows():
                st.markdown(f"- [{r['title']}]({r['reddit_url']})")

        # Comments mentioning ticker
        q_com = f"""
        SELECT
          c.subreddit,
          c.post_id,
          SUBSTR(c.body, 1, 300) AS snippet,
          c.score,
          c.created_utc
        FROM comment_tickers ct
        JOIN comments c ON c.comment_id = ct.comment_id
        WHERE ct.ticker = ?
          AND c.created_utc >= NOW() - INTERVAL '{int(drill_hours)} hours'
        ORDER BY c.score DESC
        LIMIT 50;
        """
        df_tc = read_df(q_com, [ticker])
        st.markdown("### Top comments mentioning this ticker")
        st.dataframe(df_tc)

if __name__ == "__main__":
    main()
