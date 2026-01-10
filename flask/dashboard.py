#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv, find_dotenv

import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.express as px


# ----------------------------
# Config
# ----------------------------

load_dotenv(find_dotenv(usecwd=False))

BASE_DIR = Path(os.environ["BASE_DIR"]).expanduser().resolve()
DB_PATH = Path(os.environ.get("DB_PATH", BASE_DIR / "data" / "reddit.duckdb")).expanduser().resolve()


def read_query_df(sql: str, params=None, retries: int = 5, delay_s: float = 0.4) -> pd.DataFrame:
    """
    Read-only query helper. Opens/closes connection per call to avoid long-lived locks.
    Adds small retry loop in case writer holds the lock briefly.
    """
    last_err = None
    for _ in range(retries):
        try:
            con = duckdb.connect(str(DB_PATH), read_only=True)
            try:
                if params is None:
                    return con.execute(sql).fetchdf()
                return con.execute(sql, params).fetchdf()
            finally:
                con.close()
        except Exception as e:
            last_err = e
            time.sleep(delay_s)
    raise last_err


# ----------------------------
# Queries
# ----------------------------

def q_top_tickers(hours: int, limit: int) -> str:
    return f"""
    SELECT
      ct.ticker,
      SUM(c.score) AS score_weighted,
      COUNT(DISTINCT ct.comment_id) AS unique_comments,
      SUM(ct.direction='bullish') AS bullish,
      SUM(ct.direction='bearish') AS bearish
    FROM comment_tickers ct
    JOIN comments c ON c.comment_id = ct.comment_id
    WHERE c.created_utc >= NOW() - INTERVAL '{hours} hours'
      AND (LENGTH(ct.ticker) >= 2 OR ct.method='dollar')
    GROUP BY ct.ticker
    HAVING unique_comments >= 5
    ORDER BY score_weighted DESC
    LIMIT {limit};
    """


def q_top_posts(hours: int, limit: int) -> str:
    return f"""
    SELECT
      pt.ticker,
      COUNT(*) AS post_mentions,
      SUM(p.num_comments) AS total_post_comments
    FROM post_tickers pt
    JOIN posts p ON p.post_id = pt.post_id
    WHERE p.created_utc >= NOW() - INTERVAL '{hours} hours'
      AND (LENGTH(pt.ticker) >= 2 OR pt.method='dollar')
    GROUP BY pt.ticker
    ORDER BY post_mentions DESC
    LIMIT {limit};
    """


# ----------------------------
# Dash app (Flask underneath)
# ----------------------------

app = dash.Dash(__name__)
server = app.server  # Flask server to deploy vs gunicorn

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "20px auto", "fontFamily": "system-ui"},
    children=[
        html.H2("Reddit Stock Monitor (DuckDB + Dash)"),

        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "center"},
            children=[
                html.Div([
                    html.Label("Lookback (hours)"),
                    dcc.Slider(id="hours", min=6, max=168, step=6, value=24,
                               marks={6:"6",24:"24",48:"48",72:"72",168:"168"})
                ], style={"flex": 2}),
                html.Div([
                    html.Label("Limit"),
                    dcc.Input(id="limit", type="number", value=20, min=5, max=100, step=5)
                ], style={"flex": 1}),
                html.Button("Refresh", id="refresh", n_clicks=0, style={"height": "40px", "marginTop":"22px"})
            ]
        ),

        html.H3("Top tickers by comment signal (score-weighted)"),
        dcc.Graph(id="bar_comments"),
        dash_table.DataTable(
            id="table_comments",
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px"},
            style_header={"fontWeight": "bold"},
        ),

        html.H3("Top tickers by post mentions"),
        dcc.Graph(id="bar_posts"),
        dash_table.DataTable(
            id="table_posts",
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px"},
            style_header={"fontWeight": "bold"},
        ),

        html.Div(id="status", style={"marginTop": "12px", "opacity": 0.8})
    ]
)


@app.callback(
    Output("bar_comments", "figure"),
    Output("table_comments", "data"),
    Output("table_comments", "columns"),
    Output("bar_posts", "figure"),
    Output("table_posts", "data"),
    Output("table_posts", "columns"),
    Output("status", "children"),
    Input("refresh", "n_clicks"),
    Input("hours", "value"),
    Input("limit", "value"),
)
def refresh_view(_n, hours, limit):
    try:
        df_comments = read_query_df(q_top_tickers(int(hours), int(limit)))
        df_posts = read_query_df(q_top_posts(int(hours), int(limit)))

        # Charts
        fig_comments = px.bar(df_comments, x="ticker", y="score_weighted")
        fig_posts = px.bar(df_posts, x="ticker", y="post_mentions")

        # Tables
        comments_cols = [{"name": c, "id": c} for c in df_comments.columns]
        posts_cols = [{"name": c, "id": c} for c in df_posts.columns]

        status = f"OK. DB: {DB_PATH}"
        return (
            fig_comments,
            df_comments.to_dict("records"),
            comments_cols,
            fig_posts,
            df_posts.to_dict("records"),
            posts_cols,
            status
        )
    except Exception as e:
        # Show empty visuals if DB is temporarily locked by writer
        empty = pd.DataFrame()
        fig_empty = px.bar(empty)
        return fig_empty, [], [], fig_empty, [], [], f"Error: {e}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
