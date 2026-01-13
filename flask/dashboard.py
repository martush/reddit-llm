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

# Optional: point dashboard at a snapshot DB to avoid contention
# SNAPSHOT_PATH = BASE_DIR / "data" / "reddit_snapshot.duckdb"
# DB_PATH = SNAPSHOT_PATH if SNAPSHOT_PATH.exists() else DB_PATH


def read_query_df(sql: str, params=None, retries: int = 5, delay_s: float = 0.3) -> pd.DataFrame:
    """
    Read-only query helper. Opens/closes connection per call to avoid long-lived locks.
    Retries briefly if the writer has the DB lock for a moment.
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
      t.name AS company_name,
      SUM(c.score) AS score_weighted,
      COUNT(DISTINCT ct.comment_id) AS unique_comments,
      COUNT(DISTINCT c.post_id) AS threads,
      SUM(ct.direction='bullish') AS bullish,
      SUM(ct.direction='bearish') AS bearish
    FROM comment_tickers ct
    JOIN comments c ON c.comment_id = ct.comment_id
    LEFT JOIN tickers t
      ON t.ticker = ct.ticker
    WHERE c.created_utc >= NOW() - INTERVAL '{hours} hours'
      AND (LENGTH(ct.ticker) >= 2 OR ct.method='dollar')
    GROUP BY ct.ticker, t.name
    HAVING threads >= 2 AND unique_comments >= 5
    ORDER BY score_weighted DESC
    LIMIT {limit};
    """



def q_top_posts(hours: int, limit: int) -> str:
    # Top posts by comment count, includes clickable reddit URL if present
    # If url is NULL, we’ll build a fallback link using post_id.
    return f"""
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


def q_ticker_dropdown(hours: int) -> str:
    return f"""
    WITH recent AS (
      SELECT DISTINCT ct.ticker
      FROM comment_tickers ct
      JOIN comments c ON c.comment_id = ct.comment_id
      WHERE c.created_utc >= NOW() - INTERVAL '{hours} hours'
        AND (LENGTH(ct.ticker) >= 2 OR ct.method='dollar')
    )
    SELECT
      r.ticker,
      t.name AS company_name
    FROM recent r
    LEFT JOIN tickers t
      ON t.ticker = r.ticker
    ORDER BY r.ticker;
    """



def q_posts_for_ticker(hours: int) -> str:
    return f"""
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
      AND p.created_utc >= NOW() - INTERVAL '{hours} hours'
    ORDER BY p.num_comments DESC
    LIMIT 30;
    """


def q_comments_for_ticker(hours: int) -> str:
    return f"""
    SELECT
      c.subreddit,
      c.post_id,
      SUBSTR(c.body, 1, 300) AS snippet,
      c.score,
      c.created_utc
    FROM comment_tickers ct
    JOIN comments c ON c.comment_id = ct.comment_id
    WHERE ct.ticker = ?
      AND c.created_utc >= NOW() - INTERVAL '{hours} hours'
    ORDER BY c.score DESC
    LIMIT 50;
    """


# ----------------------------
# Dash app
# ----------------------------

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


def make_markdown_link(url: str, text: str) -> str:
    # Dash DataTable supports Markdown if presentation="markdown"
    return f"[{text}]({url})" if url else text

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "20px auto", "fontFamily": "system-ui"},
    children=[
        html.H2("Reddit Stock Monitor"),

        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "center"},
            children=[
                html.Div([
                    html.Label("Lookback (hours)"),
                    dcc.Slider(
                        id="hours",
                        min=6, max=168, step=6, value=24,
                        marks={6:"6",24:"24",48:"48",72:"72",168:"168"}
                    )
                ], style={"flex": 2}),
                html.Div([
                    html.Label("Limit"),
                    dcc.Input(id="limit", type="number", value=20, min=5, max=100, step=5)
                ], style={"flex": 1}),
                html.Button("Refresh", id="refresh", n_clicks=0, style={"height": "40px", "marginTop": "22px"}),
            ]
        ),

        dcc.Tabs(
            id="tabs",
            value="tab-overview",
            children=[
                dcc.Tab(label="Overview", value="tab-overview"),
                dcc.Tab(label="Ticker drill-down", value="tab-drilldown"),
            ]
        ),

        html.Div(id="tab-content", style={"marginTop": "16px"})
    ]
)


def overview_layout():
    return html.Div(children=[
        html.H3("Top tickers (comments, score-weighted)"),
        dcc.Graph(id="bar_comments"),
        dash_table.DataTable(
            id="table_comments",
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px"},
            style_header={"fontWeight": "bold"},
        ),

        html.H3("Top posts (by comment count)"),
        dcc.Graph(id="bar_posts"),
        dash_table.DataTable(
            id="table_posts",
            page_size=20,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px"},
            style_header={"fontWeight": "bold"},
            markdown_options={"link_target": "_blank"},
        ),
    ])


def drilldown_layout():
    return html.Div(children=[
        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "end"},
            children=[
                html.Div(style={"flex": 1}, children=[
                    html.Label("Ticker"),
                    dcc.Dropdown(id="ticker", placeholder="Select a ticker…")
                ]),
                html.Div(style={"width": "220px"}, children=[
                    html.Label("Drill-down window (hours)"),
                    dcc.Input(id="drill_hours", type="number", value=72, min=6, max=720, step=6)
                ])
            ]
        ),

        html.H3("Posts mentioning this ticker"),
        dash_table.DataTable(
            id="table_ticker_posts",
            page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px"},
            style_header={"fontWeight": "bold"},
            markdown_options={"link_target": "_blank"},
        ),

        html.H3("Top comments mentioning this ticker"),
        dash_table.DataTable(
            id="table_ticker_comments",
            page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px"},
            style_header={"fontWeight": "bold"},
        ),
    ])


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-drilldown":
        return drilldown_layout()
    return overview_layout()


# ----------------------------
# Overview callbacks
# ----------------------------
@app.callback(
    Output("bar_comments", "figure"),
    Output("table_comments", "data"),
    Output("table_comments", "columns"),
    Output("bar_posts", "figure"),
    Output("table_posts", "data"),
    Output("table_posts", "columns"),
    Input("refresh", "n_clicks"),
    Input("hours", "value"),
    Input("limit", "value"),
    Input("tabs", "value"),
)
def refresh_overview(_n, hours, limit, tab):
    # Only refresh when on overview tab
    if tab != "tab-overview":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    try:
        hours = int(hours)
        limit = int(limit)

        df_comments = read_query_df(q_top_tickers(hours, limit))
        df_posts = read_query_df(q_top_posts(hours, limit))

        #fig_comments = px.bar(df_comments, x="ticker", y="score_weighted") if not df_comments.empty else px.bar(pd.DataFrame())
        x_col = "ticker"
        if "company_name" in df_comments.columns and df_comments["company_name"].notna().any():
            df_comments = df_comments.copy()
            df_comments["label"] = df_comments.apply(
                lambda r: f'{r["ticker"]} — {r["company_name"]}' if pd.notna(r["company_name"]) else r["ticker"],
                axis=1
            )
            x_col = "label"

        if df_comments.empty:
            fig_comments = px.bar(pd.DataFrame({"ticker": [], "score_weighted": []}), x="ticker", y="score_weighted")
        else:
            fig_comments = px.bar(df_comments, x=x_col, y="score_weighted")


        fig_posts = px.bar(df_posts, x="title", y="num_comments") if not df_posts.empty else px.bar(pd.DataFrame())
        fig_posts.update_layout(xaxis_title="post", yaxis_title="num_comments")

        comments_cols = [{"name": c, "id": c} for c in df_comments.columns]

        # Make post title clickable
        if not df_posts.empty:
            df_posts = df_posts.copy()
            df_posts["title"] = df_posts.apply(lambda r: make_markdown_link(r["reddit_url"], r["title"]), axis=1)

        posts_cols = []
        for c in df_posts.columns:
            if c == "title":
                posts_cols.append({"name": c, "id": c, "presentation": "markdown"})
            else:
                posts_cols.append({"name": c, "id": c})

        return (
            fig_comments,
            df_comments.to_dict("records"),
            comments_cols,
            fig_posts,
            df_posts.to_dict("records"),
            posts_cols
        )

    except Exception as e:
        print("refresh_overview error:", repr(e))
        empty = pd.DataFrame()
        fig_empty = px.bar(empty)
        return fig_empty, [], [], fig_empty, [], []



# ----------------------------
# Drilldown callbacks
# ----------------------------
@app.callback(
    Output("ticker", "options"),
    Input("tabs", "value"),
    Input("hours", "value"),
    Input("refresh", "n_clicks"),
)
def populate_ticker_options(tab, hours, _n):
    if tab != "tab-drilldown":
        return []
    try:
        hours = int(hours)
        df = read_query_df(q_ticker_dropdown(hours))
        #return [{"label": t, "value": t} for t in df["ticker"].tolist()]
        opts = []
        for _, row in df.iterrows():
            label = row["ticker"]
            if row.get("company_name"):
                label = f'{row["ticker"]} — {row["company_name"]}'
            opts.append({"label": label, "value": row["ticker"]})
        return opts
    except Exception as e:
        print("populate_ticker_options error:", repr(e))
        return []



@app.callback(
    Output("table_ticker_posts", "data"),
    Output("table_ticker_posts", "columns"),
    Output("table_ticker_comments", "data"),
    Output("table_ticker_comments", "columns"),
    Input("ticker", "value"),
    Input("drill_hours", "value"),
    Input("tabs", "value"),
)
def drilldown(ticker, drill_hours, tab):
    if tab != "tab-drilldown" or not ticker:
        return [], [], [], []

    drill_hours = int(drill_hours)

    # Posts for ticker
    df_posts = read_query_df(q_posts_for_ticker(drill_hours), [ticker])

    if not df_posts.empty:
        df_posts = df_posts.copy()
        df_posts["title"] = df_posts.apply(lambda r: make_markdown_link(r["reddit_url"], r["title"]), axis=1)

    posts_cols = []
    for c in df_posts.columns:
        if c == "title":
            posts_cols.append({"name": c, "id": c, "presentation": "markdown"})
        else:
            posts_cols.append({"name": c, "id": c})

    # Comments for ticker
    df_comments = read_query_df(q_comments_for_ticker(drill_hours), [ticker])
    comments_cols = [{"name": c, "id": c} for c in df_comments.columns]

    return (
        df_posts.to_dict("records"), posts_cols,
        df_comments.to_dict("records"), comments_cols
    )


if __name__ == "__main__":
    print("Starting Dash app on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
