import os
import time
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv, find_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
import requests
import logging

from finance_utils import get_multiple_tickers_summary, get_ticker_data, format_market_cap, format_volume, get_ticker_info

#Find env file
load_dotenv(find_dotenv(usecwd=False))

# Set up basic paths
BASE_DIR = Path(os.environ["BASE_DIR"]).expanduser().resolve()
DB_PATH = Path(os.environ.get("DB_PATH", BASE_DIR / "data" / "reddit.duckdb")).expanduser().resolve()


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler(f'{BASE_DIR}/streamlit/streamlit_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)



def read_df(sql, params=None):
    '''
    Function which reads an SQL statement and returns a df
    '''

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        if params is None:
            return con.execute(sql).fetchdf()
        return con.execute(sql, params).fetchdf()


@st.cache_data(ttl=60)  # cache for 60s
def get_top_tickers(hours, limit):
    q = f"""
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
    LEFT JOIN tickers t ON t.ticker = ct.ticker
    WHERE c.created_utc >= NOW() - INTERVAL '{hours} hours'
      AND (LENGTH(ct.ticker) >= 2 OR ct.method='dollar')
    GROUP BY ct.ticker, t.name
    HAVING threads >= 2 AND unique_comments >= 5
    ORDER BY score_weighted DESC
    LIMIT {limit};
    """
    logger.debug('Query get_top_tickers')
    logger.debug(q)
    return read_df(q)


@st.cache_data(ttl=60)
def get_top_posts(hours, limit):
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
    logger.debug('Query get_top_posts')
    logger.debug(q)
    if not df.empty:
        df["reddit_url"] = df["reddit_url"].astype(str)
    return df


@st.cache_data(ttl=120)
def get_recent_tickers(hours):
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
    logger.debug('Query get_recent_tickers')
    logger.debug(q)
    return read_df(q)



# Cache the embedding model - loads once and reuses
@st.cache_resource
def load_embedder(model_name):
    return SentenceTransformer(model_name)

# Cache the ChromaDB client
@st.cache_resource
def get_chroma_client(chroma_dir):
    return chromadb.PersistentClient(path=str(chroma_dir))


def stream_ollama_response(ollama_host, llm_model, prompt):
    """Stream response from Ollama"""
    r = requests.post(
        f"{ollama_host}/api/generate",
        json={
            "model": llm_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 500,
                "temperature": 0.7,
            }
        },
        stream=True,
        timeout=180,
    )
    r.raise_for_status()
    logger.debug('Iterating over Ollama response')
    for line in r.iter_lines():
        if line:
            import json
            chunk = json.loads(line)
            if "response" in chunk:
                yield chunk["response"]



def main():
    st.set_page_config(page_title="Reddit Stock Monitor", layout="wide")
    st.title("Reddit Stock Monitor")

    with st.sidebar:
        hours = st.slider("Lookback (hours)", min_value=6, max_value=168, step=6, value=24)
        limit = st.number_input("Limit", min_value=5, max_value=100, step=5, value=20)
        #st.caption(f"DB: {DB_PATH}")
        fin_data_period = st.selectbox("Price and volume period", ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'))
        logger.debug(f'Fin data selection {fin_data_period}')

    # Increase font size of tabs
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Set up tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Ticker drill-down", "Ask AI"])


    ################# Tab 1 ##########################
    with tab1:
        st.subheader("Top tickers (comments, score-weighted)")
        df_t = get_top_tickers(hours, limit)
        if df_t.empty:
            st.info("No ticker data yet. Run scraper + postprocess.")
        else:
            df_t_graph = df_t.copy()
            df_t_graph["label"] = df_t.apply(
                lambda r: f'{r["ticker"]} — {r["company_name"]}' if pd.notna(r["company_name"]) else r["ticker"],
                axis=1
            )
            fig = px.bar(df_t_graph, x="label", y="score_weighted")
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


        # Fetch financial data
        top_tickers = df_t['ticker'].to_list()
        with st.spinner("Fetching market data"):
            ticker_summary = get_multiple_tickers_summary(top_tickers, period=fin_data_period)

        if not ticker_summary.empty:
            # Display as a styled dataframe
            st.dataframe(
                ticker_summary.style.format({
                    'Price'             : '${:.2f}',
                    'Change'            : '${:.2f}',
                    'Change %'          : '{:.2f}%',
                    'Volume'            : lambda x: format_volume(x),
                    'Volume for period' : lambda x: format_volume(x),
                    'Avg Volume'        : lambda x: format_volume(x),
                    'Market Cap'        : lambda x: format_market_cap(x),
                }).background_gradient(subset=['Change %'], cmap='RdYlGn', vmin=-5, vmax=5),
                use_container_width=True,
                hide_index=True
            )

    ################# Tab 2 ##########################
    with tab2:
        st.subheader("Drill-down")
        # Hardcode to 1 week
        #recent = get_recent_tickers(hours)
        recent = get_recent_tickers(168)
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
            df_tp["reddit_url"] = df_tp["reddit_url"].astype(str)

            # st.dataframe(df_tp)
            # for _, r in df_tp.head(10).iterrows():
            #     st.markdown(f"- [{r['title']}]({r['reddit_url']})")

            df_tp = df_tp.rename(columns={"reddit_url": "link"})
            st.data_editor(
                df_tp,
                column_config={
                    "link": st.column_config.LinkColumn("reddit link"),
                },
                disabled=True,
                width="stretch",
            )    

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


        # Ticker Drilldown Section
        if ticker:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"{ticker} Price Chart")
                
                # Time period selector
                period = st.radio(
                    "Time Period:",
                    options=['1mo', '3mo', '6mo', '1y'],
                    horizontal=True,
                    index=0
                )
                
                # Get price data
                ticker_data = get_ticker_data(ticker, period=period)
                
                if ticker_data is not None and not ticker_data.empty:
                    # Create candlestick chart
                    fig = go.Figure(data=[go.Candlestick(
                        x=ticker_data.index,
                        open=ticker_data['Open'],
                        high=ticker_data['High'],
                        low=ticker_data['Low'],
                        close=ticker_data['Close'],
                        name='Price'
                    )])
                    
                    fig.update_layout(
                        title=f'{ticker} Price Movement',
                        yaxis_title='Price ($)',
                        xaxis_title='Date',
                        height=400,
                        hovermode='x unified',
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    fig_volume = go.Figure(data=[go.Bar(
                        x=ticker_data.index,
                        y=ticker_data['Volume'],
                        name='Volume',
                        marker_color='lightblue'
                    )])
                    
                    fig_volume.update_layout(
                        title=f'{ticker} Trading Volume',
                        yaxis_title='Volume',
                        xaxis_title='Date',
                        height=300,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_volume, use_container_width=True)
        
            with col2:
                st.subheader("Key Metrics")
                
                info = get_ticker_info(ticker)
                
                # Display metrics
                st.metric(
                    label="Current Price",
                    value=f"${info['current_price']:.2f}",
                    delta=f"{(info['current_price'] - info['previous_close']):.2f} ({((info['current_price'] - info['previous_close']) / info['previous_close'] * 100):.2f}%) vs previous close"
                )
                
                st.metric(
                    label="Volume",
                    value=format_volume(info['volume']),
                    delta=f"{((info['volume'] - info['avg_volume']) / info['avg_volume'] * 100):.1f}% vs avg" if info['avg_volume'] > 0 else None
                )
                
                st.metric(
                    label="Market Cap",
                    value=format_market_cap(info['market_cap'])
                )
                
                st.metric(
                    label="Avg Volume (3mo)",
                    value=format_volume(info['avg_volume'])
                )
                
                # Show Reddit sentiment alongside
                st.subheader("Reddit Sentiment")
                # Add your sentiment data here from DuckDB
                # sentiment = get_sentiment_for_ticker(selected_ticker)
                st.write("Bullish mentions: X")
                st.write("Bearish mentions: Y")

    ################# Tab 3 ##########################
    with tab3:
        st.header("Ask AI about Reddit Sentiment")
        
        # Load environment variables
        base_dir = Path(os.environ["BASE_DIR"]).expanduser().resolve()
        chroma_dir = Path(os.environ.get("CHROMA_DIR", base_dir / "data" / "chroma")).expanduser().resolve()
        collection_name = os.environ.get("CHROMA_COLLECTION", "reddit_high_engagement")
        embed_model_name = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        #llm_model = os.environ.get("LLM_MODEL", "llama3.1:8b")
        llm_model = os.environ.get("LLM_MODEL", "llama3.2:3b")
    
        # Input
        question = st.text_input(
            "Ask a question about Reddit sentiment:",
            placeholder="e.g., What are people saying about TSLA this week?"
        )
        
        # Add options for faster responses
        col1, col2 = st.columns([3, 1])
        with col2:
            num_results = st.selectbox("\\# of sources", [3, 5, 8], index=0)

        if st.button("Ask", type="primary"):
            if not question:
                st.warning("Please enter a question")
            else:
                with st.spinner("Searching and generating answer... (Marty accepting donations for a new pc)"):
                    try:
                        # Use the cached embedder
                        embedder = load_embedder(embed_model_name)
                        q_emb = embedder.encode([question], normalize_embeddings=True).tolist()[0]
                        
                        # Get Chroma client to retrieve relevant # results (depending on user choice)
                        client = get_chroma_client(chroma_dir)
                        col = client.get_collection(collection_name)
                        res = col.query(
                            query_embeddings = [q_emb], 
                            n_results        = num_results, 
                            include          = ["documents", "metadatas"]
                        )
                        docs  = res["documents"][0]
                        metas = res["metadatas"][0]
                        
                        # Build context with citations (limit doc length)
                        context_blocks = []
                        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
                            url = meta.get("url", "")
                            subreddit = meta.get("subreddit", "")
                            score = meta.get("score", 0)
                            title = meta.get("title", "")
                            header = f"[{i}] subreddit={subreddit} score={score} url={url}"
                            if title:
                                header += f" title={title}"
                            # Limit doc length to reduce token count
                            truncated_doc = doc[:1000] if len(doc) > 1000 else doc
                            context_blocks.append(header + "\n" + truncated_doc)
                            
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
                        
                        # Display streaming response
                        st.subheader("Answer:")
                        response_placeholder = st.empty()
                        full_response = ""

                        try:
                            for token in stream_ollama_response(ollama_host, llm_model, prompt):
                                full_response += token
                                response_placeholder.write(full_response)
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                        
                        # Display sources
                        st.subheader("Sources:")
                        for i, meta in enumerate(metas, start=1):
                            url = meta.get("url", "")
                            title = meta.get("title", "")
                            subreddit = meta.get("subreddit", "")
                            score = meta.get("score", 0)
                            
                            with st.expander(f"[{i}] {title[:100]}... (score: {score})"):
                                st.write(f"**Subreddit:** r/{subreddit}")
                                st.write(f"**Score:** {score}")
                                st.write(f"**URL:** {url}")
                                st.write(f"**Content:** {docs[i-1][:500]}...")
                    
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. Try asking a simpler question or use fewer sources.")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
