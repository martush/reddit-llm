import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import logging

logger = logging.getLogger(__name__)

# Cache for 5min
@st.cache_data(ttl=300)
def get_ticker_data(ticker, period="1mo"):
    """
    Get stock data for a ticker
    
    Args:
        ticker : stock symbol (e.g. 'AAPL')
        period : time period ('1d', '5d', '1mo', '3mo', '6mo', '1y')
    
    Returns:
        df  : data for the ticker 
    """
    try:
        logger.debug('Fetching get_ticker_data')
        logger.debug(f'Period {period}')
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Cache for 5min
@st.cache_data(ttl=300)
def get_ticker_info(ticker):
    """Get ticker info (name, market cap, etc.)"""
    try:

        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
            'previous_close': info.get('previousClose', 0),
            'market_cap': info.get('marketCap', 0),
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
        }
    except Exception as e:
        return {
            'name': ticker,
            'current_price': 0,
            'previous_close': 0,
            'market_cap': 0,
            'volume': 0,
            'avg_volume': 0,
        }

@st.cache_data(ttl=300)
def get_multiple_tickers_summary(tickers, period="1mo"):
    """Get summary data for multiple tickers"""
    results = []
    for ticker in tickers:
        info = get_ticker_info(ticker)
        df = get_ticker_data(ticker, period=period)
        
        if df is not None and not df.empty:
            #price_change = info['current_price'] - info['previous_close']
            # Price change for selected period
            price_change = df.loc[df.index.max(), 'Close'] - df.loc[df.index.min(), 'Close']
            price_change_pct = (price_change / df.loc[df.index.min(), 'Close'] * 100) if df.loc[df.index.min(), 'Close'] > 0 else 0
            
            # Volume traded for period
            volume = df['Volume'].sum()

            results.append({
                'Ticker'            : ticker,
                'Name'              : info['name'],
                'Price'             : info['current_price'],
                'Change'            : price_change,
                'Change %'          : price_change_pct,
                'Volume'            : info['volume'],
                'Volume for period' : volume,
                'Avg Volume'        : info['avg_volume'],
                'Market Cap'        : info['market_cap'],
            })
    
    return pd.DataFrame(results)

def format_market_cap(value):
    """Format market cap in B/M"""
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"

def format_volume(value):
    """Format volume in M/K"""
    if value >= 1e6:
        return f"{value/1e6:.2f}M"
    elif value >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:,.0f}"