import streamlit as st
import requests
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm

API_TOKEN=st.secrets["key"]

# Constants
BASE_URL_HISTORY = "https://api.tradier.com/v1/markets/history"
BASE_URL_QUOTE = "https://api.tradier.com/v1/markets/quotes"
BASE_URL_OPTIONS = "https://api.tradier.com/v1/markets/options/chains"
BASE_URL_EXPIRATIONS = "https://api.tradier.com/v1/markets/options/expirations"
BASE_URL_EARNINGS = "https://api.tradier.com/beta/markets/fundamentals/calendars"
HEADERS = {"Authorization": API_TOKEN, "Accept": "application/json"}

# Helper functions
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def newton_raphson_iv(market_price, S, K, T, r, initial_guess=0.2, tol=1e-5, max_iter=100):
    sigma = initial_guess
    for _ in range(max_iter):
        price = black_scholes_call(S, K, T, r, sigma)
        vega = S * norm.pdf((np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        price_diff = price - market_price
        if abs(price_diff) < tol:
            return sigma * 100
        sigma -= price_diff / vega
    return np.nan

def fetch_historical_data(symbol, start_date, end_date):
    params = {"symbol": symbol, "interval": "daily", "start": start_date, "end": end_date}
    response = requests.get(BASE_URL_HISTORY, headers=HEADERS, params=params)
    if response.status_code == 200:
        data = response.json().get("history", {}).get("day", [])
        if data:
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df['close'] = pd.to_numeric(df['close'])
            return df
    return pd.DataFrame()

def calculate_annualized_hv(symbol):
    today = datetime.date.today()
    start_date = (today - datetime.timedelta(days=3*365)).isoformat()
    end_date = today.isoformat()
    df = fetch_historical_data(symbol, start_date, end_date)
    if df.empty or len(df) < 252:
        return None
    returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    rolling_hv = [returns.iloc[i-252:i].std() * np.sqrt(252) * 100 for i in range(252, len(returns))]
    return np.mean(rolling_hv) if rolling_hv else None

def fetch_latest_price(symbol):
    params = {"symbols": symbol}
    response = requests.get(BASE_URL_QUOTE, headers=HEADERS, params=params)
    if response.status_code == 200:
        quotes = response.json().get("quotes", {}).get("quote", {})
        if isinstance(quotes, dict):
            return float(quotes["last"])
    return None

def fetch_expirations(symbol):
    response = requests.get(BASE_URL_EXPIRATIONS, headers=HEADERS, params={"symbol": symbol})
    if response.status_code == 200:
        return response.json().get("expirations", {}).get("date", [])
    return []

def fetch_option_chain(symbol, expiration, spot):
    response = requests.get(BASE_URL_OPTIONS, headers=HEADERS, params={"symbol": symbol, "expiration": expiration, "greeks": "true"})
    if response.status_code == 200:
        options = response.json().get("options", {}).get("option", [])
        calls = [opt for opt in options if opt['option_type'] == 'call']
        if calls:
            df = pd.DataFrame(calls)
            df['strike'] = pd.to_numeric(df['strike'])
            df['bid'] = pd.to_numeric(df['bid'])
            df['ask'] = pd.to_numeric(df['ask'])
            df['mid'] = (df['bid'] + df['ask']) / 2
            df['diff'] = abs(df['strike'] - spot)
            atm_strike = df.loc[df['diff'].idxmin()]['strike']
            return df, atm_strike
    return pd.DataFrame(), None

def fetch_next_earnings(symbol):
    today = datetime.date.today()
    response = requests.get(BASE_URL_EARNINGS, params={"symbols": symbol}, headers=HEADERS)
    if response.status_code != 200:
        return None, None

    try:
        data = response.json()
        earnings_events = []

        for result_index in [0, 1]:
            try:
                corporate_calendars = data[0]["results"][result_index]["tables"]["corporate_calendars"]
                if corporate_calendars and isinstance(corporate_calendars, list):
                    for event in corporate_calendars:
                        event_description = event.get('event', '').lower()
                        if 'earnings' in event_description:
                            begin_date_str = event.get('begin_date_time')
                            if begin_date_str:
                                try:
                                    begin_date = datetime.datetime.strptime(begin_date_str, "%Y-%m-%d").date()
                                    if begin_date >= today:
                                        earnings_events.append(begin_date)
                                except ValueError:
                                    print(f"Invalid date format: {begin_date_str}")
            except (IndexError, KeyError, TypeError):
                continue

        if earnings_events:
            next_earnings = min(earnings_events)
            days_left = (next_earnings - today).days
            return next_earnings, days_left
        else:
            return None, None

    except Exception as e:
        print(f"Error parsing earnings data: {e}")
        return None, None

# Main app
def main():
    st.markdown("<h1 style='text-align: center;'>📊 Options IV Analyzer & Volatility Comparison</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        stock = st.text_input("Enter Stock Ticker:", "AAPL").upper()
    with col2:
        index = st.selectbox("Select Index:", ["SPY","XLK","XLF","XLY","XLP","XLE","XLI","XLV","XLU","XLRE","XLB","XLC","QQQ","IWM","DIA","VTI","TLT","SMH","XBI","GDX","OIH","IYR"])

    today = datetime.date.today()
    r = 0.01

    # STOCK ANALYSIS
    st.markdown(f"<h2 style='text-align: center;'>{stock} Analysis</h2>", unsafe_allow_html=True)
    stock_price = fetch_latest_price(stock)
    if stock_price:
        st.write(f"💰 **Price:** ${stock_price:.2f}")
    avg_hv_stock = calculate_annualized_hv(stock)
    if avg_hv_stock:
        st.write(f"📈 **Annualized HV:** {avg_hv_stock:.2f}%")
    earnings_date, days_left = fetch_next_earnings(stock)
    if earnings_date:
        st.write(f"📅 **Next Earnings:** {earnings_date} ({days_left} days left)")

    stock_expirations = fetch_expirations(stock)
    if stock_expirations:
        selected_exp_stock = st.selectbox("Select Stock Expiration:", stock_expirations)
        calls_df, atm_strike = fetch_option_chain(stock, selected_exp_stock, stock_price)
        if not calls_df.empty:
            selected_strike = st.selectbox("Select Stock Strike:", sorted(calls_df['strike'].unique()), index=sorted(calls_df['strike'].unique()).index(atm_strike))
            selected_call = calls_df[calls_df['strike'] == selected_strike].iloc[0]
            st.write(f"**Ask Price:** ${selected_call['ask']:.2f}")
            ask_iv_stock = float(selected_call['greeks']['ask_iv']) * 100 if 'greeks' in selected_call and 'ask_iv' in selected_call['greeks'] else np.nan
            st.write(f"**Ask IV:** {ask_iv_stock:.2f}%" if not np.isnan(ask_iv_stock) else "Ask IV unavailable.")
            T_stock = (datetime.datetime.strptime(selected_exp_stock, "%Y-%m-%d").date() - today).days / 365
            bs_iv_stock = newton_raphson_iv(selected_call['ask'], stock_price, selected_strike, T_stock, r)
            mid_iv_stock = newton_raphson_iv(selected_call['mid'], stock_price, selected_strike, T_stock, r)
            st.write(f"**Black-Scholes IV:** {bs_iv_stock:.2f}%")
            st.write(f"**Black-Scholes Midpoint IV:** {mid_iv_stock:.2f}%")

    # INDEX ANALYSIS
    st.markdown(f"<h2 style='text-align: center;'>{index} Analysis</h2>", unsafe_allow_html=True)
    index_price = fetch_latest_price(index)
    if index_price:
        st.write(f"💰 **Price:** ${index_price:.2f}")
    avg_hv_index = calculate_annualized_hv(index)
    if avg_hv_index:
        st.write(f"📈 **Annualized HV:** {avg_hv_index:.2f}%")

    index_expirations = fetch_expirations(index)
    if index_expirations:
        selected_exp_index = st.selectbox("Select Index Expiration:", index_expirations)
        calls_df_idx, atm_strike_idx = fetch_option_chain(index, selected_exp_index, index_price)
        if not calls_df_idx.empty:
            selected_strike_idx = st.selectbox("Select Index Strike:", sorted(calls_df_idx['strike'].unique()), index=sorted(calls_df_idx['strike'].unique()).index(atm_strike_idx))
            selected_call_idx = calls_df_idx[calls_df_idx['strike'] == selected_strike_idx].iloc[0]
            st.write(f"**Ask Price:** ${selected_call_idx['ask']:.2f}")
            ask_iv_index = float(selected_call_idx['greeks']['ask_iv']) * 100 if 'greeks' in selected_call_idx and 'ask_iv' in selected_call_idx['greeks'] else np.nan
            st.write(f"**Ask IV:** {ask_iv_index:.2f}%" if not np.isnan(ask_iv_index) else "Ask IV unavailable.")
            T_index = (datetime.datetime.strptime(selected_exp_index, "%Y-%m-%d").date() - today).days / 365
            bs_iv_index = newton_raphson_iv(selected_call_idx['ask'], index_price, selected_strike_idx, T_index, r)
            mid_iv_index = newton_raphson_iv(selected_call_idx['mid'], index_price, selected_strike_idx, T_index, r)
            st.write(f"**Black-Scholes IV:** {bs_iv_index:.2f}%")
            st.write(f"**Black-Scholes Midpoint IV:** {mid_iv_index:.2f}%")

    # COMPARISON SUMMARY
    st.markdown("<h2 style='text-align: center;'>Comparison Summary</h2>", unsafe_allow_html=True)
    if avg_hv_stock and not np.isnan(ask_iv_stock):
        diff_stock = ask_iv_stock - avg_hv_stock
        if diff_stock > 2:
            st.write(f"📌 **Stock Option Valuation:** Overvalued by {diff_stock:.2f} points.")
        elif diff_stock < -2:
            st.write(f"📌 **Stock Option Valuation:** Undervalued by {abs(diff_stock):.2f} points.")
        else:
            st.write("📌 **Stock Valuation:** Fairly valued.")
    if avg_hv_index and not np.isnan(ask_iv_index):
        diff_index = ask_iv_index - avg_hv_index
        if diff_index > 2:
            st.write(f"📌 **Index Option Valuation:** Overvalued by {diff_index:.2f} points.")
        elif diff_index < -2:
            st.write(f"📌 **Index Option Valuation:** Undervalued by {abs(diff_index):.2f} points.")
        else:
            st.write("📌 **Index Option Valuation:** Fairly valued.")

    # HISTOGRAM
    st.markdown("<h2 style='text-align: center;'>Daily Returns Histogram</h2>", unsafe_allow_html=True)
    start_date = (today - datetime.timedelta(days=3*365)).isoformat()
    stock_df = fetch_historical_data(stock, start_date, today.isoformat())
    index_df = fetch_historical_data(index, start_date, today.isoformat())
    if not stock_df.empty and not index_df.empty:
        stock_returns = stock_df['close'].pct_change().dropna() * 100
        index_returns = index_df['close'].pct_change().dropna() * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.linspace(-10, 10, 100)
        ax.hist(stock_returns, bins=bins, alpha=0.5, label=f"{stock} Daily Change", color='goldenrod')
        ax.hist(index_returns, bins=bins, alpha=0.5, label=f"{index} Daily Change", color='turquoise')
        ax.set_xlabel("Daily % Change")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.set_title(f"Daily % Change: {stock} vs {index}")
        st.pyplot(fig)
    else:
        st.warning("Not enough data to plot histograms.")

if __name__ == "__main__":
    main()
