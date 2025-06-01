📊 Options IV Analyzer & Volatility Comparison
This app leverages the brokrage API to analyze options volatility for any stock or index ETF (e.g., SPY, QQQ, XLK). It helps traders and analysts make more informed decisions by comparing implied volatility (IV) with historical volatility (HV).

Features:
🔍 Stock & Index Selection: Choose a stock ticker and compare it to an index (SPY, QQQ, XLK, etc.).

📈 Annualized HV: Calculates rolling annualized historical volatility using a 3-year lookback.

⚡ Ask IV, Black-Scholes IV, and Midpoint IV: Automatically fetches option chain data, selecting the closest ATM call option, and computes key IV metrics.

📅 Earnings Date: Retrieves and displays the next upcoming earnings date for the selected stock.

🔗 Comparison Summary: Highlights whether the option appears overvalued, undervalued, or fairly valued compared to historical volatility.

📊 Daily Returns Histogram: Visualizes the daily percentage changes for both the stock and index.

Built With:
🐍 Python

🌐 Streamlit

📈 Matplotlib

🔌 brokrage API

