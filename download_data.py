"""
Download daily price data for the stock direction model.

Run once:  python download_data.py
Outputs (in data/):
  raw_prices.parquet   daily OHLCV for all S&P 500 stocks + SPY + VIX + sector ETFs
  tickers.csv          ticker -> company name, GICS sector, sector ETF
"""

import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

START_DATE = "2005-01-01"
DATA_DIR = Path("data")

# Market benchmark and volatility index
MARKET_TICKERS = ["SPY", "^VIX"]

# GICS sector -> SPDR sector ETF
SECTOR_ETFS = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def get_sp500_table() -> pd.DataFrame:
    """Pull the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Fetch with requests (bundles its own CA certs) so this works on macOS python.org installs
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    table = table.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    # Yahoo uses "-" where Wikipedia uses "." (e.g. BRK.B -> BRK-B)
    table["ticker"] = table["ticker"].str.replace(".", "-", regex=False)
    table["sector_etf"] = table["sector"].map(SECTOR_ETFS)
    return table[["ticker", "name", "sector", "sector_etf"]]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("Fetching S&P 500 ticker list from Wikipedia...")
    sp500 = get_sp500_table()
    sp500.to_csv(DATA_DIR / "tickers.csv", index=False)
    print(f"  {len(sp500)} tickers saved to {DATA_DIR / 'tickers.csv'}")

    all_tickers = sp500["ticker"].tolist() + MARKET_TICKERS + sorted(set(SECTOR_ETFS.values()))

    print(f"Downloading daily prices for {len(all_tickers)} symbols since {START_DATE}...")
    print("  (this takes a few minutes)")
    prices = yf.download(
        all_tickers,
        start=START_DATE,
        auto_adjust=True,   # adjust for splits and dividends
        group_by="ticker",
        threads=True,
        progress=True,
    )

    # Reshape from wide (ticker, field) columns to long rows: date, ticker, Open, High, Low, Close, Volume
    prices = prices.stack(level=0, future_stack=True).rename_axis(["date", "ticker"]).reset_index()
    prices = prices.dropna(subset=["Close"])  # drop days before a stock existed
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    out = DATA_DIR / "raw_prices.parquet"
    prices.to_parquet(out, index=False)

    print(f"Saved {len(prices):,} rows for {prices['ticker'].nunique()} symbols to {out}")
    print(f"Date range: {prices['date'].min().date()} to {prices['date'].max().date()}")
    missing = set(all_tickers) - set(prices["ticker"].unique())
    if missing:
        print(f"Warning: no data returned for {sorted(missing)}")


if __name__ == "__main__":
    main()
