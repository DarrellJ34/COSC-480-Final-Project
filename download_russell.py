"""
Download the Russell 1000 universe used for the market-breadth parts of the fear/greed index.

Run once:  python download_russell.py
Outputs (in data/):
  russell1000.csv          ticker -> company name, sector (from the iShares IWB holdings file)
  prices_russell.parquet   daily OHLCV for every Russell 1000 stock since 2005

The S&P 500 (data/raw_prices.parquet) stays the universe the MODEL predicts on. The Russell 1000 is
only used as a wider proxy for "the whole market" when counting 52-week highs/lows and up/down volume.
Both lists are today's members, so both carry survivorship bias; for a market-wide ratio that is then
z-scored against its own history, that bias is mild (see fear_greed.py).
"""

import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

START_DATE = "2005-01-01"
DATA_DIR = Path("data")
IWB_URL = "https://www.ishares.com/us/products/239707/ishares-russell-1000-etf/latest-holdings.csv"


def get_russell_table() -> pd.DataFrame:
    """Current Russell 1000 members from the iShares IWB holdings CSV (a few header lines, then a table)."""
    text = requests.get(IWB_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).text
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Ticker,"))
    as_of = next((line.split(",", 1)[1].strip('"') for line in lines[:start] if line.startswith("Fund Holdings as of")), "?")
    body = "\n".join(line for line in lines[start:] if line.strip())
    df = pd.read_csv(io.StringIO(body), on_bad_lines="skip")
    df = df[df["Asset Class"] == "Equity"].rename(columns={"Ticker": "ticker", "Name": "name", "Sector": "sector"})
    # iShares writes share classes as "BRK B"; Yahoo wants "BRK-B". Drop rights/when-issued lines.
    df["ticker"] = df["ticker"].str.strip().str.replace(" ", "-", regex=False)
    df = df[~df["ticker"].str.contains("-R-W|-W$", regex=True)]
    df = df[df["ticker"].str.fullmatch(r"[A-Z]+(-[A-Z])?")]
    print(f"  holdings as of {as_of}: {len(df)} equities")
    return df[["ticker", "name", "sector"]].drop_duplicates("ticker").reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("Fetching Russell 1000 holdings from iShares...")
    r1000 = get_russell_table()
    r1000.to_csv(DATA_DIR / "russell1000.csv", index=False)
    print(f"  saved {DATA_DIR / 'russell1000.csv'}")

    tickers = r1000["ticker"].tolist()
    print(f"Downloading daily prices for {len(tickers)} symbols since {START_DATE}...")
    print("  (this takes several minutes)")
    prices = yf.download(tickers, start=START_DATE, auto_adjust=True, group_by="ticker", threads=True, progress=True)
    prices = prices.stack(level=0, future_stack=True).rename_axis(["date", "ticker"]).reset_index()
    prices = prices.dropna(subset=["Close"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    out = DATA_DIR / "prices_russell.parquet"
    prices.to_parquet(out, index=False)
    print(f"Saved {len(prices):,} rows for {prices['ticker'].nunique()} symbols to {out}")
    print(f"Date range: {prices['date'].min().date()} to {prices['date'].max().date()}")
    missing = set(tickers) - set(prices["ticker"].unique())
    if missing:
        print(f"Warning: no data returned for {len(missing)} tickers: {sorted(missing)[:20]}{' ...' if len(missing) > 20 else ''}")


if __name__ == "__main__":
    main()
