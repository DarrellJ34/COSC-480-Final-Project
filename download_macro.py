"""
Download macro / rates data for the stock direction model.

Run once:  python download_macro.py
Requires:  FRED_API_KEY in .env  (free key: https://fred.stlouisfed.org/docs/api/api_key.html)
           If the key is missing, the FRED part is skipped and only the Yahoo series are saved.

Output (in data/):
  macro.parquet   one row per calendar date, one column per series, raw values.
                  NaN where a series has no observation that day (weekends, holidays,
                  or before the series starts). Forward-fill is left to the feature code.
"""

import os
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

START_DATE = "2005-01-01"
DATA_DIR = Path("data")
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED series id -> column name. All are market-priced, daily, and never revised,
# so they are safe to use as of the date shown (no release lag).
FRED_SERIES = {
    "DFF":          "fed_funds_effective",   # effective federal funds rate
    "DGS1":         "ust_1y",                # Treasury constant-maturity yields
    "DGS2":         "ust_2y",
    "DGS10":        "ust_10y",
    "DTB3":         "tbill_3m",              # Treasury bill discount rates
    "DTB6":         "tbill_6m",
    "T10Y2Y":       "curve_10y_2y",          # yield-curve slopes
    "T10Y3M":       "curve_10y_3m",
    "BAMLH0A0HYM2": "hy_spread",             # ICE BofA US high-yield option-adjusted spread (FRED only serves ~3 years)
    "BAMLC0A0CM":   "ig_spread",             # ICE BofA US investment-grade spread (same limit)
    "DBAA":         "moody_baa",             # Moody's Baa corporate yield, daily since 1986: full history, unlike the ICE series
    "DAAA":         "moody_aaa",             # Moody's Aaa corporate yield; Baa - Aaa = credit stress for the fear/greed index
    "T10YIE":       "breakeven_10y",         # 10y breakeven inflation
    "DFII10":       "real_10y",              # 10y TIPS real yield
    "DTWEXBGS":     "dollar_broad",          # trade-weighted broad dollar index (2006+)
}

# Yahoo ticker -> column name (daily closes).
YAHOO_SERIES = {
    "^IRX":     "yh_tbill_13w",
    "^FVX":     "yh_ust_5y",
    "^TNX":     "yh_ust_10y",
    "^TYX":     "yh_ust_30y",
    "ZQ=F":     "fedfunds_future_front",     # 30-day fed funds futures, front month
    "^VIX":     "vix",
    "DX-Y.NYB": "dxy",                       # US dollar index
    "CL=F":     "wti_oil",
    "GC=F":     "gold",
    "HYG":      "hyg",                       # high-yield bond ETF (2007+)
    "LQD":      "lqd",                       # investment-grade bond ETF
    "TLT":      "tlt",                       # 20y+ Treasury ETF
}


def fetch_fred(series_id: str, api_key: str) -> pd.Series:
    """One FRED series as a date-indexed float Series. Missing values ('.') become NaN."""
    r = requests.get(
        FRED_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": START_DATE,
        },
        timeout=60,
    )
    r.raise_for_status()
    obs = r.json()["observations"]
    s = pd.Series(
        {pd.Timestamp(o["date"]): pd.to_numeric(o["value"], errors="coerce") for o in obs},
        dtype="float64",
    )
    s.index.name = "date"
    return s


def fetch_all_fred(api_key: str) -> pd.DataFrame:
    frames = {}
    for sid, col in FRED_SERIES.items():
        try:
            frames[col] = fetch_fred(sid, api_key)
            print(f"  FRED {sid:13s} -> {col}: {frames[col].notna().sum()} obs")
        except Exception as e:  # keep going if one series fails
            print(f"  FRED {sid:13s} FAILED: {type(e).__name__}: {str(e)[:100]}")
    return pd.DataFrame(frames)


def fetch_all_yahoo() -> pd.DataFrame:
    raw = yf.download(
        list(YAHOO_SERIES),
        start=START_DATE,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    out = {}
    for tkr, col in YAHOO_SERIES.items():
        try:
            out[col] = raw[tkr]["Close"]
            print(f"  Yahoo {tkr:9s} -> {col}: {out[col].notna().sum()} obs")
        except KeyError:
            print(f"  Yahoo {tkr:9s} FAILED: no data returned")
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    # Fed funds futures price -> implied expected fed funds rate for the contract month
    if "fedfunds_future_front" in df:
        df["fedfunds_implied"] = 100.0 - df["fedfunds_future_front"]
    return df


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    load_dotenv(Path(__file__).resolve().parent / ".env")
    api_key = os.environ.get("FRED_API_KEY", "").strip()

    parts = []

    if api_key:
        print("Downloading FRED series...")
        parts.append(fetch_all_fred(api_key))
    else:
        print("FRED_API_KEY not found in .env -- skipping FRED series.")

    print("Downloading Yahoo macro series...")
    parts.append(fetch_all_yahoo())

    macro = pd.concat(parts, axis=1).sort_index()
    macro = macro.dropna(how="all")

    out = DATA_DIR / "macro.parquet"
    macro.reset_index().to_parquet(out, index=False)

    print(f"\nSaved {macro.shape[0]:,} dates x {macro.shape[1]} series to {out}")
    print(f"Date range: {macro.index.min().date()} to {macro.index.max().date()}")
    first_valid = macro.apply(lambda s: s.first_valid_index())
    late = first_valid[first_valid > pd.Timestamp(START_DATE) + pd.Timedelta(days=10)]
    if len(late):
        print("Series that start after", START_DATE + ":")
        for col, d in late.items():
            print(f"  {col}: {d.date()}")


if __name__ == "__main__":
    main()
