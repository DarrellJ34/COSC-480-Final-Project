"""
A homemade Fear & Greed index, rebuilt from its ingredients so it covers 2005-today with no lookahead.

CNN's index averages seven market signals. Six are built here from data already in data/:

  component      CNN's recipe                                   built from
  momentum       S&P 500 vs its 125-day moving average          SPY closes
  volatility     VIX vs its 50-day moving average               VIX
  safe_haven     20-day stock return minus 20-day bond return    SPY and TLT
  junk           junk vs investment-grade yield spread           Moody's Baa - Aaa (FRED DBAA, DAAA)
  strength       stocks at 52-week highs vs 52-week lows         Russell 1000 closes
  breadth        McClellan volume summation (up vs down volume)  Russell 1000 closes and volume

The seventh, the put/call ratio, needs CBOE data that is not downloaded (yet).

Every component is signed so that HIGHER = GREED, then z-scored over a trailing 252-day window
(min 126) and clipped to [-5, 5], the same leak-free scaling features.py and dataset.py use.
The composite is the mean of the six z-scores. Only past data enters any value, so a value dated t
is safe to use as of day t.

  python fear_greed.py          # builds data/fear_greed_daily.parquet and prints the "be greedy when
                                # others are fearful" check: next-month up-rate by fear/greed bucket
"""

import numpy as np
import pandas as pd

ZSCORE_WINDOW, MIN_PERIODS = 252, 126
COMPONENTS = ["fg_momentum", "fg_volatility", "fg_safe_haven", "fg_junk", "fg_strength", "fg_breadth"]
FG_COLS = COMPONENTS + ["fg_composite"]          # what the model receives, in this order


def _z(x: pd.DataFrame) -> pd.DataFrame:
    mu = x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).mean()
    sd = x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).std()
    return ((x - mu) / sd).clip(-5, 5)


def market_breadth(universe_prices: pd.DataFrame) -> pd.DataFrame:
    """strength and breadth components from a wide universe of stocks (one row per date, ticker)."""
    close = universe_prices.pivot(index="date", columns="ticker", values="Close").sort_index()
    volume = universe_prices.pivot(index="date", columns="ticker", values="Volume").reindex_like(close)

    # strength: share of stocks closing at a 52-week high minus share at a 52-week low (10-day mean)
    hi = close.rolling(252, min_periods=126).max()
    lo = close.rolling(252, min_periods=126).min()
    valid = hi.notna()
    net_new_highs = ((close >= hi) & valid).sum(axis=1) - ((close <= lo) & valid).sum(axis=1)
    strength = (net_new_highs / valid.sum(axis=1).replace(0, np.nan)).rolling(10).mean()

    # breadth: McClellan volume summation index on the ratio-adjusted net advancing volume
    ret = close.pct_change()
    adv = volume.where(ret > 0).sum(axis=1)
    dec = volume.where(ret < 0).sum(axis=1)
    net = (adv - dec) / (adv + dec).replace(0, np.nan)
    oscillator = net.ewm(span=19).mean() - net.ewm(span=39).mean()
    summation = oscillator.cumsum()

    return pd.DataFrame({"n_stocks": valid.sum(axis=1), "strength": strength, "breadth": summation})


def build_fear_greed(prices: pd.DataFrame, macro: pd.DataFrame, universe_prices: pd.DataFrame) -> pd.DataFrame:
    """Daily fear/greed table: raw components, z-scored components (FG_COLS) and n_stocks.
    prices = data/raw_prices.parquet (for SPY, ^VIX); macro = data/macro.parquet; universe_prices = Russell 1000."""
    close = prices.pivot(index="date", columns="ticker", values="Close").sort_index()
    spy, vix = close["SPY"], close["^VIX"]
    m = macro.set_index("date").sort_index().reindex(close.index).ffill()

    raw = pd.DataFrame(index=close.index)
    raw["momentum"] = spy / spy.rolling(125).mean() - 1
    raw["volatility"] = -(vix / vix.rolling(50).mean() - 1)                 # calm = greed
    raw["safe_haven"] = (spy / spy.shift(20) - 1) - (m["tlt"] / m["tlt"].shift(20) - 1)
    if {"moody_baa", "moody_aaa"} <= set(m.columns):
        raw["junk"] = -(m["moody_baa"] - m["moody_aaa"])                   # narrow spread = greed
    else:                                                                   # fallback: junk vs IG bond ETFs
        raw["junk"] = (m["hyg"] / m["hyg"].shift(20) - 1) - (m["lqd"] / m["lqd"].shift(20) - 1)
    breadth = market_breadth(universe_prices).reindex(close.index)
    raw["strength"], raw["breadth"] = breadth["strength"], breadth["breadth"]

    z = _z(raw.replace([np.inf, -np.inf], np.nan))
    z.columns = COMPONENTS
    out = pd.concat([raw.add_prefix("raw_"), z], axis=1)
    out["fg_composite"] = z.mean(axis=1, skipna=True).where(z.notna().sum(axis=1) >= 4)
    out["n_stocks"] = breadth["n_stocks"]
    out.index.name = "date"
    return out.reset_index()


def fg_static(samples: pd.DataFrame, fg_daily: pd.DataFrame):
    """[N, 7] float32 of FG_COLS as of each sample's month_end (last value on or before that date)."""
    f = fg_daily.set_index("date").sort_index()[FG_COLS]
    idx = f.index.searchsorted(samples["month_end"].to_numpy(dtype="datetime64[ns]"), side="right") - 1
    arr = f.to_numpy(dtype=np.float32)[idx]
    return np.nan_to_num(arr), FG_COLS


def bucket_check(fg_daily: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Next-month up-rate by fear/greed bucket at month end: the 'be greedy when others are fearful' test.
    Months are the unit (each month's up-rate is one observation), since every stock shares the same reading."""
    f = fg_daily.set_index("date")["fg_composite"].dropna()
    monthly = samples.groupby("month_end").agg(up_rate=("label", "mean"), fwd_ret=("fwd_ret", "mean"))
    idx = f.index.searchsorted(monthly.index.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    monthly["fg"] = f.to_numpy()[idx]
    monthly = monthly[idx >= 0].dropna()
    edges = [-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf]
    labels = ["extreme fear", "fear", "neutral", "greed", "extreme greed"]
    monthly["bucket"] = pd.cut(monthly["fg"], bins=edges, labels=labels)
    table = monthly.groupby("bucket", observed=True).agg(
        months=("fg", "size"), next_month_up_rate=("up_rate", "mean"), next_month_avg_return=("fwd_ret", "mean"))
    corr = monthly["fg"].corr(monthly["up_rate"])
    return table, corr, monthly


if __name__ == "__main__":
    from pathlib import Path
    from labels import build_samples

    DATA = Path("data")
    prices = pd.read_parquet(DATA / "raw_prices.parquet")
    prices["date"] = prices["date"].astype("datetime64[ns]")
    macro = pd.read_parquet(DATA / "macro.parquet")
    macro["date"] = macro["date"].astype("datetime64[ns]")
    universe_path = DATA / "prices_russell.parquet"
    if universe_path.exists():
        universe = pd.read_parquet(universe_path)
        print(f"breadth universe: Russell 1000 ({universe['ticker'].nunique()} tickers)")
    else:
        universe = prices[~prices["ticker"].isin({"SPY", "^VIX"}) & ~prices["ticker"].str.startswith("XL")]
        print(f"breadth universe: S&P 500 fallback ({universe['ticker'].nunique()} tickers); run download_russell.py for the Russell 1000")
    universe["date"] = universe["date"].astype("datetime64[ns]")

    fg = build_fear_greed(prices, macro, universe)
    fg.to_parquet(DATA / "fear_greed_daily.parquet", index=False)
    ok = fg.dropna(subset=["fg_composite"])
    print(f"built {len(fg):,} days -> data/fear_greed_daily.parquet | composite available {ok['date'].min().date()} -> {ok['date'].max().date()}")
    print("\nfirst valid date per component:")
    for c in COMPONENTS:
        print(f"  {c:14s} {fg.loc[fg[c].notna(), 'date'].min().date()}")

    print("\nsanity check, composite on well-known dates (negative = fear):")
    f = fg.set_index("date")["fg_composite"]
    for label, d in [("Oct 2007 peak", "2007-10-09"), ("Mar 2009 bottom", "2009-03-09"), ("Aug 2011 downgrade", "2011-08-08"),
                     ("Mar 2020 covid low", "2020-03-23"), ("Dec 2021 peak", "2021-12-31"), ("Oct 2022 low", "2022-10-12"), ("latest", str(f.index.max().date()))]:
        i = f.index.searchsorted(pd.Timestamp(d), side="right") - 1
        print(f"  {label:20s} {f.index[i].date()}  {f.iloc[i]:+.2f}")

    samples = build_samples(prices, horizon_months=1)
    table, corr, monthly = bucket_check(fg, samples)
    print("\nnext-month up-rate by fear/greed bucket at month end (unit = month):")
    print((table.assign(next_month_up_rate=lambda t: (t["next_month_up_rate"] * 100).round(1),
                        next_month_avg_return=lambda t: (t["next_month_avg_return"] * 100).round(2))
                .rename(columns={"next_month_up_rate": "up-rate %", "next_month_avg_return": "avg return %"})).to_string())
    print(f"\ncorrelation(fear/greed at month end, next-month up-rate) = {corr:+.2f} over {len(monthly)} months")
    print("negative = contrarian (fear now, up next month); near 0 = no timing signal")
