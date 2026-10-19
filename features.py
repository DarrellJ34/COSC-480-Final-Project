"""
Daily per-stock features for the LSTM sequence.

Every feature on day t uses only prices through day t (rolling windows and backward shifts only).
Each feature is then z-scored PER STOCK over a trailing 252-day window (min 126 days) and clipped
to [-5, 5]. That is leak-free, makes a volatile stock and a calm one comparable, and means no
fold-dependent scaling is needed later.

Output: one row per (date, ticker) for stocks only, columns = FEATURE_COLS. NaN during warm-up.
"""

import numpy as np
import pandas as pd

ZSCORE_WINDOW = 252
MIN_PERIODS = 126

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_21", "ret_63",      # returns over several lookbacks
    "vol_21", "vol_ratio", "hl_range_5",       # realized volatility, short/long vol, intraday range
    "volume_rel",                              # volume vs its 21-day average (log)
    "ma50_dist", "ma200_dist", "rsi_14",       # trend / mean-reversion
    "dd_63",                                   # drawdown from the 63-day high
    "rel_ret_21", "sector_ret_21",             # stock vs market, sector ETF momentum
    "spy_ret_1", "spy_ret_21", "vix", "vix_chg_5",   # market context (same for every stock)
]


def _ret(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x / x.shift(n) - 1


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / down)


def _broadcast(series: pd.Series, like: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({c: series for c in like.columns}, index=like.index)


def build_daily_features(prices: pd.DataFrame, tickers_df: pd.DataFrame) -> pd.DataFrame:
    close  = prices.pivot(index="date", columns="ticker", values="Close").sort_index()
    high   = prices.pivot(index="date", columns="ticker", values="High").reindex_like(close)
    low    = prices.pivot(index="date", columns="ticker", values="Low").reindex_like(close)
    volume = prices.pivot(index="date", columns="ticker", values="Volume").reindex_like(close)

    sector_etf = dict(zip(tickers_df["ticker"], tickers_df["sector_etf"]))
    stock_cols = [t for t in close.columns if t in sector_etf]
    spy, vix = close["SPY"], close["^VIX"]
    ret1 = _ret(close, 1)

    f = {}
    f["ret_1"], f["ret_5"], f["ret_21"], f["ret_63"] = ret1, _ret(close, 5), _ret(close, 21), _ret(close, 63)
    f["vol_21"] = ret1.rolling(21).std()
    f["vol_ratio"] = ret1.rolling(5).std() / ret1.rolling(63).std()
    f["hl_range_5"] = (high / low - 1).rolling(5).mean()
    f["volume_rel"] = np.log((volume + 1) / (volume.rolling(21).mean() + 1))
    f["ma50_dist"] = close / close.rolling(50).mean() - 1
    f["ma200_dist"] = close / close.rolling(200).mean() - 1
    f["rsi_14"] = _rsi(close)
    f["dd_63"] = close / close.rolling(63).max() - 1
    spy_ret21 = _ret(spy, 21)
    f["rel_ret_21"] = f["ret_21"].sub(spy_ret21, axis=0)
    etf_ret21 = _ret(close, 21)
    # sector ETF momentum; fall back to SPY where the ETF doesn't exist yet (XLRE 2015+, XLC 2018+)
    f["sector_ret_21"] = pd.DataFrame(
        {t: etf_ret21[sector_etf[t]].combine_first(spy_ret21) if sector_etf[t] in etf_ret21 else spy_ret21
         for t in stock_cols}, index=close.index)
    f["spy_ret_1"] = _broadcast(_ret(spy, 1), close)
    f["spy_ret_21"] = _broadcast(spy_ret21, close)
    f["vix"] = _broadcast(vix, close)
    f["vix_chg_5"] = _broadcast(_ret(vix, 5), close)

    # trailing z-score per column, then to long format for stock tickers only
    has_price = close[stock_cols].stack(future_stack=True).notna().to_numpy()
    out, index = {}, None
    for name in FEATURE_COLS:
        x = f[name][stock_cols].replace([np.inf, -np.inf], np.nan)
        mu = x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).mean()
        sd = x.rolling(ZSCORE_WINDOW, min_periods=MIN_PERIODS).std()
        z = ((x - mu) / sd).clip(-5, 5).stack(future_stack=True)
        index = z.index if index is None else index
        out[name] = z.to_numpy(dtype=np.float32)
    long = pd.DataFrame(out, index=index)
    long.index.names = ["date", "ticker"]
    long = long[has_price].reset_index()
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)


def check_no_lookahead(prices, tickers_df, cutoff, n_tickers=40):
    """Rebuild features from prices truncated at `cutoff`; rows before the cutoff must be
    identical to the full build. Returns the max absolute difference (0.0 = no lookahead)."""
    keep = sorted(set(tickers_df["ticker"].head(n_tickers)) | {"SPY", "^VIX"} | set(tickers_df["sector_etf"]))
    sub = prices[prices["ticker"].isin(keep)]
    full = build_daily_features(sub, tickers_df).set_index(["date", "ticker"]).sort_index()
    trunc = build_daily_features(sub[sub["date"] <= cutoff], tickers_df).set_index(["date", "ticker"]).sort_index()
    full = full.loc[full.index.get_level_values("date") <= cutoff]
    assert full.index.equals(trunc.index), "row sets differ"
    diff = (full - trunc).abs()
    nan_mismatch = (full.isna() != trunc.isna()).to_numpy().sum()
    return float(np.nanmax(diff.to_numpy())), int(nan_mismatch)


if __name__ == "__main__":
    import time
    prices = pd.read_parquet("data/raw_prices.parquet")
    prices["date"] = prices["date"].astype("datetime64[ns]")
    tickers_df = pd.read_csv("data/tickers.csv")
    t0 = time.time()
    feats = build_daily_features(prices, tickers_df)
    feats.to_parquet("data/features_daily.parquet", index=False)
    print(f"built {len(feats):,} rows x {len(FEATURE_COLS)} features in {time.time() - t0:.0f}s "
          f"-> data/features_daily.parquet")
    print("share of NaN per feature (warm-up):")
    print(feats[FEATURE_COLS].isna().mean().round(3).to_string())
    maxdiff, nan_mm = check_no_lookahead(prices, tickers_df, cutoff=pd.Timestamp("2015-06-30"))
    print(f"\nlookahead check: max |full - truncated| = {maxdiff:.2e}, NaN mismatches = {nan_mm}")
