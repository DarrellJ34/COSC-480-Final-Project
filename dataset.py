"""
Windows, static vectors, and the walk-forward split.

WindowDataset slices each sample's 120-day window on demand from one array per ticker,
so we never materialize the full [N, 120, F] tensor (~1 GB).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class WindowDataset(Dataset):
    def __init__(self, samples, features, feature_cols, static, labels, seq_len=120):
        """
        samples:      DataFrame with columns ticker, month_end (one row per sample)
        features:     DataFrame with date, ticker, feature_cols (daily rows, from features.py)
        static:       float32 array [N, S]   (N = len(samples); S may be 0)
        labels:       array [N] of 0/1
        """
        self.seq_len = seq_len
        self.feature_cols = list(feature_cols)
        self.arrays, self.dates, self.valid_start = {}, {}, {}
        for tkr, g in features.groupby("ticker", sort=False):
            g = g.sort_values("date")
            arr = g[self.feature_cols].to_numpy(dtype=np.float32)
            finite = np.isfinite(arr).all(axis=1)
            self.valid_start[tkr] = int(np.argmax(finite)) if finite.any() else len(arr)   # end of warm-up
            self.arrays[tkr] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            self.dates[tkr] = g["date"].to_numpy(dtype="datetime64[ns]")

        # map each sample to (ticker, index of its month_end row); keep only full, warmed-up windows
        tick = samples["ticker"].to_numpy()
        ends_dt = samples["month_end"].to_numpy(dtype="datetime64[ns]")
        keep, ends = [], []
        for i in range(len(samples)):
            d = self.dates.get(tick[i])
            if d is None:
                continue
            j = int(np.searchsorted(d, ends_dt[i], side="right")) - 1
            if j < 0 or d[j] != ends_dt[i] or j - seq_len + 1 < self.valid_start[tick[i]]:
                continue
            keep.append(i)
            ends.append(j)
        self.sample_idx = np.asarray(keep, dtype=np.int64)       # row numbers into `samples`
        self.ends = np.asarray(ends, dtype=np.int64)
        self.tickers = tick[self.sample_idx]
        self.month_end = samples["month_end"].to_numpy()[self.sample_idx]
        self.static = torch.from_numpy(np.asarray(static, dtype=np.float32)[self.sample_idx])
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32)[self.sample_idx])

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, i):
        end = self.ends[i]
        x = self.arrays[self.tickers[i]][end - self.seq_len + 1 : end + 1]    # [seq_len, F]
        return torch.from_numpy(x), self.static[i], self.labels[i]

    def last_day_matrix(self):
        """[N, F] of each sample's final-day features (for flat baselines like logistic regression)."""
        return np.stack([self.arrays[t][e] for t, e in zip(self.tickers, self.ends)])


def make_loader(dataset, mask, batch_size=128, shuffle=False):
    idx = np.flatnonzero(mask)
    return DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=shuffle)


def walk_forward_split(month_end, test_year, val_years=1, purge_months=1):
    """Boolean masks (train, val, test). Train < val < test in time, with `purge_months` dropped
    before each boundary so no training/validation label window reaches into the next block."""
    me = pd.to_datetime(pd.Series(np.asarray(month_end)))
    test_start = pd.Timestamp(year=test_year, month=1, day=1)
    test_end = test_start + pd.DateOffset(years=1)
    val_start = test_start - pd.DateOffset(years=val_years)
    purge = pd.DateOffset(months=purge_months)
    train = (me < val_start - purge).to_numpy()
    val = ((me >= val_start) & (me < test_start - purge)).to_numpy()
    test = ((me >= test_start) & (me < test_end)).to_numpy()
    return train, val, test


def sector_one_hot(samples, tickers_df):
    sectors = sorted(tickers_df["sector"].unique())
    sec = samples["ticker"].map(dict(zip(tickers_df["ticker"], tickers_df["sector"])))
    mat = pd.get_dummies(sec).reindex(columns=sectors, fill_value=0).to_numpy(dtype=np.float32)
    return mat, [f"sector_{s}" for s in sectors]


MACRO_COLS = [
    # --- interest rates and policy expectations (all FRED) ---
    # NOTE: the fed funds futures front month on Yahoo (ZQ=F) is the CURRENT month's contract, so its
    # implied rate minus the effective rate is ~0 bp every year and carries no expectations. Checked
    # 2026-10-19. Hike/cut expectations therefore come from the Treasury curve instead.
    "ust2_minus_ff",     # 2-year yield minus fed funds: the expected policy path over the next 2 years
    "d_ust2_21",         # 21-day change in the 2-year yield: the classic "policy expectations shock"
    "bill6_minus_3",     # 6-month minus 3-month bill: expected rate change over the next 3-6 months
    "curve_10y_3m",      # 10-year minus 3-month: growth/recession expectations, bank margins
    "d_ust10_21",        # 21-day change in the 10-year yield: duration shock to growth stocks, bond proxies
    # --- risk appetite and commodities (Yahoo) ---
    "credit_21",         # 21-day return of HYG minus TLT: credit risk appetite
    "vix_chg_21",        # 21-day change in VIX
    "dxy_chg_21",        # 21-day change in the dollar index
    "oil_chg_21",        # 21-day change in WTI oil
]


def macro_static(samples, macro):
    """Month-end macro summaries (spreads and changes, not levels), each z-scored over a trailing
    252-trading-day window, taken as of each sample's month_end. macro = data/macro.parquet.
    Levels like the curve slope become "relative to the past year" after the z-score, which keeps
    them comparable across the zero-rate years and the 5% years."""
    m = macro.set_index("date").sort_index()
    m = m[m["vix"].notna()].ffill()                     # trading days only
    f = pd.DataFrame(index=m.index)
    f["ust2_minus_ff"] = m["ust_2y"] - m["fed_funds_effective"]
    f["d_ust2_21"] = m["ust_2y"].diff(21)
    f["bill6_minus_3"] = m["tbill_6m"] - m["tbill_3m"]
    f["curve_10y_3m"] = m["ust_10y"] - m["tbill_3m"]
    f["d_ust10_21"] = m["ust_10y"].diff(21)
    f["credit_21"] = (m["hyg"] / m["hyg"].shift(21) - 1) - (m["tlt"] / m["tlt"].shift(21) - 1)
    f["vix_chg_21"] = m["vix"] / m["vix"].shift(21) - 1
    f["dxy_chg_21"] = m["dxy"] / m["dxy"].shift(21) - 1
    f["oil_chg_21"] = m["wti_oil"] / m["wti_oil"].shift(21) - 1
    f = f[MACRO_COLS]
    z = ((f - f.rolling(252, min_periods=126).mean()) / f.rolling(252, min_periods=126).std()).clip(-5, 5)
    idx = z.index.searchsorted(samples["month_end"].to_numpy(dtype="datetime64[ns]"), side="right") - 1
    arr = z.to_numpy(dtype=np.float32)[idx]
    return np.nan_to_num(arr), MACRO_COLS
