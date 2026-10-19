"""
Monthly samples and labels.

One row per (ticker, month_end). The label asks: if I buy at this month's last close,
is the AVERAGE close over the next `horizon_months` months higher?

    label        = 1 if next_avg > last_close
    label_vs_spy = 1 if the stock's forward return beats SPY's over the same window

The final month in the price file is treated as incomplete and never used as a label month.
A sample is dropped if the stock is missing any of the following months (no gaps allowed).
"""

import pandas as pd

# tickers in raw_prices.parquet that are inputs, not things we predict
NON_STOCK = {"SPY", "^VIX", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"}


def build_samples(prices: pd.DataFrame, horizon_months: int = 1) -> pd.DataFrame:
    prices = prices.sort_values(["ticker", "date"])
    month = prices["date"].dt.year * 12 + prices["date"].dt.month      # integer month index
    grp = prices.groupby(["ticker", month], sort=True)
    m = pd.DataFrame({
        "month_end":  grp["date"].last(),
        "last_close": grp["Close"].last(),
        "sum_close":  grp["Close"].sum(),
        "n_days":     grp["Close"].count(),
    })
    m.index.names = ["ticker", "month_idx"]
    m = m.reset_index()

    # the last month in the data is (almost always) partial -> never a label month
    m = m[m["month_idx"] < m["month_idx"].max()].copy()

    # forward window = months +1 .. +h for the same ticker, which must be contiguous
    g = m.groupby("ticker")
    fwd_sum = sum(g["sum_close"].shift(-k) for k in range(1, horizon_months + 1))
    fwd_n   = sum(g["n_days"].shift(-k)   for k in range(1, horizon_months + 1))
    contiguous = (g["month_idx"].shift(-horizon_months) - m["month_idx"]) == horizon_months

    m["next_avg"] = fwd_sum / fwd_n
    m = m[contiguous & m["next_avg"].notna()].copy()
    m["fwd_ret"] = m["next_avg"] / m["last_close"] - 1
    m["label"] = (m["fwd_ret"] > 0).astype(int)

    # market-relative label: same forward return for SPY, joined on the month
    spy = m.loc[m["ticker"] == "SPY", ["month_idx", "fwd_ret"]].rename(columns={"fwd_ret": "spy_fwd_ret"})
    m = m.merge(spy, on="month_idx", how="left")
    m["label_vs_spy"] = (m["fwd_ret"] > m["spy_fwd_ret"]).astype(int)

    m = m[~m["ticker"].isin(NON_STOCK)]
    cols = ["ticker", "month_end", "last_close", "next_avg", "fwd_ret", "spy_fwd_ret", "label", "label_vs_spy"]
    return m[cols].sort_values(["month_end", "ticker"]).reset_index(drop=True)


if __name__ == "__main__":
    prices = pd.read_parquet("data/raw_prices.parquet")
    prices["date"] = prices["date"].astype("datetime64[ns]")
    for h in (1, 3):
        s = build_samples(prices, horizon_months=h)
        print(f"horizon {h} month(s): {len(s):,} samples, {s['ticker'].nunique()} tickers, "
              f"{s['month_end'].min().date()} -> {s['month_end'].max().date()}")
        print(f"  base rate (label=1):        {s['label'].mean():.3f}")
        print(f"  base rate (beats SPY):      {s['label_vs_spy'].mean():.3f}")
        print(f"  median forward return:      {s['fwd_ret'].median():+.4f}")
    by_year = s.groupby(s["month_end"].dt.year)["label"].mean().round(3)
    print("\nbase rate by year (3-month horizon):")
    print(by_year.to_string())
