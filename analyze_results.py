"""
Diagnostics for a finished run, from its saved predictions (no retraining).

  python analyze_results.py up_h1_price-sector
  python analyze_results.py up_h1_price-sector --detailed

Answers: how spread out are the probabilities, are they calibrated, does the top decile actually
go up more often than the bottom decile (the "good buy" question), how much of the pooled AUC is
market timing vs stock selection, and which sectors the model ranks best.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from labels import build_samples

RESULTS = Path("results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--detailed", action="store_true")
    args = ap.parse_args()

    pred = pd.read_parquet(RESULTS / f"{args.name}_predictions.parquet")
    pred["month_end"] = pd.to_datetime(pred["month_end"])
    by_year = pd.read_csv(RESULTS / f"{args.name}_by_year.csv")
    tickers = pd.read_csv("data/tickers.csv")
    prices = pd.read_parquet("data/raw_prices.parquet")
    prices["date"] = prices["date"].astype("datetime64[ns]")
    horizon = 1
    cfg = RESULTS / f"{args.name}_config.json"
    if cfg.exists():
        import json
        horizon = json.loads(cfg.read_text()).get("horizon", 1)
    samples = build_samples(prices, horizon_months=horizon)[["ticker", "month_end", "fwd_ret", "spy_fwd_ret"]]
    df = pred.merge(samples, on=["ticker", "month_end"], how="left")
    df = df.merge(tickers[["ticker", "sector"]], on="ticker", how="left")
    df["rel_ret"] = df["fwd_ret"] - df["spy_fwd_ret"]

    print(f"run: {args.name} | {len(df):,} test predictions, {df['year'].min()}-{df['year'].max()}")

    # 1. probability spread
    q = df["prob"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3)
    print(f"\n1. Probability spread: mean {df['prob'].mean():.3f}, std {df['prob'].std():.3f}, "
          f"5th-95th pct {q[0.05]:.3f}-{q[0.95]:.3f}")

    # 2. calibration
    df["bin"] = pd.cut(df["prob"], bins=np.linspace(0, 1, 11), include_lowest=True)
    cal = df.groupby("bin", observed=True).agg(n=("label", "size"), predicted=("prob", "mean"), observed=("label", "mean"))
    print("\n2. Calibration (predicted prob vs observed up-rate per bin):")
    print(cal[cal["n"] >= 50].round(3).to_string())

    # 3. deciles within each month: the "good buy" test
    df["decile"] = df.groupby("month_end")["prob"].rank(pct=True).mul(10).clip(upper=9.999).astype(int) + 1
    dec = df.groupby("decile").agg(up_rate=("label", "mean"), fwd_ret=("fwd_ret", "mean"), rel_ret=("rel_ret", "mean"), n=("label", "size"))
    print("\n3. Deciles of model probability within each month (10 = most confident 'up'):")
    print((dec[["up_rate", "fwd_ret", "rel_ret"]] * 100).round(2).rename(columns=lambda c: c + " %").assign(n=dec["n"]).to_string())
    top, bot = df[df["decile"] == 10], df[df["decile"] == 1]
    monthly_spread = (top.groupby("month_end")["fwd_ret"].mean() - bot.groupby("month_end")["fwd_ret"].mean())
    t_stat = monthly_spread.mean() / (monthly_spread.std() / np.sqrt(len(monthly_spread)))
    print(f"   top-decile precision {top['label'].mean():.3f} vs base rate {df['label'].mean():.3f}  "
          f"(+{(top['label'].mean() - df['label'].mean()) * 100:.1f} pts)")
    print(f"   top minus bottom decile forward return: {monthly_spread.mean() * 100:+.2f}% per month, "
          f"t-stat {t_stat:+.2f} over {len(monthly_spread)} months  (|t| > 2 is conventionally significant)")

    # 4. timing vs selection
    monthly = df.groupby("month_end").agg(mean_prob=("prob", "mean"), up_rate=("label", "mean"), spy=("spy_fwd_ret", "first"))
    corr = monthly["mean_prob"].corr(monthly["up_rate"])
    within = [roc_auc_score(g["label"], g["prob"]) for _, g in df.groupby("month_end") if g["label"].nunique() == 2]
    pooled = roc_auc_score(df["label"], df["prob"])
    print(f"\n4. Timing vs selection: pooled AUC {pooled:.3f} | mean within-month AUC {np.mean(within):.3f} "
          f"(std across months {np.std(within):.3f}) | corr(month mean prob, month up-rate) = {corr:+.2f}")
    print("   -> the gap between pooled and within-month AUC is market timing; the correlation says which way it leaned")

    # 5. sectors
    sec = (df.groupby("sector").apply(lambda g: pd.Series({
        "n": len(g), "base_rate": g["label"].mean(),
        "within_month_auc": np.mean([roc_auc_score(x["label"], x["prob"]) for _, x in g.groupby("month_end") if x["label"].nunique() == 2 and len(x) >= 10] or [np.nan])}),
        include_groups=False).sort_values("within_month_auc", ascending=False))
    print("\n5. Within-month AUC by sector:")
    print(sec.round(3).to_string())

    # 6. year summary
    print("\n6. Per-year (from the run):")
    print(by_year[["year", "always_up_acc", "auc", "within_month_auc", "logit_auc", "mcc", "best_epochs"]].round(3).to_string(index=False))
    print(f"   mean AUC {by_year['auc'].mean():.3f} | within-month {by_year['within_month_auc'].mean():.3f} | "
          f"logit {by_year['logit_auc'].mean():.3f} | years > 0.5: {(by_year['auc'] > 0.5).sum()}/{len(by_year)}")

    if args.detailed:
        print("\nMonthly top-minus-bottom decile spread, by year (% per month):")
        print((monthly_spread.groupby(monthly_spread.index.year).mean() * 100).round(2).to_string())


if __name__ == "__main__":
    main()
