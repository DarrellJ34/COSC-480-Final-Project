"""
Walk-forward experiment on real data.

  python run_experiment.py                         # price + sector, test years 2010-2025, 1 seed
  python run_experiment.py --quick                 # pipeline check: 150 tickers, 3 test years, 3 epochs
  python run_experiment.py --groups price,sector,macro --seeds 5
  python run_experiment.py --shuffle-labels        # leakage check: every year's AUC should be ~0.5
  python run_experiment.py --label vs_spy          # "beats SPY" label
  python run_experiment.py --horizon 3             # 3-month horizon (purge grows to match)

Per test year: train on earlier years, validate on the year before (early stopping + threshold),
test once. Seeds are ensembled by averaging probabilities. Writes results/<name>_by_year.csv,
results/<name>_predictions.parquet and results/<name>_config.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import begin as M
from labels import build_samples
from features import build_daily_features, FEATURE_COLS
from dataset import WindowDataset, make_loader, walk_forward_split, sector_one_hot, macro_static

DATA, RESULTS = Path("data"), Path("results")
sys.stdout.reconfigure(line_buffering=True)   # so progress shows up live even when piped to a log


def within_month_auc(probs, labels, month_end):
    """AUC computed inside each month, then averaged: measures stock SELECTION only, not market timing."""
    df = pd.DataFrame({"p": probs, "y": labels, "m": month_end})
    aucs = [roc_auc_score(g["y"], g["p"]) for _, g in df.groupby("m") if g["y"].nunique() == 2]
    return float(np.mean(aucs)) if aucs else float("nan")


def load_features(prices, tickers_df, rebuild=False):
    cache = DATA / "features_daily.parquet"
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    print("building daily features (one-time, cached to data/features_daily.parquet)...")
    feats = build_daily_features(prices, tickers_df)
    feats.to_parquet(cache, index=False)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="price,sector", help="comma list from: price, sector, macro")
    ap.add_argument("--label", default="up", choices=["up", "vs_spy"])
    ap.add_argument("--horizon", type=int, default=1, help="label horizon in months")
    ap.add_argument("--seq-len", type=int, default=120)
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--shuffle-labels", action="store_true", help="leakage check")
    ap.add_argument("--quick", action="store_true", help="small subset to check the pipeline")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print every epoch")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    groups = set(args.groups.split(","))
    if "price" not in groups:
        raise SystemExit("the LSTM sequence needs the 'price' group")
    name = args.name or f"{args.label}_h{args.horizon}_{'-'.join(sorted(groups))}" + \
        ("_shuffled" if args.shuffle_labels else "") + ("_quick" if args.quick else "")
    RESULTS.mkdir(exist_ok=True)
    print(f"run: {name}  | device: {M.device}")

    # ---- data -------------------------------------------------------------------------------
    prices = pd.read_parquet(DATA / "raw_prices.parquet")
    prices["date"] = prices["date"].astype("datetime64[ns]")
    tickers_df = pd.read_csv(DATA / "tickers.csv")
    features = load_features(prices, tickers_df, rebuild=args.rebuild_features)
    samples = build_samples(prices, horizon_months=args.horizon)

    years = list(range(args.start_year, args.end_year + 1))
    if args.quick:
        rng = np.random.default_rng(0)
        subset = set(rng.choice(tickers_df["ticker"].to_numpy(), size=150, replace=False))
        samples = samples[samples["ticker"].isin(subset)].reset_index(drop=True)
        features = features[features["ticker"].isin(subset)]
        years, args.epochs = years[:3], min(args.epochs, 3)

    label_col = "label" if args.label == "up" else "label_vs_spy"
    y = samples[label_col].to_numpy()
    if args.shuffle_labels:
        y = np.random.default_rng(0).permutation(y)

    static_parts, static_names = [], []
    if "sector" in groups:
        mat, names = sector_one_hot(samples, tickers_df); static_parts.append(mat); static_names += names
    if "macro" in groups:
        mat, names = macro_static(samples, pd.read_parquet(DATA / "macro.parquet"))
        static_parts.append(mat); static_names += names
    static = np.concatenate(static_parts, axis=1) if static_parts else np.zeros((len(samples), 0), np.float32)

    ds = WindowDataset(samples, features, FEATURE_COLS, static, y, seq_len=args.seq_len)
    X_last = ds.last_day_matrix()                                   # for the logistic baseline
    X_flat = np.concatenate([X_last, ds.static.numpy()], axis=1)
    y_ds = ds.labels.numpy()
    print(f"samples: {len(ds):,} windows from {len(samples):,} month-ends | "
          f"seq {args.seq_len} x {len(FEATURE_COLS)} features | static {static.shape[1]} ({', '.join(sorted(groups))}) | "
          f"base rate {y_ds.mean():.3f}")

    # ---- walk-forward loop ------------------------------------------------------------------
    rows, preds = [], []
    for year in years:
        tr, va, te = walk_forward_split(ds.month_end, year, purge_months=args.horizon)
        if tr.sum() < 1000 or va.sum() < 200 or te.sum() < 200:
            print(f"{year}: skipped (train {tr.sum()}, val {va.sum()}, test {te.sum()})")
            continue
        assert not (set(ds.month_end[tr]) & set(ds.month_end[te])), "test month inside train"
        t0 = time.time()
        train_loader = make_loader(ds, tr, args.batch_size, shuffle=True)
        val_loader, test_loader = make_loader(ds, va, 1024), make_loader(ds, te, 1024)

        val_probs, test_probs, best_epochs = [], [], []
        for seed in range(args.seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            model = M.HybridLSTM(len(FEATURE_COLS), static.shape[1], hidden=args.hidden, dropout=args.dropout)
            model, hist = M.train_model(model, train_loader, val_loader, num_epochs=args.epochs, lr=args.lr,
                                        patience=args.patience, verbose=args.verbose)
            best_epochs.append(int(max(hist, key=lambda r: r["val_auc"])["epoch"]))
            val_probs.append(M.predict_proba(model, val_loader)[0])
            test_probs.append(M.predict_proba(model, test_loader)[0])
        vp, tp = np.mean(val_probs, axis=0), np.mean(test_probs, axis=0)   # seed ensemble
        thr = M.choose_threshold_from_probs(vp, y_ds[va])
        m = M.metrics_from_probs(tp, y_ds[te], thr)

        logit = LogisticRegression(max_iter=2000).fit(X_flat[tr], y_ds[tr])
        logit_auc = float(roc_auc_score(y_ds[te], logit.predict_proba(X_flat[te])[:, 1]))

        m.update(year=year, n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()),
                 within_month_auc=within_month_auc(tp, y_ds[te], ds.month_end[te]),
                 logit_auc=logit_auc, val_auc=float(roc_auc_score(y_ds[va], vp)),
                 best_epochs=",".join(map(str, best_epochs)), seconds=round(time.time() - t0))
        rows.append(m)
        preds.append(pd.DataFrame({"year": year, "ticker": ds.tickers[te], "month_end": ds.month_end[te],
                                   "label": y_ds[te], "prob": tp}))
        print(f"{year} | train {m['n_train']:>6,} test {m['n_test']:>5,} | base {m['always_up_acc']:.3f} | "
              f"AUC {m['auc']:.3f} (within-month {m['within_month_auc']:.3f}, logit {logit_auc:.3f}) | "
              f"acc {m['acc'] * 100:4.1f}% edge {(m['acc'] - m['always_up_acc']) * 100:+.1f} | "
              f"MCC {m['mcc']:+.3f} | ep {m['best_epochs']} | {m['seconds']}s")

    # ---- summary + files --------------------------------------------------------------------
    if not rows:
        raise SystemExit("no test years ran")
    res = pd.DataFrame(rows)
    cols = ["year", "n_test", "always_up_acc", "auc", "within_month_auc", "logit_auc", "acc", "mcc", "threshold"]
    print("\n" + res[cols].round(3).to_string(index=False))
    print(f"\nmean AUC {res['auc'].mean():.3f} (within-month {res['within_month_auc'].mean():.3f}, "
          f"logit {res['logit_auc'].mean():.3f}) | years with AUC > 0.5: {(res['auc'] > 0.5).sum()}/{len(res)} | "
          f"mean edge {(res['acc'] - res['always_up_acc']).mean() * 100:+.1f} pts | mean MCC {res['mcc'].mean():+.3f}")

    res.to_csv(RESULTS / f"{name}_by_year.csv", index=False)
    pd.concat(preds).to_parquet(RESULTS / f"{name}_predictions.parquet", index=False)
    (RESULTS / f"{name}_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"saved results/{name}_by_year.csv, _predictions.parquet, _config.json")


if __name__ == "__main__":
    main()
