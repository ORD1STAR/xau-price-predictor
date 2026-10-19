"""
walk_forward_baseline.py
========================
Leakage-free walk-forward evaluation with explicit baselines.

Protocol
--------
* Features come from data/cache/features.parquet (built by `python main.py --mode data`).
* Labels are recomputed from XAU_EUR: 1 if the forward log-return over the horizon is
  positive. Rows without a full forward window are dropped (the cached `next_*`
  columns encode those tail rows as 0, so they are not reused here).
* Expanding window, retrained every RETRAIN_EVERY test days. The training set for
  a prediction made on day t only contains rows whose label window ended at or
  before t (purge = horizon days), so no label overlaps the test period.
* Test period starts at TEST_START (default 2024-01-01).
* Reported next to the model: always-UP accuracy (the majority-class baseline on the
  test period), a two-sided binomial test of accuracy against 50%, and AUC.

Usage:  python evaluation/walk_forward_baseline.py [--test-start 2024-01-01] [--price-col XAU_EUR|XAU_USD]

Data caveat found with this script: in the cached data, EUR_USD returns on day t+1
correlate ~0.28 with XAU_USD returns on day t (the two yfinance series are not
timestamp-aligned), so the constructed XAU_EUR = XAU_USD / EUR_USD series has a
lag-1 autocorrelation of about -0.15 that neither component has (about -0.03). Models
can pick that up; it is a data-alignment artefact, not a tradable signal. Use
--price-col XAU_USD to see the effect on a series without it.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg  # noqa: E402

RETRAIN_EVERY = 63
MIN_TRAIN = 500
TARGET_COLS = {h for h in cfg.HORIZONS} | {f"{h}_return" for h in cfg.HORIZONS}
RAW_COLS = {"XAU_USD", "EUR_USD", "XAU_EUR", "EUR_CHF", "SILVER", "OIL_WTI", "COPPER",
            "PLATINUM", "PALLADIUM", "SP500", "NASDAQ", "VIX", "TIPS_10Y", "HY_ETF",
            "IG_ETF", "BONDS_20", "USD_JPY", "US_10Y", "US_2Y", "US_REAL10", "DXY",
            "FED_RATE", "ECB_RATE", "XAU_VOL"}


def make_model():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=24, max_depth=5,
                          min_child_samples=20, subsample=0.75, subsample_freq=1,
                          colsample_bytree=0.75, reg_lambda=1.5, verbose=-1, random_state=42)


def run(test_start: str, price_col: str = "XAU_EUR"):
    df = pd.read_parquet(cfg.DATA_DIR / "features.parquet").sort_index()
    feats = [c for c in df.columns if c not in TARGET_COLS and c not in RAW_COLS]
    X = df[feats].astype(float).replace([np.inf, -np.inf], np.nan)
    price = df[price_col]
    rows = []
    for name, h in cfg.HORIZONS.items():
        fwd = np.log(price.shift(-h) / price)
        valid = fwd.notna().values
        y = (fwd > 0).astype(int).values
        idx = np.arange(len(df))
        test_idx = idx[valid & (df.index >= pd.Timestamp(test_start))]
        preds = pd.Series(np.nan, index=test_idx)
        for s in range(0, len(test_idx), RETRAIN_EVERY):
            chunk = test_idx[s:s + RETRAIN_EVERY]
            first = chunk[0]
            train = idx[valid & (idx <= first - h)]  # purge: label window ends <= first test day
            if len(train) < MIN_TRAIN:
                continue
            m = make_model().fit(X.iloc[train], y[train])
            preds.loc[chunk] = m.predict_proba(X.iloc[chunk])[:, 1]
        ok = preds.dropna()
        yt = y[ok.index.values]
        pred = (ok.values >= 0.5).astype(int)
        n = len(yt)
        k = int((pred == yt).sum())
        rows.append({
            "target": price_col, "horizon": name, "n_test": n,
            "accuracy": k / n,
            "always_up": yt.mean(),
            "up_rate_pred": pred.mean(),
            "auc": roc_auc_score(yt, ok.values),
            "p_vs_50pct": binomtest(k, n, 0.5).pvalue,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--price-col", default="XAU_EUR",
                    help="price series the labels are built from (XAU_EUR or XAU_USD)")
    a = ap.parse_args()
    out = run(a.test_start, a.price_col)
    pd.set_option("display.width", 140)
    print(out.round(4).to_string(index=False))
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(cfg.RESULTS_DIR / f"walk_forward_baseline_{a.price_col}.csv", index=False)
