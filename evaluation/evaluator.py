"""
evaluation/evaluator.py
========================
Step 3: Test model accuracy on 2023-2026 data.

Metrics computed per horizon:
  - Direction accuracy
  - ROC-AUC score
  - Precision / Recall / F1
  - Sharpe ratio  (non-overlapping positions, horizon-correct annualisation)
  - Max drawdown  (on non-overlapping trades only)
  - Signal bias   (buy/sell/hold ratio - high BUY bias = trend following, not skill)
  - Confusion matrix
  - Per-model comparison table
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self._results: dict = {}

    def evaluate(
        self,
        feature_df: pd.DataFrame,
        ensembles: dict,
        feature_names: list,
        verbose: bool = True,
    ) -> pd.DataFrame:
        test_df = feature_df[
            (feature_df.index >= pd.Timestamp(self.config.TEST_START)) &
            (feature_df.index <= pd.Timestamp(self.config.TEST_END))
        ]
        if test_df.empty:
            logger.error("No test data found in the specified range!")
            return pd.DataFrame()

        logger.info(
            f"Evaluating on {len(test_df)} test rows: "
            f"{test_df.index[0].date()} -> {test_df.index[-1].date()}"
        )

        rows = []
        for horizon, ensemble in ensembles.items():
            if horizon not in test_df.columns:
                continue

            horizon_days = self.config.HORIZONS.get(horizon, 1)
            X_test  = test_df[feature_names]
            y_true  = test_df[horizon].values
            fwd_ret = test_df.get(
                f"{horizon}_return",
                pd.Series(np.nan, index=test_df.index)
            )

            signals, proba = ensemble.predict_signal(X_test)
            y_pred = (proba >= 0.5).astype(int)

            acc  = accuracy_score(y_true, y_pred)
            auc  = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else 0.5
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0)

            sharpe, max_dd, n_trades = self._trading_metrics(
                signals, fwd_ret.values,
                self.config.BROKER_SPREAD_PCT,
                horizon_days=horizon_days,
            )

            # Signal bias: what fraction are BUY?
            n_sig = len(signals)
            buy_pct  = int((signals == "BUY").sum())  / n_sig
            sell_pct = int((signals == "SELL").sum()) / n_sig

            row = {
                "horizon":      horizon,
                "n_test":       len(y_true),
                "n_trades":     n_trades,
                "accuracy":     round(acc,    4),
                "auc":          round(auc,    4),
                "precision":    round(prec,   4),
                "recall":       round(rec,    4),
                "f1":           round(f1,     4),
                "sharpe":       round(sharpe, 3),
                "max_dd_pct":   round(max_dd * 100, 2),
                "buy_bias_pct": round(buy_pct  * 100, 1),
                "sell_bias_pct":round(sell_pct * 100, 1),
                "buy_signals":  int((signals == "BUY").sum()),
                "sell_signals": int((signals == "SELL").sum()),
                "hold_signals": int((signals == "HOLD").sum()),
            }
            rows.append(row)

            if verbose:
                self._print_horizon_report(horizon, row, y_true, y_pred, proba)

            self._results[horizon] = {
                "metrics": row, "y_true": y_true,
                "y_pred": y_pred, "proba": proba, "signals": signals,
            }

        return pd.DataFrame(rows)

    def evaluate_base_models(
        self,
        feature_df: pd.DataFrame,
        ensembles: dict,
        feature_names: list,
        horizon: str = "next_day",
    ) -> pd.DataFrame:
        test_df = feature_df[feature_df.index >= pd.Timestamp(self.config.TEST_START)]
        X_test  = test_df[feature_names].values.astype(np.float32)
        y_true  = test_df[horizon].values

        if horizon not in ensembles:
            return pd.DataFrame()

        ensemble = ensembles[horizon]
        rows = []

        for model in ensemble.base_models:
            try:
                proba = model.predict_proba(X_test)
                if len(proba) >= len(y_true):
                    proba = proba[-len(y_true):]
                else:
                    proba = np.pad(proba, (len(y_true) - len(proba), 0),
                                   constant_values=0.5)
                acc = accuracy_score(y_true, (proba >= 0.5).astype(int))
                auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else 0.5
                buy_bias = round((proba >= 0.5).mean() * 100, 1)
            except Exception as e:
                logger.warning(f"{model.name} evaluation failed: {e}")
                acc = auc = buy_bias = np.nan

            rows.append({
                "model":        model.name,
                "accuracy":     round(acc, 4),
                "auc":          round(auc, 4),
                "buy_bias_pct": buy_bias,
            })

        signals, proba = ensemble.predict_signal(test_df[feature_names])
        rows.append({
            "model":        "ENSEMBLE (meta)",
            "accuracy":     round(accuracy_score(y_true, (proba >= 0.5).astype(int)), 4),
            "auc":          round(roc_auc_score(y_true, proba), 4),
            "buy_bias_pct": round((proba >= 0.5).mean() * 100, 1),
        })

        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────
    # CORRECTED TRADING SIMULATION
    # ─────────────────────────────────────────────────────────────

    def _trading_metrics(
        self,
        signals: np.ndarray,
        fwd_returns: np.ndarray,
        spread_cost: float = 0.0002,
        horizon_days: int = 1,
    ) -> tuple:
        """
        Non-overlapping position simulation.

        A signal on day i commits to holding for horizon_days.
        The next signal is checked at day i + horizon_days.
        This prevents the same multi-day return being counted
        multiple times (which was inflating Sharpe to 7+ and
        compounding drawdowns to -80%).

        Annualisation:
          daily (1)  -> sqrt(252)   weekly (5) -> sqrt(52)
          2-week(10) -> sqrt(26)    monthly(22)-> sqrt(12)
        """
        n = min(len(signals), len(fwd_returns))
        if n == 0 or np.all(np.isnan(fwd_returns[:n])):
            return 0.0, 0.0, 0

        periods_per_year = 252.0 / max(1, horizon_days)

        trade_returns = []
        i = 0
        while i < n:
            sig = signals[i]
            r   = fwd_returns[i]
            if not np.isnan(r):
                if sig == "BUY":
                    trade_returns.append(r - spread_cost)
                elif sig == "SELL":
                    trade_returns.append(-r - spread_cost)
                else:
                    trade_returns.append(0.0)
            i += max(1, horizon_days)

        if len(trade_returns) < 2:
            return 0.0, 0.0, len(trade_returns)

        tr  = np.array(trade_returns)
        mu  = np.nanmean(tr)
        std = np.nanstd(tr)
        sharpe = (mu / (std + 1e-9)) * np.sqrt(periods_per_year) if std > 0 else 0.0

        cum    = np.cumprod(1 + tr)
        peak   = np.maximum.accumulate(cum)
        max_dd = float(np.min((cum - peak) / (peak + 1e-9)))

        return float(sharpe), max_dd, len(trade_returns)

    # ─────────────────────────────────────────────────────────────
    # REPORTING
    # ─────────────────────────────────────────────────────────────

    def _print_horizon_report(self, horizon, metrics, y_true, y_pred, proba):
        buy_bias = metrics["buy_bias_pct"]
        bias_warn = "  <<< HIGH BUY BIAS - likely trend following" if buy_bias > 65 else ""

        print(f"\n{'='*60}")
        print(f"HORIZON: {horizon.upper()}")
        print(f"{'='*60}")
        print(f"  Test samples  : {metrics['n_test']}  (non-overlapping trades: {metrics['n_trades']})")
        print(f"  Accuracy      : {metrics['accuracy']:.2%}")
        print(f"  ROC-AUC       : {metrics['auc']:.4f}")
        print(f"  Precision     : {metrics['precision']:.4f}")
        print(f"  Recall        : {metrics['recall']:.4f}")
        print(f"  F1 Score      : {metrics['f1']:.4f}")
        print(f"  Sharpe Ratio  : {metrics['sharpe']:.3f}  (non-overlapping, horizon-adjusted)")
        print(f"  Max Drawdown  : {metrics['max_dd_pct']:.2f}%  (on non-overlapping trades)")
        print(f"  Signal bias   : {buy_bias:.1f}% BUY  {metrics['sell_bias_pct']:.1f}% SELL{bias_warn}")
        print(f"  BUY signals   : {metrics['buy_signals']}")
        print(f"  SELL signals  : {metrics['sell_signals']}")
        print(f"  HOLD signals  : {metrics['hold_signals']}")
        print(f"\n  Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(f"    Predicted:  DOWN    UP")
        print(f"    Actual DOWN: {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"    Actual UP:   {cm[1,0]:5d}  {cm[1,1]:5d}")

    def save_report(self, summary: pd.DataFrame, path: Path = None):
        path = path or self.config.RESULTS_DIR / "evaluation_report.csv"
        summary.to_csv(path, index=False)
        logger.info(f"Evaluation report saved to {path}")

    def get_results(self) -> dict:
        return self._results
