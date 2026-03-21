"""
models/ensemble.py
==================
Two-layer stacking ensemble with:
  - Purged time-series CV (no look-ahead bias)
  - L2 (Ridge) logistic regression meta-learner
  - Isotonic / Platt probability calibration
  - Optimal threshold search on OOF predictions
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from models.base_models import BaseModel, build_all_base_models

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PURGED TIME-SERIES CV
# ─────────────────────────────────────────────

class PurgedTimeSeriesCV:
    def __init__(self, n_splits=5, test_size=252, gap=10):
        self.n_splits  = n_splits
        self.test_size = test_size
        self.gap       = gap

    def split(self, X, y=None, groups=None):
        n = len(X)
        test_size = min(self.test_size, max(10, n // (self.n_splits + 2)))
        gap       = min(self.gap, max(1, test_size // 10))
        n_splits  = self.n_splits
        while n_splits > 1:
            if n >= test_size * n_splits + gap * n_splits + 20:
                break
            n_splits -= 1
        if n < test_size + gap + 20:
            raise ValueError(f"Not enough data ({n} rows) for CV.")
        if n_splits < self.n_splits:
            logger.warning(f"Auto-scaled CV: {self.n_splits} -> {n_splits} folds")
        for fold in range(n_splits):
            val_end   = n - fold * (test_size + gap)
            val_start = val_end - test_size
            train_end = val_start - gap
            if train_end < 20:
                continue
            yield np.arange(0, train_end), np.arange(val_start, val_end)


# ─────────────────────────────────────────────
# STACKING ENSEMBLE
# ─────────────────────────────────────────────

class StackingEnsemble:
    """
    7 base models -> Ridge meta-learner -> isotonic calibration -> optimal thresholds.
    """

    def __init__(self, config):
        self.config       = config
        self.base_models  = build_all_base_models(config)
        self.meta_learner = None
        self.meta_scaler  = None
        self.calibrator   = None       # isotonic/sigmoid calibrator
        self.buy_threshold  = config.BUY_THRESHOLD
        self.sell_threshold = config.SELL_THRESHOLD
        self._fitted = False

    # ─────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series,
            horizon: str = "next_day", verbose: bool = True) -> "StackingEnsemble":

        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.int32)
        n_samples = len(X_arr)
        n_models  = len(self.base_models)

        cv = PurgedTimeSeriesCV(self.config.CV_N_SPLITS,
                                self.config.CV_TEST_SIZE,
                                self.config.CV_GAP)

        # ── OOF predictions ────────────────────
        oof_preds = np.full((n_samples, n_models), 0.5, dtype=np.float32)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_arr)):
            if verbose:
                logger.info(f"[CV Fold {fold_idx+1}] train={len(train_idx)}, val={len(val_idx)}")

            X_tr, y_tr   = X_arr[train_idx], y_arr[train_idx]
            X_val, y_val = X_arr[val_idx],   y_arr[val_idx]

            for m_idx, model in enumerate(self.base_models):
                if verbose:
                    logger.info(f"  Training {model.name} ...")
                try:
                    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
                    p = model.predict_proba(X_val)
                    p = self._align(p, len(val_idx))
                    oof_preds[val_idx, m_idx] = p
                except Exception as e:
                    logger.error(f"  {model.name} fold {fold_idx+1} failed: {e}")

        # ── Meta-learner ───────────────────────
        covered = (oof_preds != 0.5).any(axis=1)
        meta_X, meta_y = oof_preds[covered], y_arr[covered]

        self.meta_scaler = StandardScaler()
        meta_Xs = self.meta_scaler.fit_transform(meta_X)

        p = self.config.META_LEARNER_PARAMS
        self.meta_learner = LogisticRegression(**p)
        self.meta_learner.fit(meta_Xs, meta_y)

        oof_proba = self.meta_learner.predict_proba(meta_Xs)[:, 1]

        from sklearn.metrics import roc_auc_score, accuracy_score
        logger.info(f"Meta OOF: acc={accuracy_score(meta_y, (oof_proba>=0.5).astype(int)):.4f}"
                    f"  AUC={roc_auc_score(meta_y, oof_proba):.4f}"
                    f"  buy_bias={( oof_proba>=0.5).mean():.1%}")

        # ── Probability calibration ────────────
        self._fit_calibration(oof_proba, meta_y)

        # ── Optimal thresholds ─────────────────
        if self.config.OPTIMIZE_THRESHOLDS:
            self._optimize_thresholds(oof_proba, meta_y)

        # ── Retrain base models on full data ───
        logger.info("Re-training base models on full training set ...")
        for model in self.base_models:
            logger.info(f"  Re-training {model.name} ...")
            try:
                model.fit(X_arr, y_arr)
            except Exception as e:
                logger.error(f"  {model.name} full retrain failed: {e}")

        self._fitted = True
        logger.info(f"Ensemble complete. BUY>{self.buy_threshold:.3f}  "
                    f"SELL<{self.sell_threshold:.3f}")
        return self

    # ─────────────────────────────────────────
    # CALIBRATION
    # ─────────────────────────────────────────

    def _fit_calibration(self, raw_proba: np.ndarray, y: np.ndarray):
        """
        Fit isotonic regression calibrator on OOF probabilities.
        This corrects for systematic over/under-confidence without
        changing the relative ranking of predictions.
        """
        method = getattr(self.config, "CALIBRATION_METHOD", "isotonic")
        if method is None:
            self.calibrator = None
            return

        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw_proba, y)
            self.calibrator = cal
            cal_proba = cal.predict(raw_proba)
        elif method == "sigmoid":
            from sklearn.linear_model import LogisticRegression as LR
            cal = LR(C=1.0, solver="lbfgs")
            cal.fit(raw_proba.reshape(-1, 1), y)
            self.calibrator = cal
            cal_proba = cal.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
        else:
            self.calibrator = None
            return

        # Log calibration improvement
        from sklearn.metrics import brier_score_loss
        bs_before = brier_score_loss(y, raw_proba)
        bs_after  = brier_score_loss(y, cal_proba)
        logger.info(f"Calibration ({method}): Brier {bs_before:.4f} -> {bs_after:.4f}"
                    f"  buy_bias after={( cal_proba>=0.5).mean():.1%}")

    def _calibrate(self, raw_proba: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return raw_proba
        if isinstance(self.calibrator, IsotonicRegression):
            return self.calibrator.predict(raw_proba)
        else:
            return self.calibrator.predict_proba(
                raw_proba.reshape(-1, 1))[:, 1]

    # ─────────────────────────────────────────
    # THRESHOLD OPTIMIZATION
    # ─────────────────────────────────────────

    def _optimize_thresholds(self, raw_proba: np.ndarray, y: np.ndarray):
        """
        Search for BUY/SELL thresholds that maximise:
          - balanced accuracy on calibrated OOF probabilities
        Uses a grid search, not gradient-based, so no look-ahead risk.
        """
        cal_proba = self._calibrate(raw_proba)
        cfg       = self.config
        best_metric, best_buy, best_sell = -1, cfg.BUY_THRESHOLD, cfg.SELL_THRESHOLD

        steps = np.linspace(cfg.THRESHOLD_MIN, cfg.THRESHOLD_MAX, cfg.THRESHOLD_STEPS)

        for buy_t in steps:
            for sell_t in steps:
                if sell_t >= buy_t:
                    continue
                preds = np.where(cal_proba >= buy_t, 1,
                        np.where(cal_proba <= sell_t, 0, -1))
                mask = preds != -1
                if mask.sum() < 20:
                    continue
                from sklearn.metrics import balanced_accuracy_score
                metric = balanced_accuracy_score(y[mask], preds[mask])
                if metric > best_metric:
                    best_metric, best_buy, best_sell = metric, buy_t, sell_t

        self.buy_threshold  = float(best_buy)
        self.sell_threshold = float(best_sell)
        logger.info(f"Optimal thresholds: BUY>{best_buy:.3f}  SELL<{best_sell:.3f}"
                    f"  balanced_acc={best_metric:.4f}")

    # ─────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_arr = X.values.astype(np.float32)
        meta  = np.zeros((len(X_arr), len(self.base_models)), dtype=np.float32)
        for m_idx, model in enumerate(self.base_models):
            try:
                p = model.predict_proba(X_arr)
                meta[:, m_idx] = self._align(p, len(X_arr))
            except Exception as e:
                logger.warning(f"{model.name} predict failed: {e}")
                meta[:, m_idx] = 0.5
        raw = self.meta_learner.predict_proba(
            self.meta_scaler.transform(meta))[:, 1]
        return self._calibrate(raw)

    def predict_signal(self, X: pd.DataFrame,
                       buy_threshold=None, sell_threshold=None):
        buy_t  = buy_threshold  or self.buy_threshold
        sell_t = sell_threshold or self.sell_threshold
        proba  = self.predict_proba(X)
        sig = np.where(proba >= buy_t, "BUY",
              np.where(proba <= sell_t, "SELL", "HOLD"))
        return sig, proba

    def predict_last(self, X: pd.DataFrame) -> dict:
        signals, proba = self.predict_signal(X)
        last_sig  = signals[-1]
        last_prob = proba[-1]
        X_arr = X.values.astype(np.float32)
        model_probas = {}
        for m in self.base_models:
            try:
                p = m.predict_proba(X_arr)[-1]
            except Exception:
                p = 0.5
            model_probas[m.name] = round(float(p), 4)
        return {
            "signal":           last_sig,
            "probability_up":   round(float(last_prob), 4),
            "probability_down": round(1 - float(last_prob), 4),
            "model_votes":      model_probas,
        }

    # ─────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────

    def save(self, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "meta_learner.pkl", "wb") as f:
            pickle.dump({
                "meta_learner":    self.meta_learner,
                "meta_scaler":     self.meta_scaler,
                "calibrator":      self.calibrator,
                "buy_threshold":   self.buy_threshold,
                "sell_threshold":  self.sell_threshold,
            }, f)
        for model in self.base_models:
            model.save(directory / f"{model.name}.pkl")
        logger.info(f"Ensemble saved to {directory}")

    def load(self, directory: Path) -> "StackingEnsemble":
        directory = Path(directory)
        with open(directory / "meta_learner.pkl", "rb") as f:
            d = pickle.load(f)
        self.meta_learner   = d["meta_learner"]
        self.meta_scaler    = d["meta_scaler"]
        self.calibrator     = d.get("calibrator")
        self.buy_threshold  = d.get("buy_threshold",  self.config.BUY_THRESHOLD)
        self.sell_threshold = d.get("sell_threshold", self.config.SELL_THRESHOLD)
        for model in self.base_models:
            p = directory / f"{model.name}.pkl"
            if p.exists():
                loaded = BaseModel.load(p)
                for i, m in enumerate(self.base_models):
                    if m.name == model.name:
                        self.base_models[i] = loaded
                        break
        self._fitted = True
        logger.info(f"Ensemble loaded from {directory}")
        return self

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Ensemble not fitted. Call .fit() first.")

    def get_meta_weights(self) -> dict:
        if self.meta_learner is None:
            return {}
        return {m.name: round(float(c), 4)
                for m, c in zip(self.base_models,
                                self.meta_learner.coef_[0])}

    @staticmethod
    def _align(p: np.ndarray, target_len: int) -> np.ndarray:
        if len(p) >= target_len:
            return p[-target_len:]
        return np.pad(p, (target_len - len(p), 0), constant_values=0.5)
