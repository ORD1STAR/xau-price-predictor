"""
models/fine_tuner.py
====================
Incremental fine-tuning after receiving new live data.

Strategy
--------
1. Tree models (XGBoost, LightGBM): add extra boosting rounds on recent data
   (no full retraining needed -- extremely fast)
2. Deep models (CNN-BiLSTM, LSTM-Attention, TFT): optional few-epoch fine-tune
   with a very small learning rate on recent sequences (backbone frozen)
3. Meta-learner: always retrained from scratch on the updated OOF stack

Usage
-----
    ft = FineTuner(config)
    ft.finetune(
        ensemble   = ensemble,       # StackingEnsemble instance
        X_new      = X_new,          # New feature rows
        y_new      = y_new,          # New target labels
        retrain_dl = False,          # True = update deep models too
    )
"""

import logging
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FineTuner:
    def __init__(self, config):
        self.config = config

    def finetune(
        self,
        ensemble,               # StackingEnsemble
        X_new: pd.DataFrame,    # New feature rows (already engineered)
        y_new: pd.Series,       # New target labels
        retrain_dl: bool = None,
    ) -> None:
        """
        Incrementally update the ensemble with new data.

        Parameters
        ----------
        ensemble   : trained StackingEnsemble
        X_new      : feature DataFrame (same columns as training)
        y_new      : binary direction labels
        retrain_dl : whether to fine-tune deep learning models.
                     Defaults to config.FINETUNE_RETRAIN_DEEP
        """
        if retrain_dl is None:
            retrain_dl = self.config.FINETUNE_RETRAIN_DEEP

        X_arr = X_new.values.astype(np.float32)
        y_arr = y_new.values.astype(np.int32)

        n_new = len(X_arr)
        logger.info(f"Fine-tuning on {n_new} new samples ...")

        # ── 1. Tree models: incremental boosting ──────────────────
        for model in ensemble.base_models:
            if model.name == "xgboost":
                self._finetune_xgb(model, X_arr, y_arr)
            elif model.name == "lightgbm":
                self._finetune_lgbm(model, X_arr, y_arr)
            elif retrain_dl:
                self._finetune_deep(model, X_arr, y_arr)

        # ── 2. Retrain meta-learner on new OOF predictions ────────
        if self.config.FINETUNE_RETRAIN_META:
            self._retrain_meta(ensemble, X_arr, y_arr)

        logger.info("Fine-tuning complete.")

    # ─────────────────────────────────────────────────────────────
    # XGBoost incremental
    # ─────────────────────────────────────────────────────────────

    def _finetune_xgb(self, model, X: np.ndarray, y: np.ndarray):
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("xgboost not installed. Skipping XGBoost fine-tune.")
            return

        extra_rounds = self.config.FINETUNE_TREE_ROUNDS
        logger.info(f"  XGBoost: +{extra_rounds} rounds on {len(y)} new samples ...")

        dtrain = xgb.DMatrix(X, label=y)
        booster = model.model.get_booster()
        params  = {
            "objective":        "binary:logistic",
            "eval_metric":      "logloss",
            "max_depth":        self.config.XGBOOST_PARAMS.get("max_depth", 6),
            "learning_rate":    self.config.XGBOOST_PARAMS.get("learning_rate", 0.01),
            "subsample":        0.8,
            "colsample_bytree": 0.8,
        }
        # Continue training from existing booster
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=extra_rounds,
            xgb_model=booster,
            verbose_eval=False,
        )
        model.model._Booster = booster
        logger.info(f"  XGBoost fine-tune done.")

    # ─────────────────────────────────────────────────────────────
    # LightGBM incremental
    # ─────────────────────────────────────────────────────────────

    def _finetune_lgbm(self, model, X: np.ndarray, y: np.ndarray):
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not installed. Skipping LightGBM fine-tune.")
            return

        extra_rounds = self.config.FINETUNE_TREE_ROUNDS
        logger.info(f"  LightGBM: +{extra_rounds} rounds on {len(y)} new samples ...")

        dtrain = lgb.Dataset(X, label=y)
        params = {
            "objective":        "binary",
            "metric":           "binary_logloss",
            "learning_rate":    self.config.LIGHTGBM_PARAMS.get("learning_rate", 0.01),
            "num_leaves":       self.config.LIGHTGBM_PARAMS.get("num_leaves", 50),
            "verbose":          -1,
        }
        booster = model.model.booster_
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=extra_rounds,
            init_model=booster,
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        model.model._Booster = booster
        logger.info(f"  LightGBM fine-tune done.")

    # ─────────────────────────────────────────────────────────────
    # Deep model incremental (frozen backbone, fine-tune head)
    # ─────────────────────────────────────────────────────────────

    def _finetune_deep(self, model, X: np.ndarray, y: np.ndarray):
        if model.model is None:
            logger.warning(f"  {model.name}: no Keras model found. Skipping.")
            return

        logger.info(f"  {model.name}: fine-tuning {self.config.FINETUNE_DL_EPOCHS} "
                    f"epochs on {len(y)} new samples ...")
        try:
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping

            # Freeze all layers except the last dense block
            for layer in model.model.layers[:-4]:
                layer.trainable = False

            # Use lower LR for fine-tuning
            model.model.optimizer.learning_rate.assign(1e-5)

            from models.base_models import build_sequences
            X_scaled = model.scaler.transform(X)
            X_seq, y_seq = build_sequences(X_scaled, y, model._seq_len)

            if len(X_seq) < 10:
                logger.warning(f"  {model.name}: not enough new sequences. Skipping.")
                return

            model.model.fit(
                X_seq, y_seq,
                epochs=self.config.FINETUNE_DL_EPOCHS,
                batch_size=32,
                callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
                verbose=0,
            )

            # Unfreeze for next time
            for layer in model.model.layers:
                layer.trainable = True

            logger.info(f"  {model.name}: fine-tune done.")
        except Exception as e:
            logger.error(f"  {model.name}: fine-tune failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # Meta-learner retrain
    # ─────────────────────────────────────────────────────────────

    def _retrain_meta(self, ensemble, X: np.ndarray, y: np.ndarray):
        """
        Generate new base-model probabilities on new data,
        then retrain the meta-learner incorporating new evidence.
        """
        logger.info("  Retraining meta-learner on new predictions ...")

        n_models = len(ensemble.base_models)
        meta_input = np.zeros((len(X), n_models), dtype=np.float32)

        for m_idx, model in enumerate(ensemble.base_models):
            try:
                proba = model.predict_proba(X)
                if len(proba) >= len(X):
                    proba = proba[-len(X):]
                else:
                    proba = np.pad(proba, (len(X) - len(proba), 0), constant_values=0.5)
                meta_input[:, m_idx] = proba
            except Exception as e:
                logger.warning(f"  {model.name} predict failed during meta retrain: {e}")
                meta_input[:, m_idx] = 0.5

        # Append to existing meta-learner training data if we have it
        # (here we retrain from scratch on new data only -- for production,
        # store and accumulate historical OOF matrix)
        meta_scaled = ensemble.meta_scaler.transform(meta_input)

        p = self.config.META_LEARNER_PARAMS
        new_meta = LogisticRegression(**p)
        new_meta.fit(meta_scaled, y)

        # Soft update: blend old and new coefficients (0.7 old + 0.3 new)
        if ensemble.meta_learner is not None:
            old_coef = ensemble.meta_learner.coef_
            new_coef = new_meta.coef_
            new_meta.coef_ = 0.7 * old_coef + 0.3 * new_coef

        ensemble.meta_learner = new_meta
        logger.info("  Meta-learner updated.")
