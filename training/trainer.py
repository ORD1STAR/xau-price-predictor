"""
training/trainer.py
===================
Orchestrates the full training pipeline (Step 2):
  1. Split data into train / test
  2. For each prediction horizon, fit the stacking ensemble
  3. Save all models to disk

Also provides convenience wrappers for:
  - Data preparation checks
  - Incremental fine-tuning (Step 5)
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.features import FeatureEngineer
from models.ensemble import StackingEnsemble
from models.fine_tuner import FineTuner

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config):
        self.config   = config
        self.feature_engineer = FeatureEngineer(config)
        self._ensembles: dict[str, StackingEnsemble] = {}

    # ─────────────────────────────────────────────────────────────
    # STEP 2: TRAIN
    # ─────────────────────────────────────────────────────────────

    def train(
        self,
        feature_df: pd.DataFrame,
        horizons: list[str] = None,
        save_dir: Path = None,
    ) -> dict[str, StackingEnsemble]:
        """
        Train one stacking ensemble per prediction horizon.

        Parameters
        ----------
        feature_df : full feature DataFrame from FeatureEngineer.build()
        horizons   : list of horizon names to train (default: all)
        save_dir   : directory to save trained ensembles

        Returns
        -------
        dict[horizon_name -> StackingEnsemble]
        """
        horizons = horizons or list(self.config.HORIZONS.keys())
        save_dir = save_dir or self.config.MODEL_DIR

        # ── Split data ────────────────────────────────────────────
        train_df, test_df = self._split_train_test(feature_df)
        logger.info(
            f"Train: {train_df.index[0].date()} -> {train_df.index[-1].date()} "
            f"({len(train_df)} rows)"
        )
        logger.info(
            f"Test:  {test_df.index[0].date()} -> {test_df.index[-1].date()} "
            f"({len(test_df)} rows)"
        )

        feature_names = self.feature_engineer.feature_names
        if not feature_names:
            # Infer from df columns
            target_cols = set(horizons)
            for h in horizons:
                target_cols.add(f"{h}_return")
            feature_names = [c for c in train_df.columns
                             if c not in target_cols
                             and not c.startswith("XAU_")
                             and c not in ("XAU_EUR",)]

        # ── Train one ensemble per horizon ───────────────────────
        for horizon in horizons:
            if horizon not in train_df.columns:
                logger.warning(f"Horizon '{horizon}' not in feature_df. Skipping.")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING HORIZON: {horizon.upper()}")
            logger.info(f"{'='*60}")

            t0 = time.time()

            X_train = train_df[feature_names]
            y_train = train_df[horizon]

            ensemble = StackingEnsemble(self.config)
            ensemble.fit(X_train, y_train, horizon=horizon)

            elapsed = time.time() - t0
            logger.info(f"Training done in {elapsed:.1f}s")

            # Save
            horizon_dir = save_dir / horizon
            ensemble.save(horizon_dir)
            self._ensembles[horizon] = ensemble

        logger.info(f"\nAll {len(horizons)} ensembles trained and saved.")
        return self._ensembles

    # ─────────────────────────────────────────────────────────────
    # STEP 5: FINE-TUNE
    # ─────────────────────────────────────────────────────────────

    def finetune(
        self,
        feature_df: pd.DataFrame,
        ensembles: dict[str, StackingEnsemble] = None,
        horizons: list[str] = None,
        lookback_days: int = None,
        retrain_dl: bool = False,
    ) -> None:
        """
        Fine-tune all ensembles on the most recent N days of data.

        Parameters
        ----------
        feature_df    : latest feature DataFrame (must include new rows)
        ensembles     : dict of ensembles (uses self._ensembles if None)
        horizons      : which horizons to fine-tune
        lookback_days : how many recent rows to use (default: config)
        retrain_dl    : whether to update deep learning models
        """
        ensembles     = ensembles or self._ensembles
        horizons      = horizons  or list(self.config.HORIZONS.keys())
        lookback_days = lookback_days or self.config.FINETUNE_LOOKBACK_DAYS

        ft = FineTuner(self.config)

        # Use only recent data
        recent_df = feature_df.iloc[-lookback_days:]
        logger.info(
            f"Fine-tuning on {len(recent_df)} rows: "
            f"{recent_df.index[0].date()} -> {recent_df.index[-1].date()}"
        )

        feature_names = self.feature_engineer.feature_names or [
            c for c in recent_df.columns
            if c not in list(self.config.HORIZONS.keys())
            and not c.endswith("_return")
        ]

        for horizon in horizons:
            if horizon not in ensembles:
                logger.warning(f"No trained ensemble for {horizon}. Skipping fine-tune.")
                continue
            if horizon not in recent_df.columns:
                logger.warning(f"Horizon {horizon} not in recent data. Skipping.")
                continue

            X_new = recent_df[feature_names].dropna()
            y_new = recent_df.loc[X_new.index, horizon]

            logger.info(f"Fine-tuning {horizon}: {len(X_new)} samples ...")
            ft.finetune(
                ensemble   = ensembles[horizon],
                X_new      = X_new,
                y_new      = y_new,
                retrain_dl = retrain_dl,
            )

            # Re-save updated ensemble
            save_dir = self.config.MODEL_DIR / horizon
            ensembles[horizon].save(save_dir)

        logger.info("Fine-tuning pipeline complete.")

    # ─────────────────────────────────────────────────────────────
    # LOAD TRAINED ENSEMBLES
    # ─────────────────────────────────────────────────────────────

    def load_ensembles(
        self,
        horizons: list[str] = None,
        model_dir: Path = None,
    ) -> dict[str, StackingEnsemble]:
        """Load previously trained ensembles from disk."""
        horizons  = horizons  or list(self.config.HORIZONS.keys())
        model_dir = model_dir or self.config.MODEL_DIR

        loaded = {}
        for horizon in horizons:
            horizon_dir = model_dir / horizon
            if not horizon_dir.exists():
                logger.warning(f"No saved model for {horizon} at {horizon_dir}")
                continue
            ens = StackingEnsemble(self.config)
            ens.load(horizon_dir)
            loaded[horizon] = ens
            logger.info(f"Loaded ensemble: {horizon}")

        self._ensembles = loaded
        return loaded

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _split_train_test(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_end  = pd.Timestamp(self.config.TRAIN_END)
        test_start = pd.Timestamp(self.config.TEST_START)

        train_df = df[df.index <= train_end]
        test_df  = df[df.index >= test_start]
        return train_df, test_df
