"""
main.py
=======
Entry point for the XAU/EUR Gold Prediction System.

Steps
-----
  1. Prepare data      (--mode data)
  2. Train models      (--mode train)     [2012-2022]
  3. Evaluate models   (--mode evaluate)  [2023-2026]
  4. Predict today     (--mode predict)
  5. Fine-tune         (--mode finetune)

Full pipeline:
  python main.py --mode full
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import config as cfg
from data.fetcher import GoldDataFetcher
from data.features import FeatureEngineer
from evaluation.evaluator import ModelEvaluator
from models.ensemble import StackingEnsemble
from prediction.predictor import GoldPredictor
from sentiment.grok_client import GrokSentimentClient, ManualSentimentInjector
from training.trainer import Trainer

# ─────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────

def setup_logging():
    import sys, io

    # Force UTF-8 on Windows (avoids CP1252 UnicodeEncodeError for any stray chars)
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    logging.basicConfig(
        level   = getattr(logging, cfg.LOG_LEVEL, logging.INFO),
        format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cfg.LOG_FILE, mode="a"),
        ],
    )
    # Quiet noisy libraries
    for lib in ["yfinance", "urllib3", "requests", "matplotlib"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────
# STEP 1 -- DATA PREPARATION
# ─────────────────────────────────────────────────────────────────

def step1_prepare_data(use_cache: bool = True):
    """Fetch raw data and build feature matrix."""
    logger = logging.getLogger("step1")
    logger.info("="*60)
    logger.info("STEP 1: Data Preparation")
    logger.info("="*60)

    # 1a. Fetch raw market data
    fetcher = GoldDataFetcher(cfg, cache_dir=cfg.DATA_DIR)
    raw_df  = fetcher.fetch_all(
        start     = cfg.TRAIN_START,
        end       = cfg.TEST_END,
        use_cache = use_cache,
    )

    logger.info(f"Raw data: {raw_df.shape}")
    logger.info(f"Columns: {list(raw_df.columns)}")

    # Print data quality report
    quality = fetcher.describe(raw_df)
    high_null = quality[quality["null_pct"] > 10]
    if not high_null.empty:
        logger.warning(f"High null columns (>10%):\n{high_null[['null_pct']]}")

    # 1b. Build sentiment (manual events + Grok if enabled)
    if cfg.GROK_ENABLED:
        logger.info("Fetching sentiment from Grok API ...")
        grok = GrokSentimentClient(cfg)
        sentiment_df = grok.build_sentiment_df(cfg.TRAIN_START, cfg.TEST_END)
    else:
        logger.info("Using manual sentiment events (Grok disabled).")
        injector = ManualSentimentInjector()
        sentiment_df = injector.to_dataframe(cfg.TRAIN_START, cfg.TEST_END)

    # 1c. Engineer features
    engineer = FeatureEngineer(cfg)
    feature_df = engineer.build(raw_df, sentiment_df=sentiment_df)

    logger.info(f"Feature matrix: {feature_df.shape}")
    logger.info(f"Feature columns ({len(engineer.feature_names)}): "
                f"{engineer.feature_names[:5]} ... {engineer.feature_names[-3:]}")

    # Save feature matrix
    feature_path = cfg.DATA_DIR / "features.parquet"
    feature_df.to_parquet(feature_path)
    logger.info(f"Features saved to {feature_path}")

    return feature_df, engineer.feature_names


# ─────────────────────────────────────────────────────────────────
# STEP 2 -- TRAINING
# ─────────────────────────────────────────────────────────────────

def step2_train(feature_df, feature_names):
    logger = logging.getLogger("step2")
    logger.info("="*60)
    logger.info("STEP 2: Training Models (2012-2022)")
    logger.info("="*60)

    trainer = Trainer(cfg)
    # Inject feature names into the engineer so trainer knows them
    trainer.feature_engineer.feature_names = feature_names

    ensembles = trainer.train(
        feature_df = feature_df,
        horizons   = list(cfg.HORIZONS.keys()),
        save_dir   = cfg.MODEL_DIR,
    )

    logger.info(f"Trained {len(ensembles)} ensembles: {list(ensembles.keys())}")
    return ensembles, trainer


# ─────────────────────────────────────────────────────────────────
# STEP 3 -- EVALUATION
# ─────────────────────────────────────────────────────────────────

def step3_evaluate(feature_df, ensembles, feature_names):
    logger = logging.getLogger("step3")
    logger.info("="*60)
    logger.info("STEP 3: Model Evaluation (2023-2026)")
    logger.info("="*60)

    evaluator = ModelEvaluator(cfg)

    # Full evaluation
    summary = evaluator.evaluate(
        feature_df    = feature_df,
        ensembles     = ensembles,
        feature_names = feature_names,
        verbose       = True,
    )

    print("\n")
    print("="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    try:
        from tabulate import tabulate
        print(tabulate(summary, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))
    except ImportError:
        print(summary.to_string())

    # Per-base-model comparison for next_day horizon
    print("\n--- Individual Model Comparison (next_day) ---")
    model_comparison = evaluator.evaluate_base_models(
        feature_df    = feature_df,
        ensembles     = ensembles,
        feature_names = feature_names,
        horizon       = "next_day",
    )
    try:
        from tabulate import tabulate
        print(tabulate(model_comparison, headers="keys", tablefmt="rounded_outline", floatfmt=".4f"))
    except ImportError:
        print(model_comparison.to_string())

    evaluator.save_report(summary)
    return summary


# ─────────────────────────────────────────────────────────────────
# STEP 4 -- PREDICT TODAY
# ─────────────────────────────────────────────────────────────────

def step4_predict(feature_df, ensembles, feature_names):
    logger = logging.getLogger("step4")
    logger.info("="*60)
    logger.info("STEP 4: Today's Prediction")
    logger.info("="*60)

    predictor = GoldPredictor(cfg)

    # Optional: get live sentiment for today
    sentiment_result = None
    if cfg.GROK_ENABLED:
        grok = GrokSentimentClient(cfg)
        sentiment_result = grok.get_sentiment_today()
        logger.info(f"Sentiment score: {sentiment_result['grok_score']:+.2f} | "
                    f"{sentiment_result['reasoning']}")

    result = predictor.predict_now(
        feature_df      = feature_df,
        ensembles       = ensembles,
        feature_names   = feature_names,
        sentiment_result = sentiment_result,
    )

    predictor.print_prediction(result)

    # Save prediction to results
    import json
    pred_path = cfg.RESULTS_DIR / f"prediction_{result['date']}.json"
    serializable = {
        k: v for k, v in result.items()
        if not callable(v)
    }
    # Convert numpy types
    def default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)
    with open(pred_path, "w") as f:
        json.dump(serializable, f, indent=2, default=default)
    logger.info(f"Prediction saved to {pred_path}")

    return result


# ─────────────────────────────────────────────────────────────────
# STEP 5 -- FINE-TUNE
# ─────────────────────────────────────────────────────────────────

def step5_finetune(feature_df, ensembles, feature_names, retrain_dl: bool = False):
    logger = logging.getLogger("step5")
    logger.info("="*60)
    logger.info("STEP 5: Fine-Tuning with Latest Data")
    logger.info("="*60)

    trainer = Trainer(cfg)
    trainer.feature_engineer.feature_names = feature_names

    trainer.finetune(
        feature_df   = feature_df,
        ensembles    = ensembles,
        lookback_days= cfg.FINETUNE_LOOKBACK_DAYS,
        retrain_dl   = retrain_dl,
    )
    logger.info("Fine-tuning complete.")


# ─────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────

def load_features_and_models():
    """Load saved features and models without retraining."""
    logger = logging.getLogger("loader")
    import pandas as pd

    feature_path = cfg.DATA_DIR / "features.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Features not found at {feature_path}. Run --mode data first."
        )
    feature_df = pd.read_parquet(feature_path)
    logger.info(f"Loaded features: {feature_df.shape}")

    # Re-derive feature names
    target_cols = set(cfg.HORIZONS.keys())
    for h in cfg.HORIZONS.keys():
        target_cols.add(f"{h}_return")
    raw_price_cols = {"XAU_USD", "EUR_USD", "XAU_EUR", "EUR_CHF", "SILVER",
                      "OIL_WTI", "COPPER", "PLATINUM", "SP500", "VIX",
                      "TIPS_10Y", "HY_ETF", "IG_ETF", "US_10Y", "US_2Y",
                      "US_REAL10", "DXY", "FED_RATE", "ECB_RATE", "XAU_VOL"}
    feature_names = [c for c in feature_df.columns
                     if c not in target_cols and c not in raw_price_cols]
    logger.info(f"Feature names ({len(feature_names)}): {feature_names[:5]} ...")

    trainer = Trainer(cfg)
    trainer.feature_engineer.feature_names = feature_names
    ensembles = trainer.load_ensembles()

    return feature_df, ensembles, feature_names


def run_full_pipeline(args):
    """Run all 5 steps sequentially."""
    feature_df, feature_names = step1_prepare_data(use_cache=not args.no_cache)
    ensembles, trainer        = step2_train(feature_df, feature_names)
    step3_evaluate(feature_df, ensembles, feature_names)
    step4_predict(feature_df, ensembles, feature_names)
    # Step 5 (fine-tune) after initial training not needed; run it after live use



# ─────────────────────────────────────────────────────────────────
# CLEAR SAVED MODELS
# ─────────────────────────────────────────────────────────────────

def _clear_saved_models(logger):
    """
    Delete all saved model .pkl files and feature cache.
    Run this after any code change to base_models.py or features.py
    before retraining, so stale pickles don't cause errors.
    """
    import shutil

    model_dir = cfg.MODEL_DIR
    data_cache = cfg.DATA_DIR

    removed = 0
    # Delete saved model directories
    if model_dir.exists():
        for horizon_dir in model_dir.iterdir():
            if horizon_dir.is_dir():
                shutil.rmtree(horizon_dir)
                logger.info(f"Removed model dir: {horizon_dir}")
                removed += 1
        # Remove any stray .pkl at root level
        for pkl in model_dir.glob("*.pkl"):
            pkl.unlink()
            logger.info(f"Removed: {pkl}")
            removed += 1

    # Delete feature cache (so features are rebuilt with new columns)
    if data_cache.exists():
        for f in data_cache.glob("features*.parquet"):
            f.unlink()
            logger.info(f"Removed feature cache: {f}")
            removed += 1

    if removed == 0:
        logger.info("Nothing to clear - no saved models or feature cache found.")
    else:
        logger.info(f"Cleared {removed} files/dirs. Ready for fresh training.")
        logger.info("Next step: python main.py --mode train")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="XAU/EUR Gold Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full      Run all 5 steps (data -> train -> evaluate -> predict)
  data      Step 1: fetch & prepare data only
  train     Step 2: train models (requires data step first)
  evaluate  Step 3: test accuracy on 2023-2026
  predict   Step 4: get today's BUY/SELL/HOLD signal
  finetune  Step 5: fine-tune with latest data
  clear     Delete saved models (run before retraining after code changes)

Examples:
  python main.py --mode full
  python main.py --mode predict
  python main.py --mode finetune --dl         (also update deep models)
  python main.py --mode train --no-cache      (force re-download data)
        """,
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "data", "train", "evaluate", "predict", "finetune", "clear"],
        help="Which pipeline step(s) to run",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-download of data (ignore cache)",
    )
    parser.add_argument(
        "--dl", action="store_true",
        help="Also fine-tune deep learning models (slow)",
    )
    parser.add_argument(
        "--horizons", type=str, nargs="+",
        default=None,
        help="Which horizons to train/predict (default: all)",
    )

    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger("main")

    start_time = datetime.now()
    logger.info(f"XAU/EUR Gold Prediction System -- Mode: {args.mode.upper()}")
    logger.info(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        if args.mode == "full":
            run_full_pipeline(args)

        elif args.mode == "data":
            step1_prepare_data(use_cache=not args.no_cache)

        elif args.mode == "train":
            feature_df, feature_names = step1_prepare_data(use_cache=True)
            step2_train(feature_df, feature_names)

        elif args.mode == "evaluate":
            feature_df, ensembles, feature_names = load_features_and_models()
            step3_evaluate(feature_df, ensembles, feature_names)

        elif args.mode == "predict":
            feature_df, ensembles, feature_names = load_features_and_models()
            # Refresh last 120 days of data for freshest features
            fetcher = GoldDataFetcher(cfg, cache_dir=cfg.DATA_DIR)
            latest_raw = fetcher.fetch_latest(n_days=120)
            engineer = FeatureEngineer(cfg)
            engineer.feature_names = feature_names
            # Quick rebuild of recent features
            injector = ManualSentimentInjector()
            import pandas as pd
            start_str = latest_raw.index[0].strftime("%Y-%m-%d")
            end_str   = latest_raw.index[-1].strftime("%Y-%m-%d")
            sentiment_df = injector.to_dataframe(start_str, end_str)
            latest_features = engineer.build(latest_raw, sentiment_df=sentiment_df)
            # Use trained feature names (align columns)
            available = [f for f in feature_names if f in latest_features.columns]
            step4_predict(latest_features, ensembles, available)

        elif args.mode == "clear":
            _clear_saved_models(logger)

        elif args.mode == "finetune":
            feature_df, ensembles, feature_names = load_features_and_models()
            # Refresh latest data
            fetcher = GoldDataFetcher(cfg, cache_dir=cfg.DATA_DIR)
            latest_raw = fetcher.fetch_latest(n_days=cfg.FINETUNE_LOOKBACK_DAYS + 30)
            engineer = FeatureEngineer(cfg)
            engineer.feature_names = feature_names
            injector = ManualSentimentInjector()
            import pandas as pd
            start_str = latest_raw.index[0].strftime("%Y-%m-%d")
            end_str   = latest_raw.index[-1].strftime("%Y-%m-%d")
            sentiment_df = injector.to_dataframe(start_str, end_str)
            latest_features = engineer.build(latest_raw, sentiment_df=sentiment_df)
            available = [f for f in feature_names if f in latest_features.columns]
            step5_finetune(latest_features, ensembles, available, retrain_dl=args.dl)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
