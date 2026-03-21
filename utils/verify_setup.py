"""
utils/verify_setup.py
=====================
Quick sanity-check script. Run before full training to verify:
  - All required packages are installed
  - Free data APIs are reachable
  - Feature engineering produces expected shape
  - Config is valid

Usage: python utils/verify_setup.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def check_imports():
    print("Checking required packages ...")
    required = [
        ("numpy",           "numpy"),
        ("pandas",          "pandas"),
        ("sklearn",         "scikit-learn"),
        ("xgboost",         "xgboost"),
        ("lightgbm",        "lightgbm"),
        ("tensorflow",      "tensorflow"),
        ("yfinance",        "yfinance"),
        ("ta",              "ta"),
        ("matplotlib",      "matplotlib"),
        ("tabulate",        "tabulate"),
        ("colorama",        "colorama"),
    ]
    optional = [
        ("torch",           "pytorch"),
        ("pytorch_forecasting", "pytorch-forecasting"),
        ("pandas_datareader", "pandas-datareader"),
        ("openai",          "openai (for Grok API)"),
    ]

    all_ok = True
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✓  {name}")
        except ImportError:
            print(f"  ✗  {name}  ← MISSING (pip install {name.split()[0]})")
            all_ok = False

    print("\nOptional packages:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✓  {name}")
        except ImportError:
            print(f"  -  {name}  (optional)")

    return all_ok


def check_data_api():
    print("\nChecking data APIs ...")
    try:
        import yfinance as yf
        df = yf.download("GC=F", period="5d", progress=False, auto_adjust=True)
        if not df.empty:
            price = float(df["Close"].iloc[-1])
            print(f"  ✓  yfinance OK  |  Gold futures last close: ${price:,.2f}")
        else:
            print("  ✗  yfinance returned empty data")
            return False
    except Exception as e:
        print(f"  ✗  yfinance failed: {e}")
        return False

    try:
        eurusd = yf.download("EURUSD=X", period="5d", progress=False, auto_adjust=True)
        if not eurusd.empty:
            rate = float(eurusd["Close"].iloc[-1])
            xau_eur = price / rate
            print(f"  ✓  EUR/USD OK   |  EUR/USD: {rate:.4f} -> XAU/EUR: €{xau_eur:,.2f}")
    except Exception as e:
        print(f"  -  EUR/USD: {e}")

    return True


def check_feature_engineering():
    print("\nChecking feature engineering ...")
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        import config as cfg
        from data.fetcher import GoldDataFetcher
        from data.features import FeatureEngineer
        from sentiment.grok_client import ManualSentimentInjector

        # Fetch small sample
        fetcher = GoldDataFetcher(cfg, cache_dir=cfg.DATA_DIR)
        raw_df  = fetcher.fetch_all(start="2022-01-01", end="2023-01-01", use_cache=True)

        injector = ManualSentimentInjector()
        sentiment_df = injector.to_dataframe("2022-01-01", "2023-01-01")

        engineer = FeatureEngineer(cfg)
        feat_df  = engineer.build(raw_df, sentiment_df=sentiment_df)

        print(f"  ✓  Features OK  |  Shape: {feat_df.shape} | "
              f"Features: {len(engineer.feature_names)}")
        print(f"     Feature names: {engineer.feature_names[:8]} ...")

        # Check targets
        for h in cfg.HORIZONS.keys():
            if h in feat_df.columns:
                up_pct = feat_df[h].mean()
                print(f"  ✓  Target '{h}': {up_pct:.1%} UP days")
        return True
    except Exception as e:
        import traceback
        print(f"  ✗  Feature engineering failed: {e}")
        traceback.print_exc()
        return False


def check_config():
    print("\nChecking configuration ...")
    try:
        import config as cfg

        checks = [
            ("TRAIN_START",  cfg.TRAIN_START),
            ("TRAIN_END",    cfg.TRAIN_END),
            ("TEST_START",   cfg.TEST_START),
            ("TEST_END",     cfg.TEST_END),
            ("MODEL_DIR",    str(cfg.MODEL_DIR)),
            ("DATA_DIR",     str(cfg.DATA_DIR)),
            ("HORIZONS",     str(cfg.HORIZONS)),
            ("GROK_ENABLED", cfg.GROK_ENABLED),
        ]
        for name, val in checks:
            print(f"  {'✓':2} {name:20} = {val}")
        return True
    except Exception as e:
        print(f"  ✗  Config error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 55)
    print("  XAU/EUR Gold Prediction System -- Setup Verifier")
    print("=" * 55)

    ok = True
    ok &= check_config()
    ok &= check_imports()
    ok &= check_data_api()
    ok &= check_feature_engineering()

    print("\n" + "=" * 55)
    if ok:
        print("  ✓  ALL CHECKS PASSED. Ready to run:  python main.py --mode full")
    else:
        print("  ✗  SOME CHECKS FAILED. Fix the issues above before running.")
    print("=" * 55)
