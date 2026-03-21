"""
config.py -- Central configuration for XAU/EUR Gold Prediction System
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data" / "cache"
MODEL_DIR = BASE_DIR / "models" / "saved"
RESULTS_DIR = BASE_DIR / "results"
LOG_DIR   = BASE_DIR / "logs"

for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# DATE RANGES
# ─────────────────────────────────────────────
TRAIN_START = "2012-01-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TEST_END    = "2026-03-21"

# ─────────────────────────────────────────────
# DATA TICKERS (yfinance)
# ─────────────────────────────────────────────
TICKERS = {
    "XAU_USD":  "GC=F",
    "EUR_USD":  "EURUSD=X",
    "EUR_CHF":  "EURCHF=X",
    "SILVER":   "SI=F",
    "OIL_WTI":  "CL=F",
    "COPPER":   "HG=F",
    "PLATINUM": "PL=F",
    "SP500":    "^GSPC",
    "VIX":      "^VIX",
    "TIPS_10Y": "TIP",
    "HY_ETF":   "HYG",
    "IG_ETF":   "LQD",
    "NASDAQ":   "^IXIC",      # Tech vs gold rotation
    "USD_JPY":  "USDJPY=X",   # Yen = safe haven competitor
    "BONDS_20": "TLT",        # 20Y Treasury ETF (duration risk)
    "PALLADIUM":"PA=F",       # Platinum group metals
}

# FRED Series
FRED_SERIES = {
    "US_10Y":    "DGS10",
    "US_2Y":     "DGS2",
    "US_REAL10": "DFII10",
    "DXY":       "DTWEXBGS",
    "FED_RATE":  "FEDFUNDS",
    "ECB_RATE":  "ECBDFR",
}

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
LAG_DAYS         = [1, 2, 3, 5, 10, 20, 40]
RSI_PERIOD       = 14
RSI_FAST         = 7     # Fast RSI for divergence
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD    = 2
ATR_PERIOD       = 14
OBV_SMOOTH       = 5

# Rolling windows for cross-asset correlations
CORR_WINDOWS     = [20, 60]

HORIZONS = {
    "next_day":    1,
    "next_week":   5,
    "next_2weeks": 10,
    "next_month":  22,
}

# ─────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────

XGBOOST_PARAMS = {
    "n_estimators":      1000,
    "max_depth":         5,
    "learning_rate":     0.02,
    "subsample":         0.75,
    "colsample_bytree":  0.75,
    "reg_alpha":         0.5,
    "reg_lambda":        2.0,
    "min_child_weight":  10,
    "gamma":             0.2,
    "eval_metric":       "auc",   # AUC better than logloss for imbalanced
    "random_state":      42,
    "n_jobs":            -1,
    # No scale_pos_weight -- let calibration handle bias, not loss reweighting
    # (reweighting causes the overcorrection we saw)
}

LIGHTGBM_PARAMS = {
    "n_estimators":      1000,
    "max_depth":         5,
    "learning_rate":     0.02,
    "num_leaves":        40,
    "subsample":         0.75,
    "colsample_bytree":  0.75,
    "reg_alpha":         0.5,
    "reg_lambda":        2.0,
    "min_child_samples": 30,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
    # No is_unbalance -- calibration handles this
}

CATBOOST_PARAMS = {
    "iterations":        1000,
    "depth":             6,
    "learning_rate":     0.02,
    "l2_leaf_reg":       5.0,
    "subsample":         0.8,
    "random_seed":       42,
    "verbose":           0,
    "eval_metric":       "AUC",
    "loss_function":     "Logloss",
    "od_type":           "Iter",   # Overfitting detector
    "od_wait":           50,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators":      500,
    "max_depth":         10,
    "min_samples_leaf":  20,
    "max_features":      "sqrt",
    "n_jobs":            -1,
    "random_state":      42,
    "class_weight":      "balanced",
}

# CNN-BiLSTM
CNN_BILSTM_PARAMS = {
    "sequence_length":        60,
    "cnn_filters":            [64, 128],
    "cnn_kernel_size":        3,
    "lstm_units":             [128, 64],
    "dropout_rate":           0.3,
    "recurrent_dropout":      0.2,
    "dense_units":            [64, 32],
    "learning_rate":          0.001,
    "batch_size":             64,
    "epochs":                 100,
    "early_stopping_patience": 15,
    "reduce_lr_patience":     8,
}

# LSTM-Attention
LSTM_ATTENTION_PARAMS = {
    "sequence_length":        60,
    "lstm_units":             [128, 64],
    "attention_units":        32,
    "dropout_rate":           0.3,
    "learning_rate":          0.001,
    "batch_size":             64,
    "epochs":                 100,
    "early_stopping_patience": 15,
}

# Temporal Fusion Transformer
TFT_PARAMS = {
    "sequence_length":        60,
    "hidden_size":            64,
    "attention_head_size":    4,
    "dropout":                0.1,
    "hidden_continuous_size": 32,
    "learning_rate":          0.001,
    "batch_size":             64,
    "max_epochs":             80,
}

# WaveNet-style dilated causal CNN (new model)
WAVENET_PARAMS = {
    "sequence_length":        120,    # longer context window
    "n_filters":              64,
    "dilations":              [1, 2, 4, 8, 16, 32],
    "kernel_size":            2,
    "dropout_rate":           0.2,
    "learning_rate":          0.001,
    "batch_size":             64,
    "epochs":                 100,
    "early_stopping_patience": 15,
}

# Meta-learner -- Ridge instead of Lasso (more stable, less overcorrection)
META_LEARNER_PARAMS = {
    "C":           1.0,
    "penalty":     "l2",       # Ridge: keeps all models, weights them
    "solver":      "lbfgs",
    "max_iter":    2000,
    "random_state": 42,
}

# Probability calibration method applied on top of meta-learner
# "isotonic" is better with enough data (>1000 samples)
# "sigmoid" (Platt scaling) better with few samples
CALIBRATION_METHOD = "isotonic"   # "isotonic" | "sigmoid" | None

# ─────────────────────────────────────────────
# TRAINING PROTOCOL
# ─────────────────────────────────────────────
CV_N_SPLITS      = 5
CV_GAP           = 10
CV_TEST_SIZE     = 252
MIN_TRAIN_SIZE   = 504

# Optimal threshold search on OOF predictions
# Instead of fixed 0.5 / 0.6 / 0.4, we search for the best thresholds
# that maximize F1 on validation data
OPTIMIZE_THRESHOLDS = True
THRESHOLD_MIN       = 0.35
THRESHOLD_MAX       = 0.75
THRESHOLD_STEPS     = 80

# ─────────────────────────────────────────────
# PREDICTION THRESHOLDS (fallback if optimization disabled)
# ─────────────────────────────────────────────
BUY_THRESHOLD    = 0.55
SELL_THRESHOLD   = 0.45

# ─────────────────────────────────────────────
# SENTIMENT / GROK API
# ─────────────────────────────────────────────
GROK_ENABLED     = False
GROK_API_KEY     = os.getenv("GROK_API_KEY", "")
GROK_BASE_URL    = "https://api.x.ai/v1"
GROK_MODEL       = "grok-2-latest"
GROK_SEARCH_QUERIES = [
    "gold price inflation",
    "Federal Reserve interest rate",
    "ECB monetary policy",
    "geopolitical risk war",
    "dollar index DXY",
    "gold XAU safe haven",
    "US Treasury yield",
    "gold ETF GLD",
]
GROK_TWEETS_PER_QUERY = 20

# ─────────────────────────────────────────────
# FINE-TUNING
# ─────────────────────────────────────────────
FINETUNE_LOOKBACK_DAYS = 90
FINETUNE_TREE_ROUNDS   = 50
FINETUNE_RETRAIN_META  = True
FINETUNE_RETRAIN_DEEP  = False
FINETUNE_DL_EPOCHS     = 10

# ─────────────────────────────────────────────
# BROKER CONFIG
# ─────────────────────────────────────────────
BROKER_SPREAD_PCT   = 0.0002
BROKER_SLIPPAGE_PCT = 0.0001
RISK_PER_TRADE_PCT  = 0.01
MIN_REWARD_RISK     = 2.0

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = LOG_DIR / "gold_prediction.log"
