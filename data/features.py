"""
data/features.py
================
Builds 70+ features across 9 categories.

1.  Technical       lag returns, RSI, MACD, Bollinger, ATR, OBV, stochastic
2.  Momentum        multi-timeframe returns, MA crossovers, detrended price
3.  Volatility      realized vol, vol-of-vol, Parkinson estimator, vol regime
4.  Regime          ADX, Hurst exponent, trending vs ranging flags
5.  Currency        EUR/USD, DXY, EUR/CHF, USD/JPY
6.  Rates/Macro     yields, real yield, yield curve, Fed/ECB rates
7.  Cross-asset     rolling correlations XAU vs S&P/bonds/VIX/USD
8.  Commodities     Silver, Oil, Copper, Platinum, Palladium ratios
9.  Calendar        day-of-week, month, quarter-end, month-end
10. Sentiment       Grok score, news score (optional)
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.feature_names: list = []

    def build(self, raw_df: pd.DataFrame,
              sentiment_df: pd.DataFrame = None) -> pd.DataFrame:
        df = raw_df.copy().sort_index()
        logger.info("Building feature set ...")

        df = self._add_technical(df)
        df = self._add_momentum(df)
        df = self._add_volatility(df)
        df = self._add_regime(df)
        df = self._add_currency(df)
        df = self._add_macro(df)
        df = self._add_cross_asset(df)
        df = self._add_commodities(df)
        df = self._add_calendar(df)
        df = self._add_sentiment(df, sentiment_df)
        df = self._add_targets(df)

        feature_cols = self._get_feature_columns(df)
        self.feature_names = feature_cols

        # Diagnose nulls
        null_counts = df[feature_cols].isnull().sum()
        bad_cols = null_counts[null_counts > len(df) * 0.15]
        if not bad_cols.empty:
            logger.warning(f"Sparse feature cols (filling with median):\n{bad_cols.to_string()}")
            for col in bad_cols.index:
                df[col] = df[col].fillna(df[col].median())

        df = df.dropna(subset=feature_cols + list(self.config.HORIZONS.keys()))
        logger.info(f"Feature matrix: {df.shape[0]} rows x {len(feature_cols)} features")
        return df

    def get_X_y(self, df, horizon="next_day"):
        return df[self.feature_names], df[horizon]

    # ── 1. Technical ────────────────────────────────────────────

    def _add_technical(self, df):
        cfg   = self.config
        price = df["XAU_EUR"]
        log_ret = np.log(price / price.shift(1))

        for lag in cfg.LAG_DAYS:
            df[f"ret_{lag}d"] = np.log(price / price.shift(lag))

        df["rsi_14"]      = self._rsi(price, cfg.RSI_PERIOD)
        df["rsi_7"]       = self._rsi(price, cfg.RSI_FAST)
        df["rsi_div"]     = df["rsi_7"] - df["rsi_14"]     # fast/slow RSI divergence

        macd, sig = self._macd(price, cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL)
        df["macd_diff"]   = macd - sig
        df["macd_hist_slope"] = (macd - sig).diff(3)       # histogram slope

        bb_u, bb_l = self._bollinger(price, cfg.BOLLINGER_WINDOW, cfg.BOLLINGER_STD)
        df["bb_width"]    = (bb_u - bb_l) / price
        df["bb_pos"]      = (price - bb_l) / (bb_u - bb_l + 1e-9)  # position within bands

        # Stochastic oscillator
        hi14 = price.rolling(14).max()
        lo14 = price.rolling(14).min()
        df["stoch_k"]     = 100 * (price - lo14) / (hi14 - lo14 + 1e-9)
        df["stoch_d"]     = df["stoch_k"].rolling(3).mean()

        if "XAU_HIGH" in df.columns and "XAU_LOW" in df.columns:
            df["atr_14"]  = self._atr(df["XAU_HIGH"], df["XAU_LOW"], price, cfg.ATR_PERIOD) / price
        else:
            df["atr_14"]  = price.pct_change().abs().rolling(cfg.ATR_PERIOD).mean()

        if "XAU_VOL" in df.columns:
            obv = self._obv(price, df["XAU_VOL"])
            df["obv_change"] = obv.pct_change(cfg.OBV_SMOOTH)
        else:
            df["obv_change"] = price.pct_change(cfg.OBV_SMOOTH)

        df["price_pos_52w"] = (price - price.rolling(252).min()) / \
                              (price.rolling(252).max() - price.rolling(252).min() + 1e-9)
        return df

    # ── 2. Momentum ─────────────────────────────────────────────

    def _add_momentum(self, df):
        price = df["XAU_EUR"]

        # Moving average crossovers (normalized)
        for fast, slow in [(10, 30), (20, 50), (50, 200)]:
            mf = price.rolling(fast).mean()
            ms = price.rolling(slow).mean()
            df[f"ma_{fast}_{slow}"] = (mf - ms) / (ms + 1e-9)

        # Detrended price vs trend
        ma_200 = price.rolling(200).mean()
        df["price_vs_200ma"] = (price - ma_200) / (ma_200 + 1e-9)

        ma_60  = price.rolling(60).mean()
        df["price_zscore_60"] = (price - ma_60) / (price.rolling(60).std() + 1e-9)

        # RSI momentum
        df["rsi_momentum_5d"] = df["rsi_14"].diff(5)

        # Consecutive streak
        log_ret = np.log(price / price.shift(1))
        d = np.sign(log_ret)
        streak = pd.Series(0.0, index=price.index)
        for i in range(1, len(d)):
            if d.iloc[i] == d.iloc[i-1] and d.iloc[i] != 0:
                streak.iloc[i] = np.clip(streak.iloc[i-1] + d.iloc[i], -7, 7)
            else:
                streak.iloc[i] = d.iloc[i]
        df["streak"] = streak

        # Time since last 52-week high/low
        roll_max = price.rolling(252).max()
        roll_min = price.rolling(252).min()
        df["days_from_52w_high"] = (price < roll_max).astype(int).rolling(252).sum() / 252
        df["days_from_52w_low"]  = (price > roll_min).astype(int).rolling(252).sum() / 252

        return df

    # ── 3. Volatility ────────────────────────────────────────────

    def _add_volatility(self, df):
        price   = df["XAU_EUR"]
        log_ret = np.log(price / price.shift(1))

        df["realized_vol_5d"]  = log_ret.rolling(5).std()  * np.sqrt(252)
        df["realized_vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)
        df["realized_vol_60d"] = log_ret.rolling(60).std() * np.sqrt(252)

        # Vol-of-vol (how stable is volatility itself)
        df["vol_of_vol"] = df["realized_vol_20d"].rolling(20).std()

        # Volatility regime
        vol_20  = df["realized_vol_20d"]
        vol_60m = vol_20.rolling(60).median()
        df["vol_regime"]  = (vol_20 > vol_60m).astype(int)
        df["vol_zscore"]  = (vol_20 - vol_20.rolling(252).mean()) / \
                             (vol_20.rolling(252).std() + 1e-9)

        # Parkinson volatility estimator (uses High-Low range if available)
        if "XAU_HIGH" in df.columns and "XAU_LOW" in df.columns:
            pk = (1 / (4 * np.log(2))) * (np.log(df["XAU_HIGH"] / df["XAU_LOW"])) ** 2
            df["parkinson_vol"] = np.sqrt(pk.rolling(20).mean() * 252)
        else:
            df["parkinson_vol"] = df["realized_vol_20d"]

        return df

    # ── 4. Regime ────────────────────────────────────────────────

    def _add_regime(self, df):
        price = df["XAU_EUR"]
        adx = self._adx(price, period=14)
        df["adx_14"]         = adx
        df["adx_change_10d"] = adx.diff(10)
        df["regime_trend"]   = (adx > 25).astype(int)
        df["hurst_40"]       = self._hurst_rolling(price, window=40)

        # Bull/bear market flags
        ma_200 = price.rolling(200).mean()
        df["bull_market"]    = (price > ma_200).astype(int)

        # Choppiness index (high = choppy, low = trending)
        atr_1d  = price.pct_change().abs()
        atr_14  = atr_1d.rolling(14).sum()
        hi14    = price.rolling(14).max()
        lo14    = price.rolling(14).min()
        df["choppiness"] = 100 * np.log10(atr_14 / (hi14 - lo14 + 1e-9)) / np.log10(14)

        return df

    # ── 5. Currency ──────────────────────────────────────────────

    def _add_currency(self, df):
        for col, name, periods in [
            ("EUR_USD", "eurusd", [1, 5, 20]),
            ("EUR_CHF", "eurchf", [5]),
        ]:
            if col in df.columns:
                for p in periods:
                    df[f"{name}_ret_{p}d"] = np.log(df[col] / df[col].shift(p))

        # DXY (strong negative correlation with gold)
        if "DXY" in df.columns:
            dxy = df["DXY"]
        elif "EUR_USD" in df.columns:
            dxy = 1.0 / df["EUR_USD"]
        else:
            dxy = None
        if dxy is not None:
            for p in [1, 5, 20]:
                df[f"dxy_ret_{p}d"] = np.log(dxy / dxy.shift(p))
            df["dxy_zscore_60"] = (dxy - dxy.rolling(60).mean()) / (dxy.rolling(60).std() + 1e-9)

        # USD/JPY (yen = safe haven, rises when risk-off like gold)
        if "USD_JPY" in df.columns:
            uj = df["USD_JPY"]
            df["usdjpy_ret_5d"] = np.log(uj / uj.shift(5))
            df["usdjpy_ret_20d"] = np.log(uj / uj.shift(20))

        return df

    # ── 6. Rates / Macro ─────────────────────────────────────────

    def _add_macro(self, df):
        if "US_10Y" in df.columns:
            df["us10y_level"]     = df["US_10Y"]
            df["us10y_chg_20d"]   = df["US_10Y"].diff(20)
            df["us10y_chg_60d"]   = df["US_10Y"].diff(60)

        if "US_2Y" in df.columns:
            df["us2y_level"] = df["US_2Y"]
            df["us2y_chg_5d"] = df["US_2Y"].diff(5)

        if "US_10Y" in df.columns and "US_2Y" in df.columns:
            df["yield_curve"]     = df["US_10Y"] - df["US_2Y"]
            df["yield_curve_chg"] = df["yield_curve"].diff(20)
            # Inversion duration (days of inverted curve)
            inv = (df["yield_curve"] < 0).astype(float)
            df["inversion_days"]  = inv.rolling(252).sum()

        if "US_REAL10" in df.columns:
            df["real_yield"]     = df["US_REAL10"]
            df["real_yield_chg"] = df["US_REAL10"].diff(20)
        elif "US_10Y" in df.columns:
            df["real_yield"]     = df["US_10Y"] - 2.0
            df["real_yield_chg"] = df["real_yield"].diff(20)

        if "FED_RATE" in df.columns:
            df["fed_rate"]     = df["FED_RATE"]
            df["fed_rate_chg"] = df["FED_RATE"].diff(60)

        if "ECB_RATE" in df.columns:
            df["ecb_rate"] = df["ECB_RATE"]

        # 20Y Treasury bond ETF (duration / long-end rates)
        if "BONDS_20" in df.columns:
            b = df["BONDS_20"]
            df["bonds20_ret_5d"]  = np.log(b / b.shift(5))
            df["bonds20_ret_20d"] = np.log(b / b.shift(20))

        return df

    # ── 7. Cross-asset correlations ──────────────────────────────

    def _add_cross_asset(self, df):
        """
        Rolling correlations between XAU/EUR and other assets.
        These are powerful regime-detection features:
        - Gold/S&P corr > 0: risk-on, both rising (unusual, often topping)
        - Gold/VIX corr > 0: fear-driven gold rally
        - Gold/DXY corr < 0: normal (dollar down = gold up)
        """
        price   = df["XAU_EUR"]
        log_ret = np.log(price / price.shift(1))

        corr_windows = getattr(self.config, "CORR_WINDOWS", [20, 60])

        assets = {}
        if "SP500" in df.columns:
            assets["sp500"] = np.log(df["SP500"] / df["SP500"].shift(1))
        if "VIX" in df.columns:
            assets["vix"]   = df["VIX"].diff()
        if "BONDS_20" in df.columns:
            assets["bonds"] = np.log(df["BONDS_20"] / df["BONDS_20"].shift(1))

        dxy_ret = None
        if "DXY" in df.columns:
            dxy_ret = np.log(df["DXY"] / df["DXY"].shift(1))
        elif "EUR_USD" in df.columns:
            dxy_ret = np.log((1 / df["EUR_USD"]) / (1 / df["EUR_USD"]).shift(1))
        if dxy_ret is not None:
            assets["dxy"] = dxy_ret

        for asset_name, asset_ret in assets.items():
            for w in corr_windows:
                col = f"corr_gold_{asset_name}_{w}d"
                df[col] = log_ret.rolling(w).corr(asset_ret)

        # S&P500 level and momentum (equity -> gold rotation)
        if "SP500" in df.columns:
            sp = df["SP500"]
            df["sp500_ret_20d"]  = np.log(sp / sp.shift(20))
            df["sp500_ret_60d"]  = np.log(sp / sp.shift(60))
            df["sp500_vs_200ma"] = (sp - sp.rolling(200).mean()) / (sp.rolling(200).mean() + 1e-9)

        # VIX level and momentum
        if "VIX" in df.columns:
            df["vix_level"]     = df["VIX"]
            df["vix_chg_5d"]    = df["VIX"].diff(5)
            df["vix_chg_20d"]   = df["VIX"].diff(20)
            df["vix_zscore_60"] = (df["VIX"] - df["VIX"].rolling(60).mean()) / \
                                   (df["VIX"].rolling(60).std() + 1e-9)

        # Nasdaq (tech sector - risk appetite proxy)
        if "NASDAQ" in df.columns:
            nq = df["NASDAQ"]
            df["nasdaq_ret_20d"] = np.log(nq / nq.shift(20))

        return df

    # ── 8. Cross-commodity ───────────────────────────────────────

    def _add_commodities(self, df):
        price = df["XAU_EUR"]
        eur   = df.get("EUR_USD", pd.Series(1.0, index=df.index))

        pairs = [
            ("SILVER",    "silver"),
            ("OIL_WTI",   "oil"),
            ("COPPER",    "copper"),
            ("PLATINUM",  "platinum"),
            ("PALLADIUM", "palladium"),
        ]
        for col, name in pairs:
            if col in df.columns:
                c_eur = df[col] / eur
                df[f"{name}_ret_5d"]  = np.log(c_eur / c_eur.shift(5))
                df[f"{name}_ret_20d"] = np.log(c_eur / c_eur.shift(20))

        if "SILVER" in df.columns:
            silver_eur = df["SILVER"] / eur
            df["gold_silver_ratio"] = price / (silver_eur + 1e-9)
            df["gs_ratio_zscore"]   = (df["gold_silver_ratio"] -
                                       df["gold_silver_ratio"].rolling(252).mean()) / \
                                      (df["gold_silver_ratio"].rolling(252).std() + 1e-9)

        # Commodity inflation basket
        commodity_rets = []
        for col in ["SILVER", "OIL_WTI", "COPPER"]:
            if col in df.columns:
                c = df[col] / eur
                commodity_rets.append(np.log(c / c.shift(20)))
        if commodity_rets:
            df["commodity_basket_ret"] = pd.concat(commodity_rets, axis=1).mean(axis=1)

        return df

    # ── 9. Calendar ──────────────────────────────────────────────

    def _add_calendar(self, df):
        df["day_of_week"] = df.index.dayofweek
        df["month"]       = df.index.month
        df["quarter"]     = df.index.quarter
        df["quarter_end"] = df.index.is_quarter_end.astype(int)
        df["month_end"]   = df.index.is_month_end.astype(int)
        # Option expiry proxy (3rd Friday of month is typically high vol)
        df["week_of_month"] = (df.index.day - 1) // 7
        return df

    # ── 10. Sentiment ─────────────────────────────────────────────

    def _add_sentiment(self, df, sentiment_df=None):
        if sentiment_df is not None and not sentiment_df.empty:
            df = df.join(sentiment_df[["grok_score", "news_score"]], how="left")
            df["grok_score"] = df["grok_score"].ffill(limit=3).fillna(0)
            df["news_score"] = df["news_score"].ffill(limit=3).fillna(0)
            logger.info("Sentiment features added from Grok/news.")
        else:
            df["grok_score"] = 0.0
            df["news_score"] = 0.0
        return df

    # ── Targets ──────────────────────────────────────────────────

    def _add_targets(self, df):
        price = df["XAU_EUR"]
        for horizon_name, n_days in self.config.HORIZONS.items():
            fwd = np.log(price.shift(-n_days) / price)
            df[horizon_name] = (fwd > 0).astype(int)
            df[f"{horizon_name}_return"] = fwd
        return df

    # ── Feature column selection ─────────────────────────────────

    def _get_feature_columns(self, df):
        raw_cols = set(self.config.TICKERS.keys()) | set(self.config.FRED_SERIES.keys())
        raw_cols |= {"XAU_EUR", "XAU_VOL", "XAU_USD", "EUR_USD", "EUR_CHF",
                     "USD_JPY", "SILVER", "OIL_WTI", "COPPER", "PLATINUM",
                     "PALLADIUM", "SP500", "NASDAQ", "VIX", "TIPS_10Y",
                     "HY_ETF", "IG_ETF", "BONDS_20"}
        target_cols = set(self.config.HORIZONS.keys())
        for h in self.config.HORIZONS:
            target_cols.add(f"{h}_return")
        return [c for c in df.columns
                if c not in raw_cols and c not in target_cols]

    # ── Indicator implementations ────────────────────────────────

    @staticmethod
    def _rsi(price, period=14):
        if TA_AVAILABLE:
            return ta.momentum.RSIIndicator(price, window=period).rsi()
        d = price.diff()
        g = d.clip(lower=0).rolling(period).mean()
        l = (-d.clip(upper=0)).rolling(period).mean()
        return 100 - 100 / (1 + g / (l + 1e-9))

    @staticmethod
    def _macd(price, fast, slow, signal):
        if TA_AVAILABLE:
            ind = ta.trend.MACD(price, window_fast=fast, window_slow=slow, window_sign=signal)
            return ind.macd(), ind.macd_signal()
        ef = price.ewm(span=fast, adjust=False).mean()
        es = price.ewm(span=slow, adjust=False).mean()
        m  = ef - es
        return m, m.ewm(span=signal, adjust=False).mean()

    @staticmethod
    def _bollinger(price, window, std):
        if TA_AVAILABLE:
            ind = ta.volatility.BollingerBands(price, window=window, window_dev=std)
            return ind.bollinger_hband(), ind.bollinger_lband()
        sma = price.rolling(window).mean()
        sd  = price.rolling(window).std()
        return sma + std * sd, sma - std * sd

    @staticmethod
    def _atr(high, low, close, period=14):
        if TA_AVAILABLE:
            return ta.volatility.AverageTrueRange(high, low, close, window=period).average_true_range()
        tr = pd.concat([high - low,
                        (high - close.shift(1)).abs(),
                        (low  - close.shift(1)).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _obv(price, volume):
        return (np.sign(price.diff()) * volume).cumsum()

    @staticmethod
    def _adx(price, period=14):
        alpha  = 1.0 / period
        delta  = price.diff()
        dm_p   = delta.clip(lower=0).ewm(alpha=alpha, adjust=False).mean()
        dm_m   = (-delta.clip(upper=0)).ewm(alpha=alpha, adjust=False).mean()
        atr    = (price.pct_change().abs() * price).ewm(alpha=alpha, adjust=False).mean()
        di_p   = 100 * dm_p  / (atr + 1e-9)
        di_m   = 100 * dm_m  / (atr + 1e-9)
        dx     = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
        return dx.ewm(alpha=alpha, adjust=False).mean()

    @staticmethod
    def _hurst_rolling(price, window=40, lags=None):
        if lags is None:
            lags = [2, 4, 8, 16]
        lp  = np.log(price.values + 1e-9)
        n   = len(lp)
        out = np.full(n, np.nan)
        for i in range(window, n + 1):
            chunk = lp[i - window: i]
            rs_vals = []
            for lag in lags:
                if lag >= len(chunk):
                    continue
                sub  = chunk[:lag]
                dev  = np.cumsum(sub - sub.mean())
                s    = sub.std()
                if s > 1e-10:
                    rs_vals.append((dev.max() - dev.min()) / s)
            if len(rs_vals) >= 2:
                try:
                    h = np.polyfit(np.log(lags[:len(rs_vals)]),
                                   np.log(rs_vals), 1)[0]
                    out[i - 1] = np.clip(h, 0.0, 1.0)
                except Exception:
                    pass
        return pd.Series(out, index=price.index)
