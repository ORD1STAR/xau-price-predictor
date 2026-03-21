"""
data/fetcher.py
===============
Fetches XAU/EUR price data and all macro/feature data from free APIs.

Sources used (all 100% free, no API key required):
  - yfinance  : Gold, Silver, Oil, Copper, VIX, ETFs, FX pairs
  - pandas-datareader (FRED): Treasury yields, DXY, Fed/ECB rates

XAU/EUR is constructed as:  XAU/EUR = XAU/USD ÷ EUR/USD
"""

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# FRED helper (graceful fallback if unavailable)
# ─────────────────────────────────────────────────────────────────

def _fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a FRED series with pandas-datareader. Returns NaN series on failure."""
    try:
        import pandas_datareader.data as web
        df = web.DataReader(series_id, "fred", start, end)
        s = df.iloc[:, 0].rename(series_id)
        return s
    except Exception as e:
        logger.warning(f"FRED {series_id} unavailable: {e}. Filling with NaN.")
        idx = pd.date_range(start, end, freq="B")
        return pd.Series(np.nan, index=idx, name=series_id)


# ─────────────────────────────────────────────────────────────────
# MAIN DATA FETCHER
# ─────────────────────────────────────────────────────────────────

class GoldDataFetcher:
    """
    Fetches and assembles all raw data needed for feature engineering.
    Returns a single wide DataFrame indexed by business day.
    """

    def __init__(self, config, cache_dir: Path = None):
        self.config = config
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ──────────────────────────────────────────────────

    def fetch_all(
        self,
        start: str = None,
        end: str = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch everything and return a merged, forward-filled DataFrame.

        Returns columns:
          XAU_USD, EUR_USD, XAU_EUR (target),
          EUR_CHF, SILVER, OIL_WTI, COPPER, PLATINUM,
          SP500, VIX, TIPS_10Y, HY_ETF, IG_ETF,
          US_10Y, US_2Y, US_REAL10, DXY, FED_RATE, ECB_RATE
        """
        start = start or self.config.TRAIN_START
        end   = end   or self.config.TEST_END

        cache_file = self.cache_dir / f"raw_{start}_{end}.parquet"
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached data from {cache_file}")
            return pd.read_parquet(cache_file)

        logger.info(f"Fetching data {start} -> {end} ...")

        # 1) yfinance tickers
        yf_df = self._fetch_yfinance(start, end)

        # 2) FRED macro series
        fred_df = self._fetch_fred_all(start, end)

        # 3) Merge on business-day index
        df = pd.concat([yf_df, fred_df], axis=1)
        df = df.resample("B").last()  # business days only

        # --- Forward-fill strategy ---
        # yfinance data (prices): short fill only (up to 5 days for weekends/holidays)
        yf_cols  = [c for c in df.columns if c in list(self.config.TICKERS.keys()) + ["XAU_VOL"]]
        fred_cols = [c for c in df.columns if c in list(self.config.FRED_SERIES.keys())]

        if yf_cols:
            df[yf_cols] = df[yf_cols].ffill(limit=5)
        if fred_cols:
            # FRED series are monthly (Fed rate, ECB rate, DXY, yields).
            # A rate decision is valid until the next one -- carry it forward fully.
            df[fred_cols] = df[fred_cols].ffill(limit=None).bfill(limit=5)

        # 4) Construct XAU/EUR
        df["XAU_EUR"] = df["XAU_USD"] / df["EUR_USD"]

        # 5) Volume proxy: use gold ETF GLD volume
        if "GLD_VOL" in df.columns:
            df["XAU_VOLUME"] = df["GLD_VOL"]

        # 6) Drop rows before 2012 (sparse data)
        df = df[df.index >= pd.Timestamp(start)]
        df = df[df.index <= pd.Timestamp(end)]

        # 7) Drop rows where XAU_EUR is missing
        df = df.dropna(subset=["XAU_EUR"])

        logger.info(f"Dataset shape: {df.shape} | {df.index[0].date()} -> {df.index[-1].date()}")

        df.to_parquet(cache_file)
        logger.info(f"Saved to {cache_file}")
        return df

    def fetch_latest(self, n_days: int = 120) -> pd.DataFrame:
        """Fetch the most recent N calendar days (for real-time prediction)."""
        end   = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=n_days + 30)).strftime("%Y-%m-%d")
        return self.fetch_all(start=start, end=end, use_cache=False)

    # ── Private ─────────────────────────────────────────────────

    def _fetch_yfinance(self, start: str, end: str) -> pd.DataFrame:
        """Download all yfinance tickers and extract Adj Close + Volume for gold."""
        tickers = self.config.TICKERS
        all_frames = {}

        # Group download for efficiency
        ticker_list = list(tickers.values())
        logger.info(f"Downloading {len(ticker_list)} yfinance tickers ...")

        try:
            raw = yf.download(
                ticker_list,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.error(f"yfinance bulk download failed: {e}. Trying one-by-one ...")
            raw = self._fetch_yfinance_individual(tickers, start, end)
            return raw

        # Extract Close prices
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]

        # Rename columns to our logical names
        reverse_map = {v: k for k, v in tickers.items()}
        close.columns = [reverse_map.get(c, c) for c in close.columns]

        # Grab gold volume separately
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                vol = raw["Volume"][tickers["XAU_USD"]].rename("XAU_VOL")
            else:
                vol = raw["Volume"].rename("XAU_VOL")
            close = pd.concat([close, vol], axis=1)
        except Exception:
            pass

        return close

    def _fetch_yfinance_individual(self, tickers: dict, start: str, end: str) -> pd.DataFrame:
        """Fallback: fetch each ticker individually."""
        frames = {}
        for name, symbol in tickers.items():
            try:
                df = yf.download(symbol, start=start, end=end,
                                 auto_adjust=True, progress=False)
                if not df.empty:
                    frames[name] = df["Close"]
                    if name == "XAU_USD":
                        frames["XAU_VOL"] = df["Volume"]
            except Exception as e:
                logger.warning(f"Failed to download {symbol}: {e}")
        return pd.DataFrame(frames)

    def _fetch_fred_all(self, start: str, end: str) -> pd.DataFrame:
        """Fetch all FRED series and return as DataFrame."""
        frames = {}
        for name, series_id in self.config.FRED_SERIES.items():
            s = _fetch_fred(series_id, start, end)
            frames[name] = s
            logger.debug(f"FRED {name} ({series_id}): {s.count()} data points")
        return pd.DataFrame(frames)

    # ── Utility ─────────────────────────────────────────────────

    @staticmethod
    def describe(df: pd.DataFrame) -> pd.DataFrame:
        """Print a concise data quality report."""
        report = pd.DataFrame({
            "dtype":    df.dtypes,
            "non_null": df.count(),
            "null_pct": (df.isnull().mean() * 100).round(2),
            "min":      df.min().round(4),
            "max":      df.max().round(4),
            "mean":     df.mean().round(4),
            "std":      df.std().round(4),
        })
        return report
