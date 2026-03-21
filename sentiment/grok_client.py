"""
sentiment/grok_client.py
========================
Fetches political/news sentiment from the Grok API (xAI) via Twitter/X.

The Grok API is OpenAI-compatible, so we use the openai SDK.
Set GROK_ENABLED = True and GROK_API_KEY in config.py to activate.

When disabled, this module returns a neutral score (0.0) for all dates,
so the rest of the pipeline works with zero changes.

Sentiment score scale:
  -1.0 = very bearish for gold (dollar strength, risk-on, peace)
   0.0 = neutral
  +1.0 = very bullish for gold (inflation, war, rate cuts, fear)

To get an xAI Grok API key: https://x.ai/api
"""

import logging
import re
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════

SCORING_SYSTEM_PROMPT = """You are a quantitative gold market analyst. 
Your task is to analyze news headlines and tweets about macroeconomics, 
monetary policy, geopolitics, and financial markets, and score their 
collective impact on gold (XAU) prices.

Score on a scale from -1.0 to +1.0:
  +1.0: Extremely bullish for gold 
        (e.g., war outbreak, hyperinflation, Fed pivots to cuts, dollar crash, banking crisis)
  +0.5: Moderately bullish 
        (e.g., inflation above expectations, geopolitical tensions, dovish Fed signals)
   0.0: Neutral 
        (e.g., mixed economic data, sideways dollar, no major news)
  -0.5: Moderately bearish 
        (e.g., dollar strength, risk-on sentiment, rate hike expectations)
  -1.0: Extremely bearish for gold 
        (e.g., aggressive rate hikes, peace deals, strong jobs data, dollar surge)

Respond ONLY with a JSON object: {"score": <float>, "reasoning": "<one sentence>"}
Do not include any other text."""

SCORING_USER_TEMPLATE = """Date: {date}
Recent headlines and tweets about gold, the dollar, monetary policy, and geopolitics:

{headlines}

Score the collective impact on gold prices:"""


# ═══════════════════════════════════════════════════════════════════
# GROK CLIENT
# ═══════════════════════════════════════════════════════════════════

class GrokSentimentClient:
    """
    Fetches and scores market sentiment using Grok (xAI) on Twitter/X data.
    
    Two modes:
      Live  : calls Grok API (requires GROK_API_KEY and GROK_ENABLED=True)
      Mock  : returns neutral scores (for development / when no key)
    """

    def __init__(self, config):
        self.config  = config
        self.enabled = config.GROK_ENABLED
        self._client = None

        if self.enabled:
            self._init_client()

    def _init_client(self):
        """Initialize the xAI OpenAI-compatible client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key  = self.config.GROK_API_KEY,
                base_url = self.config.GROK_BASE_URL,
            )
            logger.info("Grok API client initialized.")
        except Exception as e:
            logger.error(f"Grok client init failed: {e}. Disabling sentiment.")
            self.enabled = False

    # ─────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────

    def get_sentiment_today(self) -> dict:
        """
        Fetch and score today's sentiment.
        
        Returns
        -------
        dict with keys:
          date        : str  (today)
          grok_score  : float in [-1, 1]
          news_score  : float in [-1, 1] (same as grok_score for now)
          reasoning   : str
          headlines   : list[str]
        """
        date = datetime.today().strftime("%Y-%m-%d")
        return self.get_sentiment_for_date(date)

    def get_sentiment_for_date(self, date: str) -> dict:
        """Score sentiment for a specific date."""
        if not self.enabled:
            return self._mock_result(date)

        headlines = self._fetch_headlines(date)
        score, reasoning = self._score_headlines(headlines, date)

        return {
            "date":       date,
            "grok_score": score,
            "news_score": score,
            "reasoning":  reasoning,
            "headlines":  headlines,
        }

    def build_sentiment_df(
        self,
        start: str,
        end: str,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Build a daily sentiment DataFrame for a date range.
        Used to inject sentiment as features during training.
        
        For historical data: scores are approximated from available
        financial news (Grok can search X/Twitter historical content).
        
        Returns DataFrame with index=date, columns=[grok_score, news_score]
        """
        dates = pd.date_range(start, end, freq="B")  # business days
        records = []

        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")
            if verbose and len(records) % 50 == 0:
                logger.info(f"  Scoring sentiment {date_str} ...")
            result = self.get_sentiment_for_date(date_str)
            records.append({
                "date":       dt,
                "grok_score": result["grok_score"],
                "news_score": result["news_score"],
            })

        df = pd.DataFrame(records).set_index("date")
        return df

    # ─────────────────────────────────────────────────────────────
    # HEADLINE FETCHING
    # ─────────────────────────────────────────────────────────────

    def _fetch_headlines(self, date: str) -> list[str]:
        """
        Use Grok to search X/Twitter for gold-relevant headlines on a date.
        Grok has native X/Twitter search access.
        """
        if not self.enabled or self._client is None:
            return []

        headlines = []
        cfg = self.config

        for query in cfg.GROK_SEARCH_QUERIES[:4]:  # Limit API calls
            try:
                response = self._client.chat.completions.create(
                    model=cfg.GROK_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                f"Search X/Twitter for posts on {date} about: {query}. "
                                f"Return the top {cfg.GROK_TWEETS_PER_QUERY // 4} "
                                f"most relevant headlines as a simple numbered list. "
                                f"Focus on financial/market implications only."
                            ),
                        }
                    ],
                    max_tokens=500,
                    temperature=0.1,
                )
                text = response.choices[0].message.content
                # Extract lines as headlines
                lines = [
                    re.sub(r"^\d+\.\s*", "", line).strip()
                    for line in text.split("\n")
                    if line.strip() and len(line.strip()) > 20
                ]
                headlines.extend(lines[:5])
            except Exception as e:
                logger.debug(f"Headline fetch for '{query}' failed: {e}")

        return headlines[:20]  # Cap at 20 headlines

    # ─────────────────────────────────────────────────────────────
    # SENTIMENT SCORING
    # ─────────────────────────────────────────────────────────────

    def _score_headlines(
        self,
        headlines: list[str],
        date: str,
    ) -> tuple[float, str]:
        """
        Send headlines to Grok for scoring.
        Returns (score: float, reasoning: str)
        """
        if not headlines or not self.enabled:
            return 0.0, "No headlines available."

        headlines_text = "\n".join(f"- {h}" for h in headlines)
        prompt = SCORING_USER_TEMPLATE.format(date=date, headlines=headlines_text)

        try:
            import json
            response = self._client.chat.completions.create(
                model=self.config.GROK_MODEL,
                messages=[
                    {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=100,
                temperature=0.0,  # Deterministic scoring
            )
            text = response.choices[0].message.content.strip()
            # Parse JSON response
            data = json.loads(text)
            score = float(data.get("score", 0.0))
            score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
            reasoning = str(data.get("reasoning", ""))
            return score, reasoning

        except Exception as e:
            logger.warning(f"Sentiment scoring failed for {date}: {e}")
            return 0.0, f"Scoring error: {e}"

    # ─────────────────────────────────────────────────────────────
    # MOCK / FALLBACK
    # ─────────────────────────────────────────────────────────────

    def _mock_result(self, date: str) -> dict:
        """Return neutral mock result when Grok is disabled."""
        return {
            "date":       date,
            "grok_score": 0.0,
            "news_score": 0.0,
            "reasoning":  "Grok API disabled. Set GROK_ENABLED=True in config.py.",
            "headlines":  [],
        }


# ═══════════════════════════════════════════════════════════════════
# MANUAL HEADLINE INJECTOR (alternative to Grok API)
# ═══════════════════════════════════════════════════════════════════

class ManualSentimentInjector:
    """
    Allows manual injection of news/events into the sentiment stream.
    Use this while you set up the Grok API, or to add known events.

    Example
    -------
    injector = ManualSentimentInjector()
    injector.add("2024-03-20", score=0.8, note="Fed signals rate cut in June")
    injector.add("2024-01-12", score=-0.5, note="Strong US jobs data")
    df = injector.to_dataframe("2024-01-01", "2024-12-31")
    """

    KNOWN_EVENTS = {
        # Historical events with known gold impact
        "2022-02-24": (+0.9, "Russia invades Ukraine -- massive gold spike"),
        "2022-03-16": (-0.4, "Fed begins aggressive rate hike cycle"),
        "2023-03-10": (+0.7, "Silicon Valley Bank collapse -- banking crisis fear"),
        "2023-07-26": (-0.3, "Fed hikes to 5.25-5.50% -- highest since 2001"),
        "2024-01-01": (+0.2, "New year risk-off positioning"),
        "2024-09-18": (+0.6, "Fed first rate cut -- 50bp surprise"),
        "2024-11-05": (-0.3, "Trump election uncertainty mixed signals"),
        "2025-01-20": (-0.1, "Trump inauguration -- dollar strength initially"),
    }

    def __init__(self):
        self._events: dict[str, tuple[float, str]] = dict(self.KNOWN_EVENTS)

    def add(self, date: str, score: float, note: str = ""):
        score = max(-1.0, min(1.0, score))
        self._events[date] = (score, note)

    def to_dataframe(self, start: str, end: str) -> pd.DataFrame:
        """
        Build a daily sentiment DataFrame, interpolating between known events.
        Days without known events get 0.0 (neutral).
        """
        dates = pd.date_range(start, end, freq="B")
        records = []
        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")
            score, note = self._events.get(date_str, (0.0, ""))
            records.append({
                "date":       dt,
                "grok_score": score,
                "news_score": score,
            })

        df = pd.DataFrame(records).set_index("date")

        # Smooth: non-zero events decay over 3 days
        df["grok_score"] = df["grok_score"].replace(0.0, np.nan)
        df["grok_score"] = df["grok_score"].ffill(limit=3).fillna(0.0)
        df["news_score"] = df["grok_score"]
        return df
