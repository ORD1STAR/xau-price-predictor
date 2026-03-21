"""
prediction/predictor.py
=======================
Step 4: Generate the final trading recommendation for today.

Output includes:
  - Overall signal: BUY / SELL / HOLD
  - Confidence score (meta-ensemble probability)
  - Multi-horizon breakdown (next day / week / 2 weeks / month)
  - Individual model votes
  - Risk/confidence context
  - Sentiment context (if Grok enabled)
  - Suggested entry/stop/target based on recent ATR
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ANSI colors for terminal output
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


class GoldPredictor:
    def __init__(self, config):
        self.config = config

    # ─────────────────────────────────────────────────────────────
    # MAIN PREDICTION
    # ─────────────────────────────────────────────────────────────

    def predict_now(
        self,
        feature_df: pd.DataFrame,
        ensembles: dict,
        feature_names: list[str],
        sentiment_result: dict = None,
    ) -> dict:
        """
        Generate the full trading recommendation for today.

        Parameters
        ----------
        feature_df      : full feature DataFrame (most recent row = today)
        ensembles       : dict[horizon -> StackingEnsemble]
        feature_names   : list of feature column names
        sentiment_result: optional dict from GrokSentimentClient.get_sentiment_today()

        Returns
        -------
        Full prediction dict with signal, probabilities, and details.
        """
        if feature_df.empty:
            logger.error("Empty feature DataFrame. Cannot predict.")
            return {}

        # Use the latest available row as "today"
        today_row = feature_df.iloc[[-1]]
        today_date = today_row.index[0].strftime("%Y-%m-%d")
        current_price = self._get_current_price(feature_df)

        logger.info(f"Predicting for {today_date} | XAU/EUR = {current_price:.2f}")

        # ── Get predictions for all horizons ─────────────────────
        horizon_results = {}
        for horizon, ensemble in ensembles.items():
            try:
                # Use last seq_len rows for deep models (need context window)
                seq_len = max(
                    getattr(self.config, "CNN_BILSTM_PARAMS", {}).get("sequence_length", 60),
                    60
                )
                context = feature_df[feature_names].iloc[-seq_len:] if len(feature_df) >= seq_len \
                          else feature_df[feature_names]

                result = ensemble.predict_last(context)
                horizon_results[horizon] = result
            except Exception as e:
                logger.error(f"Prediction failed for {horizon}: {e}")
                horizon_results[horizon] = {
                    "signal": "HOLD", "probability_up": 0.5,
                    "probability_down": 0.5, "model_votes": {}
                }

        # ── Aggregate across horizons ─────────────────────────────
        aggregate = self._aggregate_signals(horizon_results)

        # ── Risk / entry parameters ───────────────────────────────
        risk_params = self._compute_risk_params(feature_df, current_price, aggregate["signal"])

        # ── Build full result ─────────────────────────────────────
        result = {
            "date":            today_date,
            "current_price":   round(current_price, 2),
            "signal":          aggregate["signal"],
            "confidence":      aggregate["confidence"],
            "confidence_label": aggregate["confidence_label"],
            "probability_up":  aggregate["probability_up"],
            "probability_down": aggregate["probability_down"],
            "horizons":        horizon_results,
            "agreement_score": aggregate["agreement_score"],
            "risk_params":     risk_params,
            "sentiment":       sentiment_result,
            "model_weights":   self._get_meta_weights(ensembles),
        }

        return result

    # ─────────────────────────────────────────────────────────────
    # SIGNAL AGGREGATION ACROSS HORIZONS
    # ─────────────────────────────────────────────────────────────

    def _aggregate_signals(self, horizon_results: dict) -> dict:
        """
        Combine signals across horizons with horizon-weighted averaging.
        Longer horizons get more weight (less noise).
        """
        HORIZON_WEIGHTS = {
            "next_day":    0.15,
            "next_week":   0.25,
            "next_2weeks": 0.30,
            "next_month":  0.30,
        }

        total_weight = 0
        weighted_prob = 0
        n_buy = n_sell = n_hold = 0

        for horizon, result in horizon_results.items():
            w = HORIZON_WEIGHTS.get(horizon, 0.25)
            prob_up = result.get("probability_up", 0.5)
            weighted_prob += w * prob_up
            total_weight  += w

            sig = result.get("signal", "HOLD")
            if sig == "BUY":   n_buy  += 1
            elif sig == "SELL": n_sell += 1
            else:               n_hold += 1

        if total_weight > 0:
            avg_prob = weighted_prob / total_weight
        else:
            avg_prob = 0.5

        # Final signal
        buy_thr  = self.config.BUY_THRESHOLD
        sell_thr = self.config.SELL_THRESHOLD

        if avg_prob >= buy_thr:
            signal = "BUY"
        elif avg_prob <= sell_thr:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Horizon agreement score (0-1)
        n_total = len(horizon_results)
        if signal == "BUY":
            agreement = n_buy / n_total if n_total > 0 else 0
        elif signal == "SELL":
            agreement = n_sell / n_total if n_total > 0 else 0
        else:
            agreement = n_hold / n_total if n_total > 0 else 0

        # Confidence label
        if avg_prob >= 0.75 or avg_prob <= 0.25:
            confidence_label = "HIGH"
        elif avg_prob >= 0.65 or avg_prob <= 0.35:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"

        return {
            "signal":           signal,
            "probability_up":   round(float(avg_prob), 4),
            "probability_down": round(1 - float(avg_prob), 4),
            "confidence":       round(float(max(avg_prob, 1 - avg_prob)), 4),
            "confidence_label": confidence_label,
            "agreement_score":  round(float(agreement), 2),
            "n_buy_horizons":   n_buy,
            "n_sell_horizons":  n_sell,
            "n_hold_horizons":  n_hold,
        }

    # ─────────────────────────────────────────────────────────────
    # RISK PARAMETERS
    # ─────────────────────────────────────────────────────────────

    def _compute_risk_params(
        self,
        feature_df: pd.DataFrame,
        current_price: float,
        signal: str,
    ) -> dict:
        """Compute entry, stop-loss, take-profit based on recent ATR."""
        try:
            # Get recent ATR (normalized -> convert back to price)
            atr_pct = feature_df["atr_14"].iloc[-5:].mean()
            atr_abs = atr_pct * current_price if atr_pct < 1 else atr_pct
            atr_abs = max(atr_abs, current_price * 0.005)  # Minimum 0.5% ATR

            if signal == "BUY":
                entry  = current_price
                stop   = round(entry - 1.5 * atr_abs, 2)
                target = round(entry + 3.0 * atr_abs, 2)  # 2:1 R:R
                rr_ratio = round((target - entry) / (entry - stop + 1e-6), 2)
            elif signal == "SELL":
                entry  = current_price
                stop   = round(entry + 1.5 * atr_abs, 2)
                target = round(entry - 3.0 * atr_abs, 2)
                rr_ratio = round((entry - target) / (stop - entry + 1e-6), 2)
            else:
                entry = stop = target = current_price
                rr_ratio = 0.0

            return {
                "entry_price":    round(entry, 2),
                "stop_loss":      round(stop,  2),
                "take_profit":    round(target, 2),
                "atr_14":         round(atr_abs, 2),
                "risk_pct":       round(abs(entry - stop) / entry * 100, 3),
                "reward_pct":     round(abs(target - entry) / entry * 100, 3),
                "risk_reward":    rr_ratio,
            }
        except Exception as e:
            logger.warning(f"Risk params computation failed: {e}")
            return {}

    # ─────────────────────────────────────────────────────────────
    # FORMATTING / DISPLAY
    # ─────────────────────────────────────────────────────────────

    def format_prediction(self, result: dict) -> str:
        """Return a formatted string report of the prediction."""
        if not result:
            return "No prediction available."

        sig   = result["signal"]
        color = GREEN if sig == "BUY" else (RED if sig == "SELL" else YELLOW)
        conf  = result.get("confidence_label", "")
        prob  = result.get("probability_up", 0.5)
        price = result.get("current_price", 0)
        risk  = result.get("risk_params", {})
        sent  = result.get("sentiment") or {}
        hor   = result.get("horizons", {})

        lines = [
            "",
            f"{BOLD}{CYAN}{'═'*60}{RESET}",
            f"{BOLD}{CYAN}  XAU/EUR GOLD PREDICTION -- {result['date']}{RESET}",
            f"{BOLD}{CYAN}{'═'*60}{RESET}",
            "",
            f"  {'Current Price':20} {BOLD}€{price:,.2f}{RESET}",
            "",
            f"  {'Signal':20} {color}{BOLD}{sig}{RESET}  [{conf} CONFIDENCE]",
            f"  {'Prob(UP)':20} {prob:.1%}",
            f"  {'Prob(DOWN)':20} {1-prob:.1%}",
            f"  {'Horizon Agreement':20} {result.get('agreement_score', 0):.0%}",
            "",
            f"{BOLD}  ─── Multi-Horizon Breakdown ─────────────────────{RESET}",
        ]

        horizon_labels = {
            "next_day":    "Next Day",
            "next_week":   "Next Week (5d)",
            "next_2weeks": "Next 2 Weeks (10d)",
            "next_month":  "Next Month (22d)",
        }

        for h, label in horizon_labels.items():
            if h in hor:
                h_sig  = hor[h].get("signal", "HOLD")
                h_prob = hor[h].get("probability_up", 0.5)
                h_col  = GREEN if h_sig == "BUY" else (RED if h_sig == "SELL" else YELLOW)
                lines.append(f"  {'  '+label:22} {h_col}{h_sig:6}{RESET}  (prob UP: {h_prob:.1%})")

        if risk:
            lines += [
                "",
                f"{BOLD}  ─── Trade Parameters ────────────────────────────{RESET}",
                f"  {'Entry':20} €{risk.get('entry_price', 0):,.2f}",
                f"  {'Stop Loss':20} €{risk.get('stop_loss', 0):,.2f}  ({risk.get('risk_pct', 0):.2f}% risk)",
                f"  {'Take Profit':20} €{risk.get('take_profit', 0):,.2f}  ({risk.get('reward_pct', 0):.2f}% gain)",
                f"  {'R:R Ratio':20} {risk.get('risk_reward', 0):.1f}:1",
                f"  {'ATR(14)':20} €{risk.get('atr_14', 0):,.2f}",
            ]

        # Model votes
        model_votes = {}
        for h, hr in hor.items():
            for m, v in hr.get("model_votes", {}).items():
                if m not in model_votes:
                    model_votes[m] = []
                model_votes[m].append(v)

        if model_votes:
            lines += [
                "",
                f"{BOLD}  ─── Individual Model Votes (next_day) ──────────{RESET}",
            ]
            day_votes = hor.get("next_day", {}).get("model_votes", {})
            for model_name, p in day_votes.items():
                bar_len = int(p * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                model_col = GREEN if p >= 0.6 else (RED if p <= 0.4 else YELLOW)
                lines.append(f"  {'  '+model_name:22} {model_col}{bar}{RESET} {p:.1%}")

        # Sentiment
        if sent and sent.get("grok_score") is not None and sent.get("grok_score") != 0.0:
            gs = sent["grok_score"]
            gs_label = "BULLISH 📈" if gs > 0.2 else ("BEARISH 📉" if gs < -0.2 else "NEUTRAL ➡️")
            lines += [
                "",
                f"{BOLD}  ─── Sentiment (Grok/Twitter) ────────────────────{RESET}",
                f"  {'Score':20} {gs:+.2f}  {gs_label}",
                f"  {'Reasoning':20} {sent.get('reasoning', '')[:60]}",
            ]

        lines += [
            "",
            f"{BOLD}{CYAN}{'═'*60}{RESET}",
            f"  {YELLOW}⚠ Not financial advice. Always use proper risk management.{RESET}",
            f"{BOLD}{CYAN}{'═'*60}{RESET}",
            "",
        ]

        return "\n".join(lines)

    def print_prediction(self, result: dict):
        print(self.format_prediction(result))

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _get_current_price(self, feature_df: pd.DataFrame) -> float:
        if "XAU_EUR" in feature_df.columns:
            return float(feature_df["XAU_EUR"].iloc[-1])
        return 0.0

    def _get_meta_weights(self, ensembles: dict) -> dict:
        weights = {}
        for horizon, ens in ensembles.items():
            try:
                weights[horizon] = ens.get_meta_weights()
            except Exception:
                pass
        return weights
