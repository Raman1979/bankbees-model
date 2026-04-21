"""
bankbees_scalper.py
--------------------
BankBeesScalper — ML signal engine for BANKBEES ETF.
Drop this file into your existing bot folder alongside:
  bot.py, strategy.py, data_feed.py, order_manager.py, risk_manager.py

Model path: /home/ubuntu/models/BANKBEES_5min_model.pkl  (EC2 local)
"""

import pandas as pd
import numpy as np
import ta
import joblib
import logging
from datetime import datetime, time
from pathlib import Path

logger = logging.getLogger(__name__)


class BankBeesScalper:
    """
    ML-based scalper for BANKBEES ETF on 5-min candles.
    Plugs into your existing Fyers bot modules.
    """

    # ── Signal Params ─────────────────────────────────────────────────────
    THRESHOLD = 0.68    # min model probability to fire BUY signal
    TARGET    = 0.002   # 0.20% profit target
    SL        = 0.0015  # 0.15% stop-loss
    COST      = 0.0004  # 0.04% round-trip cost

    # ── Time Filter (IST) ─────────────────────────────────────────────────
    TRADE_START = time(9, 20)   # skip first 20 min (gap/volatility risk)
    TRADE_END   = time(14, 55)  # stop 35 min before close

    # ── Fyers Symbol ─────────────────────────────────────────────────────
    SYMBOL = "NSE:BANKBEES-EQ"

    def __init__(self, model_path: str = "/home/ubuntu/models/BANKBEES_5min_model.pkl", fyers=None):
        self.model_path = Path(model_path)
        self.model = None
        self.fyers = fyers   # Live balance ਲਈ
        self._load_model()
        self._last_signal_time = None   # prevent duplicate signals same candle

    # ─────────────────────────────────────────────────────────────────────
    # Model
    # ─────────────────────────────────────────────────────────────────────

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[BankBeesScalper] Model not found: {self.model_path}\n"
                f"Train & save:  joblib.dump(model, '{self.model_path}')"
            )
        self.model = joblib.load(self.model_path)
        logger.info(f"[BankBeesScalper] Model loaded ✓ ({self.model_path})")

    def reload_model(self):
        """Hot-reload model without restarting bot (call after retraining)."""
        self._load_model()

    # ─────────────────────────────────────────────────────────────────────
    # Feature Engineering  (must match training pipeline exactly)
    # ─────────────────────────────────────────────────────────────────────

    def create_features(self, df_5min: pd.DataFrame) -> pd.DataFrame:
        """
        Build all features from 5-min OHLCV DataFrame.

        Fyers column names expected:
            timestamp (epoch int OR datetime str), open, high, low, close, volume

        Returns enriched DataFrame (same rows).
        """
        df = df_5min.copy()

        # Normalise Fyers timestamp (epoch seconds → datetime IST)
        if pd.api.types.is_integer_dtype(df['timestamp']):
            df['datetime'] = (pd.to_datetime(df['timestamp'], unit='s')
                                .dt.tz_localize('UTC')
                                .dt.tz_convert('Asia/Kolkata')
                                .dt.tz_localize(None))
        else:
            df['datetime'] = pd.to_datetime(df['timestamp'])

        df = df.sort_values('datetime').reset_index(drop=True)

        # ── Candle Anatomy ────────────────────────────────────────────
        df['returns']    = df['close'].pct_change()
        df['body']       = df['close'] - df['open']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range']      = df['high'] - df['low']

        # ── Momentum ──────────────────────────────────────────────────
        df['rsi']     = ta.momentum.RSIIndicator(df['close'], 14).rsi()
        df['stoch_k'] = ta.momentum.StochasticOscillator(
                            df['high'], df['low'], df['close']).stoch()
        df['macd']    = ta.trend.MACD(df['close']).macd_diff()

        # ── Trend ─────────────────────────────────────────────────────
        df['ema_9']     = ta.trend.EMAIndicator(df['close'], 9).ema_indicator()
        df['ema_21']    = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
        df['ema_cross'] = (df['ema_9'] > df['ema_21']).astype(int)

        # ── Volatility ────────────────────────────────────────────────
        df['atr']     = ta.volatility.AverageTrueRange(
                            df['high'], df['low'], df['close']).average_true_range()
        df['atr_pct'] = df['atr'] / df['close'] * 100
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_pos']   = ((df['close'] - df['bb_lower']) /
                          (df['bb_upper'] - df['bb_lower']).replace(0, np.nan))

        # ── Volume ────────────────────────────────────────────────────
        df['vol_sma20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma20']

        # ── Time Features ─────────────────────────────────────────────
        df['hour']          = df['datetime'].dt.hour
        df['minute']        = df['datetime'].dt.minute
        df['dayofweek']     = df['datetime'].dt.dayofweek
        df['mins_to_close'] = (15*60 + 30) - (df['hour']*60 + df['minute'])

        # ── Lagged Features ───────────────────────────────────────────
        for i in range(1, 4):
            df[f'close_lag_{i}'] = df['close'].shift(i)
            df[f'vol_lag_{i}']   = df['volume'].shift(i)
            df[f'range_lag_{i}'] = df['range'].shift(i)
            df[f'high_lag_{i}']  = df['high'].shift(i)

        df['momentum_3'] = df['close'] - df['close_lag_3']
        df['high_3']     = df[['high', 'high_lag_1', 'high_lag_2']].max(axis=1)

        df = df.fillna(0)
        return df

    # ─────────────────────────────────────────────────────────────────────
    # Signal Generation
    # ─────────────────────────────────────────────────────────────────────

    _EXCLUDE_COLS = {
        'datetime', 'date_only', 'label', 'future_high', 'future_low',
        'future_return', 'open', 'high', 'low', 'close',
        'volume', 'timestamp', 'bb_upper', 'bb_lower'
    }

    def predict_signal(self, df_5min: pd.DataFrame) -> dict | None:
        """
        Main entry point — call every time a new 5-min candle closes.

        Args:
            df_5min: DataFrame with AT LEAST 30 rows of 5-min OHLCV from Fyers.
                     Must have columns: timestamp, open, high, low, close, volume

        Returns:
            dict with trade details  OR  None (no signal / filter blocked it).

        Signal dict keys:
            symbol, signal, entry_price, target, sl, prob,
            time, exit_time, qty
        """
        if len(df_5min) < 30:
            logger.warning("[BankBeesScalper] Need >=30 candles, got %d", len(df_5min))
            return None

        df   = self.create_features(df_5min)
        last = df.iloc[-1]
        now  = last['datetime']

        # Prevent duplicate signal on same candle
        if self._last_signal_time == now:
            return None

        # ── Pre-filters ───────────────────────────────────────────────
        atr_ok  = 0.07 <= last['atr_pct'] <= 0.40
        ema_ok  = last['ema_cross'] == 1
        rsi_ok  = 40 <= last['rsi'] <= 75
        time_ok = self.TRADE_START <= now.time() <= self.TRADE_END
        mins_ok = last['mins_to_close'] >= 30

        if not all([atr_ok, ema_ok, rsi_ok, time_ok, mins_ok]):
            logger.debug(
                "[BankBeesScalper] Filters → atr:%s ema:%s rsi:%s time:%s mins:%s",
                atr_ok, ema_ok, rsi_ok, time_ok, mins_ok
            )
            return None

        # ── Model Prediction ──────────────────────────────────────────
        feat_cols = [c for c in df.columns if c not in self._EXCLUDE_COLS]
        X         = df[feat_cols].iloc[[-1]]
        prob      = self.model.predict_proba(X)[0, 1]

        logger.info("[BankBeesScalper] prob=%.3f threshold=%.2f @ %s",
                    prob, self.THRESHOLD, now)

        if prob <= self.THRESHOLD:
            return None

        # ── Assemble Signal ───────────────────────────────────────────
        entry = round(float(last['close']), 2)
        self._last_signal_time = now

        signal = {
            'symbol'     : self.SYMBOL,
            'signal'     : 'BUY',
            'entry_price': entry,
            'target'     : round(entry * (1 + self.TARGET), 2),
            'sl'         : round(entry * (1 - self.SL), 2),
            'prob'       : round(prob * 100, 1),
            'time'       : now,
            'exit_time'  : now + pd.Timedelta(minutes=30),
            'qty'        : self.calc_qty(entry),
        }

        logger.info("[BankBeesScalper] SIGNAL FIRED >>> %s", signal)
        return signal

    # ─────────────────────────────────────────────────────────────────────
    # Position Sizing
    # ─────────────────────────────────────────────────────────────────────

    def calc_qty(self, price: float,
                 capital: float = None,
                 risk_pct: float = 0.5) -> int:
        """
        Live Balance ਤੋਂ Capital ਲੈ ਕੇ QTY calculate ਕਰੋ।
        Fyers Available Balance ਦਾ 90% ਵਰਤੋ।
        """
        # ── Live Balance ਤੋਂ Capital ਲਓ ──────────────────────────────
        if capital is None:
            try:
                funds = self.fyers.funds()
                available = funds['fund_limit'][9]['equityAmount']  # Available Balance
                capital = max(available, 0)
                logger.info(f"[BankBeesScalper] Live Balance: ₹{capital:.0f}")
            except Exception as e:
                logger.warning(f"[BankBeesScalper] Balance fetch failed: {e} — using ₹10,000")
                capital = 10_000  # fallback

        # ── 90% Balance ਵਰਤੋ (10% reserve) ──────────────────────────
        usable = capital * 0.90

        # ── Risk based QTY ───────────────────────────────────────────
        risk_amt     = capital * risk_pct / 100
        qty_by_risk  = int(risk_amt / (price * self.SL))

        # ── Price based QTY (ਪੂਰਾ balance ਨਾ ਵਰਤੋ) ─────────────────
        qty_by_bal   = int(usable / price)

        # ── ਦੋਵਾਂ ਵਿੱਚੋਂ ਘੱਟ ਲਓ ─────────────────────────────────────
        final_qty = min(qty_by_risk, qty_by_bal)

        logger.info(
            f"[BankBeesScalper] Capital:₹{capital:.0f} "
            f"Usable:₹{usable:.0f} "
            f"QTY_risk:{qty_by_risk} "
            f"QTY_bal:{qty_by_bal} "
            f"Final QTY:{final_qty}"
        )
        return max(final_qty, 1)
