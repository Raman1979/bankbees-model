# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# strategy.py — 11 Strategies + BankBees ML Scalper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import pandas as pd
import numpy as np
from indicators import calc_ema, calc_rsi, calc_macd, calc_bollinger, calc_vwap, calc_gap
from bankbees_scalper import BankBeesScalper
from data_feed import get_bankbees_5min

# ── Base ──────────────────────────────────────────────
class BaseStrategy:
    name = "base"
    description = ""
    def signal(self, df: pd.DataFrame) -> str:
        raise NotImplementedError

# ── 1. EMA Crossover (Improved) ───────────────────────
class EmaCrossover(BaseStrategy):
    name = "ema_crossover"
    description = "EMA Fast/Slow Crossover + RSI Confirm"
    def __init__(self, fast=9, slow=21):
        self.fast = fast
        self.slow = slow
    def signal(self, df):
        if len(df) < self.slow + 5: return "HOLD"
        ema_f = calc_ema(df, self.fast)
        ema_s = calc_ema(df, self.slow)
        rsi   = calc_rsi(df)
        pf, cf = ema_f.iloc[-2], ema_f.iloc[-1]
        ps, cs = ema_s.iloc[-2], ema_s.iloc[-1]
        rsi_now = rsi.iloc[-1]
        crossed_up   = pf < ps and cf > cs
        crossed_down = pf > ps and cf < cs
        trend_bull = cf > cs and rsi_now > 55
        trend_bear = cf < cs and rsi_now < 45
        mom_up = cf > pf and cs > ps
        mom_dn = cf < pf and cs < ps
        if crossed_up   or (trend_bull and mom_up): return "BUY"
        if crossed_down or (trend_bear and mom_dn):  return "SELL"
        return "HOLD"

# ── 2. RSI Reversal ───────────────────────────────────
class RsiReversal(BaseStrategy):
    name = "rsi_reversal"
    description = "RSI Oversold/Overbought Reversal"
    def __init__(self, oversold=30, overbought=70):
        self.oversold = oversold
        self.overbought = overbought
    def signal(self, df):
        if len(df) < 20: return "HOLD"
        rsi = calc_rsi(df).iloc[-1]
        if pd.isna(rsi): return "HOLD"
        if rsi < self.oversold:   return "BUY"
        if rsi > self.overbought: return "SELL"
        return "HOLD"

# ── 3. EMA + RSI Combined ────────────────────────────
class EmaRsiCombined(BaseStrategy):
    name = "ema_rsi_combined"
    description = "EMA Trend + RSI Dual Filter"
    def __init__(self, fast=9, slow=21, rsi_low=40, rsi_high=60):
        self.fast = fast; self.slow = slow
        self.rsi_low = rsi_low; self.rsi_high = rsi_high
    def signal(self, df):
        if len(df) < self.slow + 5: return "HOLD"
        ef  = calc_ema(df, self.fast).iloc[-1]
        es  = calc_ema(df, self.slow).iloc[-1]
        rsi = calc_rsi(df).iloc[-1]
        if pd.isna(rsi): return "HOLD"
        if ef > es and rsi > self.rsi_high: return "BUY"
        if ef < es and rsi < self.rsi_low:  return "SELL"
        return "HOLD"

# ── 4. Bollinger Breakout ────────────────────────────
class BollingerBreakout(BaseStrategy):
    name = "bollinger_breakout"
    description = "Bollinger Bands Breakout"
    def signal(self, df):
        if len(df) < 25: return "HOLD"
        upper, mid, lower = calc_bollinger(df, 20, 2.0)
        price = df["close"].iloc[-1]
        if price > upper.iloc[-1]: return "SELL"
        if price < lower.iloc[-1]: return "BUY"
        return "HOLD"

# ── 5. MACD Crossover ────────────────────────────────
class MacdCrossover(BaseStrategy):
    name = "macd_crossover"
    description = "MACD Line Crosses Signal Line"
    def signal(self, df):
        if len(df) < 35: return "HOLD"
        macd, sig, _ = calc_macd(df)
        if macd.iloc[-2] < sig.iloc[-2] and macd.iloc[-1] > sig.iloc[-1]: return "BUY"
        if macd.iloc[-2] > sig.iloc[-2] and macd.iloc[-1] < sig.iloc[-1]: return "SELL"
        return "HOLD"

# ── 6. EMA + MACD Combined ───────────────────────────
class EmaMacdCombined(BaseStrategy):
    name = "ema_macd_combined"
    description = "EMA + MACD + RSI Triple Confirm"
    def __init__(self, fast=9, slow=21):
        self.fast = fast; self.slow = slow
    def signal(self, df):
        if len(df) < 35: return "HOLD"
        ema_f        = calc_ema(df, self.fast).iloc[-1]
        ema_s        = calc_ema(df, self.slow).iloc[-1]
        macd, sig, _ = calc_macd(df)
        rsi          = calc_rsi(df).iloc[-1]
        if pd.isna(rsi): return "HOLD"
        if ema_f > ema_s and macd.iloc[-1] > sig.iloc[-1] and rsi > 50: return "BUY"
        if ema_f < ema_s and macd.iloc[-1] < sig.iloc[-1] and rsi < 50: return "SELL"
        return "HOLD"

# ── 7. VWAP + EMA9 ───────────────────────────────────
class VwapEma9(BaseStrategy):
    name = "vwap_ema9"
    description = "VWAP + EMA9 Position Confirm"
    def signal(self, df):
        if len(df) < 10 or "volume" not in df.columns: return "HOLD"
        price = float(df["close"].iloc[-1])
        try:
            vwap = calc_vwap(df).iloc[-1]
            ema9 = calc_ema(df, 9).iloc[-1]
        except: return "HOLD"
        if pd.isna(vwap) or pd.isna(ema9): return "HOLD"
        if price > vwap and price > ema9 and ema9 > vwap: return "BUY"
        if price < vwap and price < ema9 and ema9 < vwap: return "SELL"
        return "HOLD"

# ── 8. VWAP + EMA9 + RSI ─────────────────────────────
class VwapEma9Rsi(BaseStrategy):
    name = "vwap_ema9_rsi"
    description = "VWAP + EMA9 + RSI Triple Confirm"
    def signal(self, df):
        if len(df) < 15 or "volume" not in df.columns: return "HOLD"
        try:
            price = float(df["close"].iloc[-1])
            vwap  = calc_vwap(df).iloc[-1]
            ema9  = calc_ema(df, 9).iloc[-1]
            rsi   = calc_rsi(df).iloc[-1]
        except: return "HOLD"
        if any(pd.isna(x) for x in [vwap, ema9, rsi]): return "HOLD"
        if price > vwap and ema9 > vwap and 45 < rsi < 70: return "BUY"
        if price < vwap and ema9 < vwap and 30 < rsi < 55: return "SELL"
        return "HOLD"

# ── 9. 5-Condition Scalper ───────────────────────────
class FiveConditionScalper(BaseStrategy):
    name = "five_condition_scalper"
    description = "5-ਸ਼ਰਤ Scalper (EMA+VWAP+RSI+Vol+Time)"
    def signal(self, df):
        if len(df) < 25 or "volume" not in df.columns: return "HOLD"
        try:
            close = df["close"].astype(float)
            vol   = df["volume"].astype(float)
            cur   = df.iloc[-1]
            try:
                ts = pd.to_datetime(cur["timestamp"])
                hm = ts.hour * 60 + ts.minute
                if not (9*60+30 <= hm <= 11*60): return "HOLD"
            except: pass
            ema9    = close.ewm(span=9, adjust=False).mean().iloc[-1]
            vwap    = calc_vwap(df).iloc[-1] if "volume" in df.columns else ema9
            delta   = close.diff()
            gain    = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
            loss    = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
            rsi     = (100 - 100/(1+gain/loss.replace(0,np.nan))).iloc[-1]
            vol_avg = vol.rolling(20).mean().iloc[-1]
            c, o, h, l = float(cur["close"]), float(cur["open"]), float(cur["high"]), float(cur["low"])
            v       = float(cur["volume"])
            if any(pd.isna(x) for x in [ema9, vwap, rsi, vol_avg]): return "HOLD"
            body = abs(c-o); rng = h-l
            body_r = body/rng if rng > 0 else 0
            if c>ema9 and ema9>vwap and rsi>55 and c>o and body_r>=0.5 and v>vol_avg*1.5: return "BUY"
            if c<ema9 and ema9<vwap and rsi<45 and c<o and v>vol_avg*1.5: return "SELL"
        except: pass
        return "HOLD"

# ── 10. ORB 9:15-9:30 ────────────────────────────────
class Orb915930(BaseStrategy):
    name = "orb_915_930"
    description = "Opening Range Breakout 9:15-9:30"
    def __init__(self, target_pts=35, vol_mult=1.5):
        self.target_pts = target_pts
        self.vol_mult   = vol_mult
        self.last_sl = self.last_target = self.last_entry = None
    def signal(self, df):
        if len(df) < 5: return "HOLD"
        try:
            d = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(d["timestamp"]):
                d["timestamp"] = pd.to_datetime(d["timestamp"])
            d["_tm"] = d["timestamp"].dt.strftime("%H:%M")
            orb  = d[(d["_tm"]>="09:15")&(d["_tm"]<="09:30")]
            post = d[d["_tm"]>"09:30"]
            if orb.empty or post.empty: return "HOLD"
            orb_h = orb["high"].max(); orb_l = orb["low"].min()
            avg_v = orb["volume"].mean()
            cur   = post.iloc[-1]
            c, v  = float(cur["close"]), float(cur["volume"])
            vol_ok = v >= avg_v * self.vol_mult
            if c > orb_h and vol_ok:
                self.last_sl=orb_l; self.last_target=orb_h+self.target_pts; self.last_entry=orb_h
                return "BUY"
            if c < orb_l and vol_ok:
                self.last_sl=orb_h; self.last_target=orb_l-self.target_pts; self.last_entry=orb_l
                return "SELL"
        except: pass
        return "HOLD"

# ── 11. Gap Fade ─────────────────────────────────────
class GapFade(BaseStrategy):
    name = "gap_fade"
    description = "Gap Fade — Open vs Prev Close"
    def __init__(self, gap_pct=0.5):
        self.gap_pct = gap_pct
    def signal(self, df):
        if len(df) < 2: return "HOLD"
        try:
            gap = calc_gap(df, threshold_pct=self.gap_pct)
            if gap["gap_up"]:   return "SELL"
            if gap["gap_down"]: return "BUY"
        except: pass
        return "HOLD"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. BankBees ML Scalper — FIXED (Live Balance + User QTY)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BankBeesMLScalper(BaseStrategy):
    name        = "bankbees_ml_scalper"
    description = "ML-based BANKBEES ETF Scalper (5-min candles)"

    def __init__(self, config: dict):
        model_path = config.get("bankbees", {}).get(
            "model_path", "/home/ubuntu/models/BANKBEES_5min_model.pkl"
        )
        self.capital      = config.get("bankbees", {}).get("capital",  100_000)
        self.risk_pct     = config.get("bankbees", {}).get("risk_pct", 0.5)
        self.scalper      = BankBeesScalper(model_path=model_path)
        self.in_trade     = False
        # ── User QTY (GUI ਤੋਂ) — default 10 ──────────────────────────
        self.user_qty     = 10

    def signal(self, df: pd.DataFrame) -> str:
        return "HOLD"

    def set_user_qty(self, qty: int):
        """GUI ਤੋਂ qty set ਕਰੋ"""
        self.user_qty = max(int(qty), 1)

    def run_bankbees(self, fyers) -> dict | None:
        if self.in_trade:
            return None

        df = get_bankbees_5min(fyers, days_back=5)
        if df.empty:
            return None

        # ── user_qty pass ਕਰੋ predict_signal ਨੂੰ ────────────────────
        signal = self.scalper.predict_signal(df, user_qty=self.user_qty)

        # ── ਜੇ user_qty=0 ਹੈ ਤਾਂ calc_qty already handle ਕਰ ਲੈਂਦਾ ──
        # ── ਇੱਥੇ extra override ਦੀ ਲੋੜ ਨਹੀਂ ──────────────────────────

        return signal

    def mark_trade_open(self):
        self.in_trade = True

    def mark_trade_closed(self):
        self.in_trade = False


# ── Registry ─────────────────────────────────────────
STRATEGIES = {
    "ema_crossover":          EmaCrossover,
    "rsi_reversal":           RsiReversal,
    "ema_rsi_combined":       EmaRsiCombined,
    "bollinger_breakout":     BollingerBreakout,
    "macd_crossover":         MacdCrossover,
    "ema_macd_combined":      EmaMacdCombined,
    "vwap_ema9":              VwapEma9,
    "vwap_ema9_rsi":          VwapEma9Rsi,
    "five_condition_scalper": FiveConditionScalper,
    "orb_915_930":            Orb915930,
    "gap_fade":               GapFade,
    "bankbees_ml_scalper":    BankBeesMLScalper,
}

STRATEGY_DESCRIPTIONS = {k: v.description for k, v in STRATEGIES.items()}


def get_strategy(config: dict) -> BaseStrategy:
    name = config["bot"].get("strategy", "ema_crossover")
    if name not in STRATEGIES:
        print(f"Strategy '{name}' ਨਹੀਂ ਮਿਲੀ — ema_crossover ਵਰਤਾਂਗੇ")
        name = "ema_crossover"
    cls = STRATEGIES[name]
    if name == "ema_crossover":
        return cls(fast=config["bot"].get("ema_fast", 9), slow=config["bot"].get("ema_slow", 21))
    if name == "ema_rsi_combined":
        return cls(fast=config["bot"].get("ema_fast", 9), slow=config["bot"].get("ema_slow", 21))
    if name == "ema_macd_combined":
        return cls(fast=config["bot"].get("ema_fast", 9), slow=config["bot"].get("ema_slow", 21))
    if name == "bankbees_ml_scalper":
        return cls(config=config)
    return cls()
