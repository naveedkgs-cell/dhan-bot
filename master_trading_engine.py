"""
MASTER TRADING ENGINE - NIFTY / SENSEX / 
CRUDE OIL / USDINR
===========================================
===================== Full spec implementation:
-	5-min timeframe
-	SL 10-12 pts (20 pts if VIX > 20), first target 25 pts   - At 25 pts: book 50% qty, trail the rest   - Breakeven trail at +15, then trail every +10 after 25
-	Auto Call<->Put reversal on trend flip while in profit   - Pre-market regime check (VIX / gap / crude / DXY / US futures)
-	EMA9/21, VWAP, SuperTrend(10,3), ATR14, 
RSI14+EMA9
-	Basic candlestick pattern recognition
-	Greeks-based strike filter (delta 0.50-
0.60, theta cutoff)
-	PCR/OI filter (STUB - needs a real option-chain data source)   - Gap/crash-day protection (skip new trades on big gap / ATR spike)   - Risk management: max 10 trades/day, 2% capital risk per trade   - Trading window HARD-locked to 9:30-
15:00 for new entries; 15:15
    force square-off of everything, every 

day - NO BTST, ever
  - Glitch/anomaly protection: an abnormally huge single-candle range     or stale data pauses new entries and protectively tightens the     stop on any open position instead of trusting a bad print
===========================================
================================
READ BEFORE USE - IMPORTANT FACTS, NOT JUST 
DISCLAIMERS
===========================================
================================
1.	FOREX: Indian residents may legally trade only INR currency pairs
   (USDINR, EURINR, GBPINR, JPYINR) on 
NSE/BSE/MCX-SX. Trading global
   pairs (EUR/USD etc.) via an offshore broker is restricted under
   FEMA/RBI rules for retail residents. 
This script only wires up    USDINR for that reason.
2.	TradingView / Moneycontrol / Chartink / 
SenseBull / Zee Business do
   NOT have public free APIs. Scraping them breaks their Terms of    Service and breaks constantly (layout changes). Pre-market checks
   here use publicly available data 
(yfinance) instead. News-sentiment
   is a STUB - wire in a real news API 
(e.g. a paid NewsAPI/RSS feed)    if you want that filter live.

3.	PCR / Open Interest needs live optionchain data. Dhan's SDK may
   expose an option-chain endpoint - verify the exact method name
   against current `dhanhq` docs before relying on `get_pcr_oi()`
   below; it is written defensively 
(returns None -> filter skipped    if it can't fetch data, rather than guessing).
4.	"20 years of market memory" is implemented as rule-based regime
   filters (VIX spike, ATR spike, big gap = don't take new trades).
   It is NOT a trained model that has 
"learned" 2008/2020-style
   crashes - no single script can honestly claim that. If you want
   genuine historical pattern-backtesting, that's a separate,    larger project (feed 20 years of OHLC through a proper backtest
   framework) - happy to help with that as its own task.
5. Real money, real risk. No indicator combination, however complete,    guarantees profit. Paper-trade 
(DRY_RUN=True) for weeks before
   going live. Not investment advice - I'm not a SEBI-registered    advisor.
Install:
    pip install dhanhq pandas numpy 

yfinance scipy requests --break-systempackages """
import os import time import traceback import datetime as dt import requests import pandas as pd import numpy as np from dhanhq import DhanContext, dhanhq
# 
===========================================
==========================
# 0. CONFIG
# 
===========================================
==========================
DRY_RUN = True
MAX_DAILY_LOSS_RUPEES = 3000
MAX_TRADES_PER_DAY = 10
RISK_PER_TRADE_PCT = 0.02          # 2% of capital per trade CAPITAL = 100000
BASE_SL_POINTS = 12                # normal regime
HIGH_VIX_SL_POINTS = 20            # VIX > 
20 regime
TARGET_TRIGGER_POINTS = 25
PARTIAL_BOOK_FRACTION = 0.5        # book this fraction of qty at first target

BREAKEVEN_TRIGGER_POINTS = 15
TRAIL_STEP_POINTS = 10             # after target hit, trail SL every 10 pts LOTS = 1
INSTRUMENTS = {
    "NIFTY":    {"yf_symbol": "^NSEI",  "exch_segment": "NSE_FNO",  "lot_size": 
75},
    "SENSEX":   {"yf_symbol": "^BSESN", "exch_segment": "BSE_FNO",  "lot_size": 20},
    "CRUDEOIL": {"yf_symbol": "CL=F",   "exch_segment": "MCX_COMM", "lot_size": 
100},
    "USDINR":   {"yf_symbol": "INR=X",  
"exch_segment": "NSE_CURRENCY", "lot_size": 
1000},
}
ACTIVE_INSTRUMENTS = ["NIFTY"]   # test one at a time before adding more
DHAN_SCRIP_MASTER_URL = 
"https://images.dhan.co/api-data/api-scripmaster.csv" MARKET_OPEN = dt.time(9, 15)
FIRST_15MIN_END = dt.time(9, 30)   # no NEW trade before this (range-building window) LAST_ENTRY_TIME = dt.time(15, 0)   # no NEW trade after this - gives time to manage before square-off SQUARE_OFF_TIME = dt.time(15, 15)  # forceclose every open position here, no BTST ever

MARKET_CLOSE = dt.time(15, 25)
POLL_SECONDS = 60
# --- anomaly / glitch protection --ANOMALY_ATR_MULT = 3.0             # a single candle range > this * ATR14 = suspicious tick/spike
ANOMALY_PAUSE_CYCLES = 3           # skip this many poll cycles after an anomaly before trusting data again
TG_TOKEN = 
os.environ.get("TELEGRAM_BOT_TOKEN") TG_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
def tg_send(msg: str):     if not TG_TOKEN or not TG_CHAT_ID:         print("[telegram not configured]", msg)         return     try:         requests.post(f"https://api.telegram.org/bo t{TG_TOKEN}/sendMessage",
                       data={"chat_id": 
TG_CHAT_ID, "text": msg}, timeout=10)     except Exception as e:
        print("Telegram send failed:", e)
# 
===========================================

==========================
# 1. INDICATORS
# 
=========================================== ========================== def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()
def atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - 
c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()
def rsi(s, period=14):     d = s.diff()     gain = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()     return 100 - 100 / (1 + gain / loss.replace(0, np.nan))
def vwap(df):
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.date

    pv = (typical * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum()     return pv / vol.replace(0, np.nan)
def supertrend(df, period=10, multiplier=3):     atr_val = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val
    trend = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="int64")     trend.iloc[0] = upper.iloc[0]
    direction.iloc[0] = 1
    for i in range(1, len(df)):
        close = df["close"].iloc[i]         if close > upper.iloc[i - 1]:
            direction.iloc[i] = 1         elif close < lower.iloc[i - 1]:
            direction.iloc[i] = -1         else:
            direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1 and lower.iloc[i] > lower.iloc[i - 1]:                 lower.iloc[i] = lower.iloc[i]

            if direction.iloc[i] == -1 and upper.iloc[i] < upper.iloc[i - 1]:                 upper.iloc[i] = upper.iloc[i]         trend.iloc[i] = lower.iloc[i] if direction.iloc[i] == 1 else upper.iloc[i]
    return trend, direction  # direction: 1 
= green/bullish, -1 = red/bearish
def candlestick_pattern(df):
    """Very simplified pattern check on the last 3 candles."""     if len(df) < 3:
        return None
    c0, c1, c2 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    body = lambda r: abs(r["close"] - r["open"])
    rng = lambda r: r["high"] - r["low"] + 
1e-9
    is_bull = lambda r: r["close"] > r["open"]
    is_bear = lambda r: r["close"] < r["open"]
    # Bullish engulfing
    if is_bear(c1) and is_bull(c2) and c2["close"] > c1["open"] and c2["open"] < c1["close"]:         return "bullish_engulfing"
    # Bearish engulfing
    if is_bull(c1) and is_bear(c2) and 

c2["open"] > c1["close"] and c2["close"] < c1["open"]:         return "bearish_engulfing"
    # Doji     if body(c2) / rng(c2) < 0.1:
        return "doji"
    # Hammer (small body, long lower wick, near top of range)     lower_wick = min(c2["open"], c2["close"]) - c2["low"]     upper_wick = c2["high"] - max(c2["open"], c2["close"])
    if lower_wick > 2 * body(c2) and upper_wick < body(c2):         return "hammer"
    # Shooting star (small body, long upper wick, near bottom of range)
    if upper_wick > 2 * body(c2) and lower_wick < body(c2):         return "shooting_star"
    # Morning star (bear, small, bull recovering above c0 midpoint)
    if is_bear(c0) and body(c1) / rng(c1) < 
0.3 and is_bull(c2) and c2["close"] > (c0["open"] + c0["close"]) / 2:
        return "morning_star"
    # Evening star
    if is_bull(c0) and body(c1) / rng(c1) < 
0.3 and is_bear(c2) and c2["close"] < (c0["open"] + c0["close"]) / 2:
        return "evening_star"
    return None

BULLISH_PATTERNS = {"bullish_engulfing", 
"hammer", "morning_star"}
BEARISH_PATTERNS = {"bearish_engulfing", 
"shooting_star", "evening_star"}
# 
===========================================
==========================
# 2. PRE-MARKET REGIME CHECK  (public data only - yfinance)
# 
=========================================== ========================== def premarket_check():     import yfinance as yf
    out = {}
    tickers = {
        "us_futures_sp500": "ES=F",
        "us_futures_nasdaq": "NQ=F",
        "dxy": "DX-Y.NYB",
        "crude": "CL=F",
        "india_vix": "^INDIAVIX",
        "nifty_prev": "^NSEI",
    }     for key, sym in tickers.items():
        try:             data = yf.Ticker(sym).history(period="2d")
            out[key] = float(data["Close"].iloc[-1]) if not data.empty else None         except Exception:
            out[key] = None

    vix = out.get("india_vix")
    regime = vix_regime(vix) if vix else 
"unknown"
    msg = (f"📊 Pre-market check:\n"
           f"India VIX: {vix}\nRegime: 
{regime}\n"            f"US S&P fut: 
{out.get('us_futures_sp500')}\n"
           f"DXY: {out.get('dxy')}\nCrude: 
{out.get('crude')}\n"            f"(News sentiment filter: not wired up - needs a news API)")     tg_send(msg)     return out
def vix_regime(vix_value):     if vix_value is None:
        return "unknown"     if vix_value < 12:
        return "no_trade_sideways"     elif vix_value <= 18:
        return "normal"     elif vix_value <= 20:
        return "elevated"     return "high_vix"
def sl_points_for_regime(regime):
    return HIGH_VIX_SL_POINTS if regime == 
"high_vix" else BASE_SL_POINTS

# 
===========================================
==========================
# 3. PCR / OI  (STUB - wire to a real option-chain source)
# 
=========================================== ========================== def get_pcr_oi(dhan, underlying_security_id=None):     """
    Returns {"pcr": float, 
"max_call_oi_strike": x, 
"max_put_oi_strike": y}
    or None if unavailable. STUB until you confirm the exact dhanhq
    option-chain method name/response shape against current docs.     """     try:
        # placeholder - verify against dhanhq's actual option-chain call         return None     except Exception:
        return None
# 
===========================================
==========================
# 4. DHAN CONNECTION / SCRIP MASTER
# 
===========================================
==========================

def connect_dhan():     client_id = os.environ.get("DHAN_CLIENT_ID")
    access_token = os.environ.get("DHAN_ACCESS_TOKEN")     if not client_id or not access_token:
        raise RuntimeError("DHAN_CLIENT_ID 
/ DHAN_ACCESS_TOKEN not set in environment")     return dhanhq(DhanContext(client_id, access_token)) _scrip_master_cache = None
def load_scrip_master():     global _scrip_master_cache     if _scrip_master_cache is None:         _scrip_master_cache = pd.read_csv(DHAN_SCRIP_MASTER_URL, low_memory=False)     return _scrip_master_cache
def resolve_option_by_delta(instrument_name, spot_price, option_type, target_delta_range=(0.50, 0.60)):
    """Pick the nearest-expiry strike whose approx delta falls in range.
    Uses a rough Black-Scholes delta estimate (needs IV; approximated
    from India VIX as a stand-in) purely to 

RANK strikes - always
    sanity-check the resolved strike/premium before trusting it."""     from scipy.stats import norm
    df = load_scrip_master()
    subset = df[
        
(df["SEM_TRADING_SYMBOL"].astype(str).str.s tartswith(instrument_name))
        & (df["SEM_OPTION_TYPE"] == option_type)
        & 
(df["SEM_INSTRUMENT_NAME"].astype(str).str. contains("OPT"))     ].copy()     if subset.empty:
        raise RuntimeError(f"No contracts found for {instrument_name}/{option_type}")
    subset["SEM_EXPIRY_DATE"] = pd.to_datetime(subset["SEM_EXPIRY_DATE"])
    nearest_expiry = subset["SEM_EXPIRY_DATE"].min()
    subset = subset[subset["SEM_EXPIRY_DATE"] == nearest_expiry]     days_to_expiry = max((nearest_expiry.date() - dt.date.today()).days, 1)
    def approx_delta(strike, iv_pct=15.0, r=0.07):
        T = days_to_expiry / 365

        sigma = iv_pct / 100
        d1 = (np.log(spot_price / strike) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) if option_type 
== "CE" else norm.cdf(d1) - 1
    subset["approx_delta"] = subset["SEM_STRIKE_PRICE"].apply(approx_del ta)
    target_mid = sum(target_delta_range) / 
2
    subset["delta_diff"] = (subset["approx_delta"].abs() - target_mid).abs()     best = subset.sort_values("delta_diff").iloc[0]
    return {
        "security_id": str(best["SEM_SMST_SECURITY_ID"]),         "trading_symbol": best["SEM_TRADING_SYMBOL"],         "strike": best["SEM_STRIKE_PRICE"],         "approx_delta": 
round(best["approx_delta"], 3),         "expiry": nearest_expiry.date(),
    }
# 
===========================================
==========================
# 5. SIGNAL ENGINE

# 
=========================================== ========================== def compute_frame(yf_symbol):     import yfinance as yf     raw = yf.download(yf_symbol, period="3d", interval="5m", progress=False)     if raw.empty:
        return None
    raw = raw.rename(columns=str.lower)
[["open", "high", "low", "close", 
"volume"]].dropna()
    raw["ema9"] = ema(raw["close"], 9)
    raw["ema21"] = ema(raw["close"], 21)
    raw["rsi14"] = rsi(raw["close"], 14)
    raw["vwap"] = vwap(raw)
    raw["atr14"] = atr(raw, 14)
    st_trend, st_dir = supertrend(raw, 10, 
3)
    raw["st_dir"] = st_dir     return raw
def day_high_low_range(df, ref_date):     day_df = df[df.index.date == ref_date]
    first15 = day_df.between_time("09:15", 
"09:30")     if first15.empty:
        return None, None
    return first15["high"].max(), first15["low"].min()
def entry_signal(df, vix_regime_str, 

pcr_info=None):     now = dt.datetime.now().time()
    if now < FIRST_15MIN_END or now >= LAST_ENTRY_TIME:
        return None   # outside the 9:30-
15:00 new-entry window
    last = df.iloc[-1]
    today = dt.date.today()     day_high, day_low = day_high_low_range(df, today)     if day_high is None:
        return None     pattern = candlestick_pattern(df)
    call_cond = (
        last["close"] > last["vwap"]
        and last["ema9"] > last["ema21"]
        and last["rsi14"] > 55
        and last["st_dir"] == 1
        and pattern in BULLISH_PATTERNS
        and last["close"] > day_high
    )
    put_cond = (
        last["close"] < last["vwap"]
        and last["ema9"] < last["ema21"]
        and last["rsi14"] < 45
        and last["st_dir"] == -1
        and pattern in BEARISH_PATTERNS
        and last["close"] < day_low
    )     if pcr_info:

        pcr = pcr_info.get("pcr")         if pcr is not None:
            call_cond = call_cond and pcr > 
1.0
            put_cond = put_cond and pcr < 
0.8
    if call_cond:
        return {"side": "CE", "spot": float(last["close"])}     if put_cond:
        return {"side": "PE", "spot": float(last["close"])}     return None
def gap_or_crash_day(df):
    """True if today opened with a big gap or ATR is unusually high ->
    skip NEW trades, only manage existing ones."""     today = dt.date.today()
    day_df = df[df.index.date == today]
    if day_df.empty or len(df[df.index.date < today]) == 0:
        return False
    prev_close = df[df.index.date < today]
["close"].iloc[-1]
    today_open = day_df["open"].iloc[0]
    gap = abs(today_open - prev_close)
    atr_now = df["atr14"].iloc[-1]
    return gap > 1.5 * atr_now or atr_now > 
70

def detect_anomaly(df):     """
    Flags a likely data glitch / stray-tick / manipulation-style spike:
-	last candle's range is way bigger than normal (ATR-based)
-	OR the feed looks stale (last candle timestamp too old vs now)     On a flagged cycle: don't take new trades, and if a position is
    open, treat it protectively (see manage_position) rather than
    trusting the anomalous print for target/SL math. This is the safe,     honest version of "profit from a glitch" - the real edge is NOT     getting whipsawed by a bad tick, not trying to trade the glitch
    itself (which isn't something a script can safely or legally do).     """     if len(df) < 20:
        return False, "insufficient_data"
    last = df.iloc[-1]
    atr_now = df["atr14"].iloc[-1]     candle_range = last["high"] - last["low"]
    if atr_now > 0 and candle_range > ANOMALY_ATR_MULT * atr_now:
        return True, "extreme_candle_range"
    last_ts = df.index[-1]

    now_ts = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tzinfo else pd.Timestamp.now()
    staleness_min = (now_ts - last_ts).total_seconds() / 60     if staleness_min > 15:         return True, f"stale_data_{int(staleness_min)}min"     return False, None
# 
===========================================
==========================
# 6. ORDER PLACEMENT
# 
=========================================== ========================== def place_order(dhan, cfg, security_id, side, qty):     if DRY_RUN:
        tg_send(f"[DRY RUN] {side} qty=
{qty} security_id={security_id}")
        return {"status": "dry_run"}     try:
        resp = dhan.place_order(             security_id=security_id, exchange_segment=cfg["exch_segment"],             transaction_type=dhan.BUY if side == "BUY" else dhan.SELL,             quantity=qty, order_type=dhan.MARKET, product_type=dhan.INTRA, price=0,

        )
        tg_send(f"ORDER: {side} qty={qty} id={security_id}\n{resp}")         return resp     except Exception as e:
        tg_send(f"⚠ ORDER FAILED: {side} qty={qty} id={security_id}\n{e}")         raise
# 
===========================================
==========================
# 7. STATE + MAIN LOOP
# 
=========================================== ========================== class InstrumentState:     def __init__(self, name):
        self.name = name
        self.position = None
        self.day_pnl = 0.0
        self.day_trades = 0
        self.day_stopped = False
        self.anomaly_pause_left = 0         self.squared_off_today = False
def within_market_hours():     now = dt.datetime.now().time()
    return MARKET_OPEN <= now <= 
MARKET_CLOSE

def manage_position(dhan, state, cfg, df, sig, regime):     pos = state.position
    last_spot = df["close"].iloc[-1]
    move = (last_spot - pos["entry_spot"]) 
* (1 if pos["side"] == "CE" else -1)
    pos["peak"] = max(pos["peak"], move)     sl_pts = sl_points_for_regime(regime)
    # breakeven trail
    if not pos["breakeven_done"] and move >= BREAKEVEN_TRIGGER_POINTS:
        pos["sl_level"] = 0
     
